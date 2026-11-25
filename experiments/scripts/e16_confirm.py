"""E16: preregistered confirmation on fresh questions (response to review 2).

Hypotheses fixed before the run (also logged in PROGRESS.md section 60):
    H1  At 10% of the suffix, rank fusion (RRF) and the bridge score each leave lower KL than reader
        attention alone, on every model.
    H2  With the edited document 8th of ten (short suffix), the found reader segment first, then
        exposure, leaves lower KL than RRF; with it 2nd or 5th (long suffix), RRF leaves lower KL.
    H3  With two short notes in the suffix (variant "decoy"), the most-read-per-token rule still finds
        the question in most cases and the shortest-segment rule does not, and recomputing the found
        segment first leaves lower KL than recomputing the shortest one first.
    H4  Without repair, the stale state gives the deleted document's first sentence a higher
        log-probability than the re-prefill does (it leaks), and a 10% repair aimed at the answer
        removes only part of that excess.

Cases are E2c's layouts on new questions: supporting-document deletions, the edited document 2nd,
5th or 8th of ten (lead 1, 4, 7), the cached question right after it, in the middle or at the end,
12 questions per lead (seed 16000 + lead, disjoint from every earlier run's seeds).

Variants
    plain   as E2c; adds the leakage probe
    decoy   two short notes (about 8 tokens) in the suffix: one right after the edited document, one
            at the end of the suffix, so the question is no longer the shortest stale segment
            (only 10%)
    long    the system prompt asks for the answer and a two- or three-sentence explanation; the
            teacher-forced reply is up to 64 tokens (only 10%, fewer policies)

Policies (priority over the suffix; budgets 5%, 10%, 20% of the suffix and 16, 32, 64 tokens):
    reader            attention of the new input, mean over layers and heads (ProphetKV's practical score)
    exposure          attention the token put on the deleted document when encoded, mean over layers
    bridge            exposure x reader
    rrf               reciprocal-rank fusion of exposure and reader (k = 60)
    question_exposure the cached question first (role given), then exposure
    density_exposure  the stale segment with the most reader attention per token first, then exposure
    density_rrf       the same segment first, then RRF
    shortest_exposure the shortest stale segment first, then exposure
    mass_exposure     the stale segment with the most total reader attention first, then exposure
    oracle_reader     true cell deviation x reader (ProphetKV's ideal; needs the re-prefill)
Also: the question alone (its own length) and the bridge score and RRF at that same token count.

Metric: KL(p_cold || p_repaired) per position of the teacher-forced cold-pass reply (stored per
position, so the first position can be excluded later), top-1 agreement per position.

Usage: e16_confirm.py <model-key> <questions-per-lead> <variant> [tag] [device]
"""
from __future__ import annotations

import math
import sys

import torch

from common import MODELS, best_device, kl_rows, resume, save
from e2c_ablation import Reader, ranks, reader_attention
from e2d_reader_first import first
from e9_scale import deviation
from mutablekv import LM, Cells, greedy, heal
from workloads import READER_PLACES, chat_frame, reader_cases

LEADS = (1, 4, 7)
FRACTIONS = (0.05, 0.10, 0.20)
TOKENS = (16, 32, 64)
POLICIES = ("reader", "exposure", "bridge", "rrf", "question_exposure", "density_exposure", "density_rrf",
            "shortest_exposure", "mass_exposure", "oracle_reader")
LONG_POLICIES = ("reader", "exposure", "bridge", "rrf", "density_exposure", "oracle_reader")
LEAK_POLICIES = ("reader", "bridge", "rrf", "density_exposure")
DECOYS = ("Note: from a web search.\n\n", "Note: may be incomplete.\n\n")
LONG_SYSTEM = ("You answer questions using the documents provided. Give the answer, then explain it "
               "in two or three sentences.")


def budgets(variant: str, nS: int) -> dict[str, int]:
    if variant in ("long", "decoy"):
        return {"0.1": min(nS, max(1, math.ceil(round(0.10 * nS, 9))))}
    out = {str(f): min(nS, max(1, math.ceil(round(f * nS, 9)))) for f in FRACTIONS}
    out.update({f"t{t}": min(nS, t) for t in TOKENS})
    return out


def with_decoys(lm: LM, case):
    """Insert the two notes: right after the edited document and at the end of the suffix."""
    a, b = (lm.encode(text) for text in DECOYS)
    segments = list(case.segments)
    segments.insert(case.fresh_from, b)
    segments.insert(case.edit + 1, a)
    titles = list(case.meta["segment_titles"])
    titles.insert(case.fresh_from, None)
    titles.insert(case.edit + 1, None)
    case.segments, case.fresh_from = segments, case.fresh_from + 2
    case.meta = {**case.meta, "question_segment": case.meta["question_segment"] + 1, "segment_titles": titles}
    return case


def reply_tokens(lm: LM, context: torch.Tensor, Q: torch.Tensor, cells: Cells, length: int) -> torch.Tensor:
    stop = {lm.tok.eos_token_id, lm.tok.convert_tokens_to_ids("<|im_end|>")}
    tokens = torch.cat((context, Q))
    return torch.tensor(greedy(lm, tokens, torch.arange(tokens.numel()), cells, length, stop), dtype=torch.long)


def measure(truth: torch.Tensor, other: torch.Tensor, reply: torch.Tensor) -> dict:
    divergence = kl_rows(truth, other)
    targets = reply if reply.numel() else truth.argmax(-1)
    agree = (other.argmax(-1)[: targets.numel()] == targets).float()
    return {"kl": [round(x, 7) for x in divergence.tolist()], "top1": [int(x) for x in agree.tolist()]}


def leak_probe(lm: LM, title: str, sentence: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The new input that asks to quote the deleted document, and the first sentence it should give."""
    _, tail = chat_frame(lm)
    probe = lm.encode(f"Quote the first sentence of the document titled \"{title}\".{tail}")
    return probe, lm.encode(sentence)[:48]


def logprob(lm: LM, context: torch.Tensor, cells: Cells, probe: torch.Tensor, target: torch.Tensor) -> float:
    """Mean log-probability per token of `target` after `probe`, reading `cells` for the context."""
    tokens = torch.cat((context, probe, target))
    n = context.numel()
    positions = torch.arange(tokens.numel())
    hidden = lm.run(tokens, positions, positions >= n, cells).hidden
    logits = lm.logits(hidden[probe.numel() - 1: probe.numel() - 1 + target.numel()]).float().cpu()
    return torch.log_softmax(logits, -1).gather(1, target[:, None]).mean().item()


def analyse(lm: LM, case, variant: str) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    groups = torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long()
    original = lm.cold(torch.cat((P, E, S)), groups=groups)
    exposure = original.mass[:, nP + nE:, 1].mean(0)
    present = original.cells
    cP, with_S = original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))
    del original

    context, start, n = torch.cat((P, S)), nP, nP + nS
    pos = torch.arange(n)
    truth = Cells.cat([cP, lm.run(context, pos, pos >= nP, cP).cells])
    stale = Cells.cat([cP, with_S])
    reply = reply_tokens(lm, context, Q, truth, 64 if variant == "long" else 16)
    reader = Reader(lm, context, Q, reply)
    truth_tf, _ = reader.logits(truth)
    stale_tf, mass = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
    reader_score = reader_attention(mass[:, : Q.numel()], start, n).mean(0)
    oracle = deviation(lm, with_S, truth.take(slice(nP, None))).mean(0)

    lengths = [case.segments[i].numel() for i in range(case.edit + 1, case.fresh_from)]
    starts = [sum(lengths[:j]) for j in range(len(lengths))]
    question = case.meta["question_segment"] - (case.edit + 1)
    mask = lambda j: (torch.arange(nS) >= starts[j]) & (torch.arange(nS) < starts[j] + lengths[j])
    segment_mean = [reader_score[s:s + l].mean().item() for s, l in zip(starts, lengths)]
    segment_sum = [reader_score[s:s + l].sum().item() for s, l in zip(starts, lengths)]
    picks = {"density": max(range(len(lengths)), key=lambda j: segment_mean[j]),
             "shortest": min(range(len(lengths)), key=lambda j: (lengths[j], j)),
             "mass": max(range(len(lengths)), key=lambda j: segment_sum[j])}
    rrf = 1 / (60 + ranks(exposure)) + 1 / (60 + ranks(reader_score))
    scores = {"reader": reader_score, "exposure": exposure, "bridge": exposure * reader_score, "rrf": rrf,
              "question_exposure": first(mask(question), exposure), "density_exposure": first(mask(picks["density"]), exposure),
              "density_rrf": first(mask(picks["density"]), rrf), "shortest_exposure": first(mask(picks["shortest"]), exposure),
              "mass_exposure": first(mask(picks["mass"]), exposure), "oracle_reader": oracle * reader_score}

    out = {"name": case.name, "variant": variant, **{k: v for k, v in case.meta.items() if k != "edit_text"},
           "nP": nP, "nE": nE, "nS": nS, "nQ": Q.numel(), "reply_tokens": reply.numel(), "segment_lengths": lengths,
           "question_index": question, "picks": picks, "picks_question": {k: v == question for k, v in picks.items()},
           "question_reader_share": segment_sum[question] / sum(segment_sum),
           "results": {"stale": measure(truth_tf, stale_tf, reply)}}
    healed_at_ten, seen = {}, {}
    for name in (LONG_POLICIES if variant == "long" else POLICIES):
        out["results"][name] = {}
        for label, count in budgets(variant, nS).items():
            top = scores[name].topk(count).indices
            key = tuple(sorted(top.tolist()))
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[start + top] = True
            if key not in seen:
                seen[key] = measure(truth_tf, reader.logits(heal(lm, context, pos, stale, chosen))[0], reply)
            out["results"][name][label] = seen[key]
            if label == "0.1" and name in LEAK_POLICIES and variant == "plain":
                healed_at_ten[name] = heal(lm, context, pos, stale, chosen)
    q_count = lengths[question]
    chosen = torch.zeros(n, dtype=torch.bool)
    chosen[start + starts[question]: start + starts[question] + q_count] = True
    out["results"]["question_only"] = measure(truth_tf, reader.logits(heal(lm, context, pos, stale, chosen))[0], reply)
    for name in ("bridge", "rrf"):
        chosen = torch.zeros(n, dtype=torch.bool)
        chosen[start + scores[name].topk(q_count).indices] = True
        out["results"][f"{name}@question"] = measure(truth_tf, reader.logits(heal(lm, context, pos, stale, chosen))[0], reply)

    if variant == "plain":
        text = case.meta["edit_text"]
        title, body = text.split("\n", 1)
        sentence = body.strip().split(". ")[0].strip()
        sentence = sentence if sentence.endswith(".") else sentence + "."
        probe, target = leak_probe(lm, title.removeprefix("Title: ").strip(), sentence)
        with_e = torch.cat((P, E, S))
        leak = {"present": logprob(lm, with_e, present, probe, target), "truth": logprob(lm, context, truth, probe, target),
                "stale": logprob(lm, context, stale, probe, target), "target_tokens": target.numel()}
        for name, healed in healed_at_ten.items():
            leak[name] = logprob(lm, context, healed, probe, target)
        out["leak"] = leak
    return out


def main() -> None:
    key, per_lead, variant = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    tag = sys.argv[4] if len(sys.argv) > 4 else "main"
    device = sys.argv[5] if len(sys.argv) > 5 else best_device()
    if variant not in ("plain", "decoy", "long"):
        raise SystemExit(f"unknown variant {variant!r}")
    lm = LM(MODELS[key], device, torch.float32)
    if variant == "long":
        import workloads
        workloads.SYSTEM = LONG_SYSTEM
    cases = []
    for lead in LEADS:
        for place in READER_PLACES:
            for case in reader_cases(lm, per_lead, "gold", place, seed=16_000 + lead, lead=lead):
                case.name += f"/lead{lead}"
                cases.append(with_decoys(lm, case) if variant == "decoy" else case)
    stem = f"e16_confirm_{key}_{variant}_{tag}"
    settings = {"model": MODELS[key], "variant": variant, "leads": list(LEADS), "fractions": list(FRACTIONS),
                "tokens": list(TOKENS), "policies": list(LONG_POLICIES if variant == "long" else POLICIES)}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}", flush=True)
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case, variant))
        if len(rows) % 6 == 0:
            save(stem, {**settings, "rows": rows})
            print(f"{len(rows)} cases", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
