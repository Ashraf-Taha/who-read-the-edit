"""E2c: the main repair experiment: every policy, three placements of the cached question, three suffix lengths.

Context P + E + S, then new input Q (the assistant turn opener). The cached question sits in S.
Layouts vary the edited document's position (2nd, 5th or 8th of 10 documents) and the
question's place in S (right after E, in the middle, at the end). E is a supporting
document or a distractor.

Directions
    delete   E is removed; the cache holds S as encoded with E present.
    replace  E is replaced by E', in which every occurrence of the answer string is swapped
             for another answer from the dataset (a counterfactual edit that should change the
             answer); the cache holds S as encoded with E. Only cases whose answer occurs in E.

Policies rank the tokens of S; the top fraction is recomputed in one pass that reads every
other cell from the cache. Scores (higher = recompute first), all signals averaged over heads:
    boundary           tokens right after the edit
    epic               the first tokens of every segment of S, in turn (EPIC's selection)
    tail               tokens right before the new input
    random             seeded uniform
    probe1             deviation at the first contextual layer (CacheBlend's first selection step)
    reader             attention of Q on the token, mean over layers (ProphetKV's practical score)
    exposure           attention the token put on E when encoded, mean over layers
    question_exposure  the cached question first, then by exposure
    sum                rank of exposure + rank of reader
    rrf                reciprocal-rank fusion of exposure and reader (k = 60)
    union              alternate between the exposure and the reader rankings
    bridge             exposure x reader (layer means multiplied)
    bridge_layer       sum over layers of exposure x reader at that layer
    bridge_below       sum over layers of reader at that layer x exposure accumulated below it
    oracle             true relative cell deviation from the cold pass, mean over layers
    oracle_reader      oracle x reader (ProphetKV's ideal value function; needs the full recompute)
Also reported: question_only, recomputing just the cached question (its own budget).

Metrics against the cold pass over the edited context: KL(p_cold || p_reuse) at the first reply
token; mean KL and top-1 agreement over the first 16 tokens of the cold-pass reply, teacher-forced;
greedy 16-token answers for some policies. A noise floor per case: the same metrics for the cold pass
over a context with two non-supporting documents swapped.

Usage: e2c_ablation.py <model-key> <supporting-per-cell> <distractor-per-cell> [tag] [device]
"""
from __future__ import annotations

import math
import os
import random
import sys
import zlib

import torch

from common import MODELS, best_device, kl_rows, normalise, resume, save
from e2_healing import decode, relative_deviation, shallow
from mutablekv import LM, Cells, greedy, heal
from workloads import READER_PLACES, Case, reader_cases

BUDGETS = (0.05, 0.10, 0.20)
LEADS = (1, 4, 7)
REPLY = 16
DELETE_POLICIES = ["boundary", "epic", "tail", "random", "probe1", "reader", "exposure", "question_exposure",
                   "sum", "rrf", "union", "bridge", "bridge_layer", "bridge_below", "oracle", "oracle_reader"]
REPLACE_POLICIES = ["boundary", "tail", "reader", "exposure", "question_exposure", "bridge", "oracle_reader"]
DECODED = ("boundary", "tail", "reader", "exposure", "question_exposure", "sum", "bridge", "oracle_reader")


def budget_count(fraction: float, n: int) -> int:
    return min(n, max(1, math.ceil(round(fraction * n, 9))))


def ranks(x: torch.Tensor) -> torch.Tensor:
    """0 for the largest value."""
    return x.argsort(descending=True).argsort().float()


def interleave(*orders: torch.Tensor) -> torch.Tensor:
    seen, priority = set(), []
    for column in zip(*[o.tolist() for o in orders]):
        for index in column:
            if index not in seen:
                seen.add(index)
                priority.append(index)
    score = torch.zeros(len(priority))
    score[torch.tensor(priority)] = -torch.arange(len(priority), dtype=torch.float)
    return score


class Reader:
    """Runs the new input (and optionally a teacher-forced reply) over a given state of the context."""

    def __init__(self, lm: LM, context: torch.Tensor, Q: torch.Tensor, reply: torch.Tensor):
        self.lm, self.n, self.nQ = lm, context.numel(), Q.numel()
        self.tokens = torch.cat((context, Q, reply))
        self.pos = torch.arange(self.tokens.numel())
        self.fresh = self.pos >= self.n
        self.span = slice(self.nQ - 1, self.nQ - 1 + max(1, reply.numel()))

    def logits(self, cells: Cells, groups: torch.Tensor | None = None):
        result = self.lm.run(self.tokens, self.pos, self.fresh, cells, groups=groups)
        return self.lm.logits(result.hidden[self.span]).float().cpu(), result.mass


def compare(truth: torch.Tensor, other: torch.Tensor, reply: torch.Tensor) -> dict:
    divergence = kl_rows(truth, other)
    targets = reply if reply.numel() else truth.argmax(-1)
    return {"kl_first": divergence[0].item(), "kl_tf": divergence.mean().item(),
            "top1_first": bool(truth[0].argmax() == other[0].argmax()),
            "top1_tf": (other.argmax(-1)[: targets.numel()] == targets).float().mean().item()}


def policy_scores(lm: LM, case: Case, S_len: int, exposure_by_layer: torch.Tensor, reader_by_layer: torch.Tensor,
                  probe: torch.Tensor | None, oracle: torch.Tensor) -> dict[str, torch.Tensor]:
    order = torch.arange(S_len).float()
    starts, lengths = [], []
    offset = 0
    for index in range(case.edit + 1, case.fresh_from):
        starts.append(offset)
        lengths.append(case.segments[index].numel())
        offset += lengths[-1]
    within, segment_of = torch.zeros(S_len), torch.zeros(S_len)
    for number, (start, length) in enumerate(zip(starts, lengths)):
        within[start:start + length] = torch.arange(length).float()
        segment_of[start:start + length] = number
    q_start = sum(case.segments[i].numel() for i in range(case.edit + 1, case.meta["question_segment"]))
    q_end = q_start + case.segments[case.meta["question_segment"]].numel()
    in_question = torch.zeros(S_len)
    in_question[q_start:q_end] = 1.0

    exposure, reader = exposure_by_layer.mean(0), reader_by_layer.mean(0)
    below = torch.cumsum(exposure_by_layer, 0) - exposure_by_layer
    scores = {
        "boundary": -order,
        "epic": -within - 1e-3 * segment_of,
        "tail": order,
        "random": torch.rand(S_len, generator=torch.Generator().manual_seed(zlib.crc32(case.name.encode()))),
        "reader": reader,
        "exposure": exposure,
        "question_exposure": 1e3 * in_question + exposure,
        "sum": -(ranks(exposure) + ranks(reader)),
        "rrf": 1 / (60 + ranks(exposure)) + 1 / (60 + ranks(reader)),
        "union": interleave(exposure.argsort(descending=True), reader.argsort(descending=True)),
        "bridge": exposure * reader,
        "bridge_layer": (exposure_by_layer * reader_by_layer).sum(0),
        "bridge_below": (below * reader_by_layer).sum(0),
        "oracle": oracle,
        "oracle_reader": oracle * reader,
    }
    if probe is not None:
        scores["probe1"] = probe
    return scores, (q_start, q_end)


def repair_all(lm: LM, context: torch.Tensor, start: int, stale: Cells, reader: Reader, truth_tf: torch.Tensor,
               reply: torch.Tensor, scores: dict, policies: list[str], question: tuple[int, int]) -> tuple[dict, dict]:
    n, S_len = context.numel(), context.numel() - start
    pos = torch.arange(n)
    results, answers = {}, {}
    for name in policies:
        results[name] = {}
        for budget in BUDGETS:
            top = scores[name].topk(budget_count(budget, S_len)).indices
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[start + top] = True
            healed = heal(lm, context, pos, stale, chosen)
            results[name][str(budget)] = compare(truth_tf, reader.logits(healed)[0], reply)
            if budget == 0.10 and name in DECODED:
                answers[name] = decode(lm, reader.tokens[: reader.n + reader.nQ], reader.pos[: reader.n + reader.nQ], healed)
    chosen = torch.zeros(n, dtype=torch.bool)
    chosen[start + question[0]: start + question[1]] = True
    healed = heal(lm, context, pos, stale, chosen)
    results["question_only"] = {"own": {**compare(truth_tf, reader.logits(healed)[0], reply), "tokens": question[1] - question[0]}}
    return results, answers


def reader_attention(mass: torch.Tensor, start: int, n: int) -> torch.Tensor:
    """Attention of the new input on S, per layer, averaged over heads (done in run) and over the input tokens."""
    return mass[:, :, start:n].mean(1)


def substitute_for(case: Case, pool: list[str]) -> str | None:
    answer, text = case.meta["answer"], case.meta["edit_text"]
    if len(answer) < 4 or answer.lower() in ("yes", "no") or answer not in text:
        return None
    rng = random.Random(case.name)
    words = len(answer.split())
    options = [a for a in pool if a != answer and len(a.split()) == words and a.lower() not in ("yes", "no") and len(a) >= 4]
    return rng.choice(options) if options else None


def reply_of(lm: LM, context: torch.Tensor, Q: torch.Tensor, cells: Cells) -> tuple[torch.Tensor, str]:
    """Greedy reply token ids (up to REPLY) and the answer string as `decode` would give it."""
    stop = {lm.tok.eos_token_id, lm.tok.convert_tokens_to_ids("<|im_end|>")}
    tokens = torch.cat((context, Q))
    ids = greedy(lm, tokens, torch.arange(tokens.numel()), cells, REPLY, stop)
    return torch.tensor(ids, dtype=torch.long), lm.tok.decode(ids).strip().split("\n")[0]


def analyse(lm: LM, case: Case, pool: list[str]) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    stop = {lm.tok.eos_token_id, lm.tok.convert_tokens_to_ids("<|im_end|>")}
    with_e = torch.cat((P, E, S))
    groups = torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long()
    original = lm.cold(with_e, groups=groups)
    exposure_by_layer = original.mass[:, nP + nE:, 1]
    cP, cE, with_S = original.cells.take(slice(0, nP)), original.cells.take(slice(nP, nP + nE)), original.cells.take(slice(nP + nE, None))
    del original
    out = {"name": case.name, **{k: v for k, v in case.meta.items() if k != "edit_text"}, "nP": nP, "nE": nE, "nS": nS, "nQ": Q.numel()}

    # delete
    context, start = torch.cat((P, S)), nP
    truth = Cells.cat([cP, lm.run(context, torch.arange(nP + nS), torch.arange(nP + nS) >= nP, cP).cells])
    stale = Cells.cat([cP, with_S])
    reply, truth_answer = reply_of(lm, context, Q, truth)
    reader = Reader(lm, context, Q, reply)
    truth_tf, _ = reader.logits(truth)
    stale_tf, mass = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
    reader_by_layer = reader_attention(mass[:, : Q.numel()], start, nP + nS)
    probe_cells = lm.run(context, torch.arange(nP + nS), torch.arange(nP + nS) >= nP, shallow(cP), depth=2).cells
    probe = relative_deviation(lm, shallow(with_S), probe_cells)[1]
    oracle = relative_deviation(lm, with_S, truth.take(slice(nP, None))).mean(0)
    scores, question = policy_scores(lm, case, nS, exposure_by_layer, reader_by_layer, probe, oracle)
    results, answers = repair_all(lm, context, start, stale, reader, truth_tf, reply, scores, DELETE_POLICIES, question)
    results["stale"] = {"0": compare(truth_tf, stale_tf, reply)}
    answers["truth"] = truth_answer
    answers["stale"] = decode(lm, torch.cat((context, Q)), torch.arange(nP + nS + Q.numel()), stale)
    bridge = scores["bridge"]
    out["delete"] = {"results": results, "answers": answers, "reply_tokens": reply.numel(),
                     "bridge_mass": bridge.sum().item(), "exposure_mass": scores["exposure"].sum().item(),
                     "reader_mass": scores["reader"].sum().item(), "question_tokens": question[1] - question[0]}

    # noise floor: swap the first two non-supporting documents of the edited context
    titles = [t for i, t in enumerate(case.meta["segment_titles"][: case.fresh_from]) if i != case.edit]
    segments = [s for i, s in enumerate(case.segments[: case.fresh_from]) if i != case.edit]
    swappable = [i for i, t in enumerate(titles) if t is not None and t not in case.meta["gold_titles"]]
    if len(swappable) >= 2:
        a, b = swappable[:2]
        segments[a], segments[b] = segments[b], segments[a]
        swapped = Reader(lm, torch.cat(segments), Q, reply)
        out["noise_floor"] = compare(truth_tf, swapped.logits(lm.cold(torch.cat(segments)).cells)[0], reply)

    # replace
    substitute = substitute_for(case, pool)
    if substitute is not None:
        E2 = lm.encode(case.meta["edit_text"].replace(case.meta["answer"], substitute))
        nE2 = E2.numel()
        context = torch.cat((P, E2, S))
        start = nP + nE2
        n = context.numel()
        cE2 = lm.run(torch.cat((P, E2)), torch.arange(nP + nE2), torch.arange(nP + nE2) >= nP, cP).cells
        truth = Cells.cat([cP, cE2, lm.run(context, torch.arange(n), torch.arange(n) >= start, Cells.cat([cP, cE2])).cells])
        stale = Cells.cat([cP, cE2, with_S])
        ids, truth_answer = reply_of(lm, context, Q, truth)
        reader = Reader(lm, context, Q, ids)
        truth_tf, _ = reader.logits(truth)
        stale_tf, mass = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
        reader_by_layer = reader_attention(mass[:, : Q.numel()], start, n)
        oracle = relative_deviation(lm, with_S, truth.take(slice(start, None))).mean(0)
        scores, question = policy_scores(lm, case, nS, exposure_by_layer, reader_by_layer, None, oracle)
        results, answers = repair_all(lm, context, start, stale, reader, truth_tf, ids, scores, REPLACE_POLICIES, question)
        results["stale"] = {"0": compare(truth_tf, stale_tf, ids)}
        answers["truth"] = truth_answer
        answers["stale"] = decode(lm, torch.cat((context, Q)), torch.arange(n + Q.numel()), stale)
        follows = {k: normalise(substitute) in normalise(v) for k, v in answers.items()}
        keeps_old = {k: normalise(case.meta["answer"]) in normalise(v) for k, v in answers.items()}
        out["replace"] = {"substitute": substitute, "results": results, "answers": answers, "follows_edit": follows, "keeps_old": keeps_old}
    return out


def build_cases(lm: LM, supporting: int, distractor: int, **options) -> list[Case]:
    cases = []
    for lead in LEADS:
        for place in READER_PLACES:
            for kind, count in (("gold", supporting), ("distractor", distractor)):
                for case in reader_cases(lm, count, kind, place, seed=1000 + lead, lead=lead, **options):
                    case.name += f"/lead{lead}"
                    cases.append(case)
    return cases


def main() -> None:
    from datasets import load_dataset

    key, supporting, distractor = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    tag = sys.argv[4] if len(sys.argv) > 4 else "main"
    device = sys.argv[5] if len(sys.argv) > 5 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    pool = sorted(set(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")["answer"]))
    cases = build_cases(lm, supporting, distractor)
    stem = f"e2c_ablation_{key}_{tag}"
    settings = {"model": MODELS[key], "budgets": list(BUDGETS), "leads": list(LEADS), "delete_policies": DELETE_POLICIES,
                "replace_policies": REPLACE_POLICIES, "reply": REPLY}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}", flush=True)
    limit = int(os.environ.get("E2C_MAX_CASES", "0")) or len(cases)
    for case in cases:
        if case.name in done:
            continue
        if len(rows) >= limit:
            break
        rows.append(analyse(lm, case, pool))
        if len(rows) % 5 == 0:
            save(stem, {**settings, "rows": rows})
            print(f"{len(rows)} cases", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
