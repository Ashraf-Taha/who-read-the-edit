"""E2d: reader-first repair. Can the reader segment be found without knowing the prompt's roles?

Same supporting-document cases as E2c (edited document 2nd, 5th or 8th of ten; cached question
right after it, in the middle or at the end). E2c found that recomputing the cached question first
and then the tokens with the most recorded exposure beats every policy that ignores the prompt's
structure. Here the question segment is either given (as in E2c) or found from reader attention
alone: the suffix segment on which the new input puts the most attention per token. Policies, at
5%, 10% and 20% of the suffix:

    question_exposure        the cached question first, then recorded exposure (E2c's)
    question_bridge          the cached question first, then the bridge score
    segment_exposure         the segment with the most reader attention per token first, then exposure
    segment_bridge           the same segment first, then the bridge score
    density                  whole segments in order of mean bridge score per token while they fit,
                             then the bridge score for what is left of the budget
    bridge                   the bridge score alone (E2c's)

Records which segment reader attention picks and whether it is the question.

Usage: e2d_reader_first.py <model-key> <supporting-per-cell> [tag] [device]
"""
from __future__ import annotations

import sys

import torch

from common import MODELS, best_device, resume, save
from e2_healing import decode
from e2c_ablation import BUDGETS, Reader, budget_count, build_cases, compare, reader_attention, reply_of
from mutablekv import LM, Cells, heal

POLICIES = ("question_exposure", "question_bridge", "segment_exposure", "segment_bridge", "density", "bridge")
DECODED = ("question_exposure", "segment_exposure", "segment_bridge", "bridge")


def first(mask: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Priority that takes every token of `mask` before any other, each group ordered by `score`."""
    return mask.float() * (score.max() - score.min() + 1.0) + score


def density_choice(starts: list[int], lengths: list[int], bridge: torch.Tensor, count: int) -> torch.Tensor:
    chosen = torch.zeros(bridge.numel(), dtype=torch.bool)
    order = sorted(range(len(starts)), key=lambda s: -bridge[starts[s]:starts[s] + lengths[s]].mean().item())
    for s in order:
        if int(chosen.sum()) + lengths[s] <= count:
            chosen[starts[s]:starts[s] + lengths[s]] = True
    rest = count - int(chosen.sum())
    if rest > 0:
        leftover = bridge.clone()
        leftover[chosen] = -float("inf")
        chosen[leftover.topk(rest).indices] = True
    return chosen


def analyse(lm: LM, case) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    groups = torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long()
    original = lm.cold(torch.cat((P, E, S)), groups=groups)
    exposure = original.mass[:, nP + nE:, 1].mean(0)
    cP, with_S = original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))
    del original

    context, start, n = torch.cat((P, S)), nP, nP + nS
    pos = torch.arange(n)
    truth = Cells.cat([cP, lm.run(context, pos, pos >= nP, cP).cells])
    stale = Cells.cat([cP, with_S])
    reply, truth_answer = reply_of(lm, context, Q, truth)
    reader = Reader(lm, context, Q, reply)
    truth_tf, _ = reader.logits(truth)
    _, mass = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
    reader_score = reader_attention(mass[:, : Q.numel()], start, n).mean(0)
    bridge = exposure * reader_score

    lengths = [case.segments[i].numel() for i in range(case.edit + 1, case.fresh_from)]
    starts = [sum(lengths[:j]) for j in range(len(lengths))]
    question = case.meta["question_segment"] - (case.edit + 1)
    density_of = [reader_score[s:s + length].mean().item() for s, length in zip(starts, lengths)]
    picked = max(range(len(lengths)), key=lambda j: density_of[j])
    in_question, in_picked = torch.zeros(nS, dtype=torch.bool), torch.zeros(nS, dtype=torch.bool)
    in_question[starts[question]:starts[question] + lengths[question]] = True
    in_picked[starts[picked]:starts[picked] + lengths[picked]] = True
    priority = {"question_exposure": first(in_question, exposure), "question_bridge": first(in_question, bridge),
                "segment_exposure": first(in_picked, exposure), "segment_bridge": first(in_picked, bridge), "bridge": bridge}

    out = {"name": case.name, **{k: v for k, v in case.meta.items() if k != "edit_text"}, "nP": nP, "nE": nE, "nS": nS,
           "segments": len(lengths), "question_tokens": lengths[question], "picked_is_question": picked == question,
           "question_reader_share": reader_score[in_question].sum().item() / max(reader_score.sum().item(), 1e-12),
           "results": {}, "answers": {"truth": truth_answer}}
    for name in POLICIES:
        out["results"][name] = {}
        for budget in BUDGETS:
            count = budget_count(budget, nS)
            selected = density_choice(starts, lengths, bridge, count) if name == "density" else torch.zeros(nS, dtype=torch.bool).index_fill_(0, priority[name].topk(count).indices, True)
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[start:] = selected
            healed = heal(lm, context, pos, stale, chosen)
            out["results"][name][str(budget)] = compare(truth_tf, reader.logits(healed)[0], reply)
            if budget == 0.10 and name in DECODED:
                out["answers"][name] = decode(lm, reader.tokens[: reader.n + reader.nQ], reader.pos[: reader.n + reader.nQ], healed)
    return out


def main() -> None:
    key, supporting = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = build_cases(lm, supporting, 0)
    stem = f"e2d_reader_first_{key}_{tag}"
    settings = {"model": MODELS[key], "budgets": list(BUDGETS), "policies": list(POLICIES)}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}", flush=True)
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case))
        if len(rows) % 10 == 0:
            save(stem, {**settings, "rows": rows})
            print(f"{len(rows)} cases", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)
    found = sum(r["picked_is_question"] for r in rows)
    print(f"reader attention picks the question segment in {found} of {len(rows)} cases")
    for name in POLICIES:
        mean = sum(r["results"][name]["0.1"]["kl_tf"] for r in rows) / len(rows)
        print(f"  {name:18s} teacher-forced KL at 10%: {mean:.4f}")


if __name__ == "__main__":
    main()
