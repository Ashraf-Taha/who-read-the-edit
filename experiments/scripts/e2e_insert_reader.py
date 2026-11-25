"""E2e: reader-first repair after an insert, which needs no record of exposure.

E2c's cases read the other way round: a supporting document arrives late and is inserted before
a suffix that was encoded without it, and the cached question in that suffix never read it.
Policies, at 10% of the suffix:

    reader              reader attention alone (ProphetKV's practical score)
    bridge_qs4g         kept queries (4 layers, group mean) scored against the new keys, x reader
    bridge_true         true exposure from the cold pass x reader (not available in practice)
    question_reader     the cached question (given) first, then reader attention
    segment_reader      the stale segment with the most reader attention per token first, then
                        reader attention: needs only the reader pass
    segment_qs4g        the same segment first, then the kept-query estimate x reader
    boundary            tokens right after the insert

Metrics as in E2c: teacher-forced KL over the first 16 reply tokens and KL at the first token.

Usage: e2e_insert_reader.py <model-key> <supporting-per-cell> [tag] [device]
"""
from __future__ import annotations

import sys

import torch

from common import MODELS, best_device, resume, save
from e2c_ablation import Reader, budget_count, build_cases, compare, reader_attention, reply_of
from e2d_reader_first import first
from mutablekv import LM, Cells, heal, would_be_exposure
from mutablekv.sketch import spread

POLICIES = ("boundary", "reader", "bridge_qs4g", "bridge_true", "question_reader", "segment_reader", "segment_qs4g")
BUDGET = 0.10


def analyse(lm: LM, case) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    with_e = torch.cat((P, E, S))
    groups = torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long()
    original = lm.cold(with_e, groups=groups)
    true_exposure = original.mass[:, nP + nE:, 1].mean(0)
    truth = original.cells
    del original
    cP, cE = truth.take(slice(0, nP)), truth.take(slice(nP, nP + nE))
    before = lm.run(torch.cat((P, S)), torch.arange(nP + nS), torch.arange(nP + nS) >= nP, cP, keep_queries=True)
    four = spread(lm.L)
    grouped_q = before.queries.reshape(lm.L, lm.G, lm.H // lm.G, nS, lm.d_head).mean(2)
    grouped_lse = before.lse.reshape(lm.L, lm.G, lm.H // lm.G, nS).mean(2)
    estimate = would_be_exposure(lm, grouped_q, grouped_lse, nE, cE.K, layers=four)
    stale = Cells.cat([cP, cE, before.cells])
    del before, grouped_q, grouped_lse

    start, n = nP + nE, nP + nE + nS
    pos = torch.arange(n)
    reply, _ = reply_of(lm, with_e, Q, truth)
    reader = Reader(lm, with_e, Q, reply)
    truth_tf, _ = reader.logits(truth)
    stale_tf, mass = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
    reader_score = reader_attention(mass[:, : Q.numel()], start, n).mean(0)

    lengths = [case.segments[i].numel() for i in range(case.edit + 1, case.fresh_from)]
    starts = [sum(lengths[:j]) for j in range(len(lengths))]
    question = case.meta["question_segment"] - (case.edit + 1)
    picked = max(range(len(lengths)), key=lambda j: reader_score[starts[j]:starts[j] + lengths[j]].mean().item())
    mask = lambda j: (torch.arange(nS) >= starts[j]) & (torch.arange(nS) < starts[j] + lengths[j])
    scores = {"boundary": -torch.arange(nS).float(), "reader": reader_score, "bridge_qs4g": estimate * reader_score,
              "bridge_true": true_exposure * reader_score, "question_reader": first(mask(question), reader_score),
              "segment_reader": first(mask(picked), reader_score), "segment_qs4g": first(mask(picked), estimate * reader_score)}
    out = {"name": case.name, **{k: v for k, v in case.meta.items() if k != "edit_text"}, "nP": nP, "nE": nE, "nS": nS,
           "picked_is_question": picked == question, "results": {"stale": compare(truth_tf, stale_tf, reply)}}
    for name in POLICIES:
        chosen = torch.zeros(n, dtype=torch.bool)
        chosen[start + scores[name].topk(budget_count(BUDGET, nS)).indices] = True
        out["results"][name] = compare(truth_tf, reader.logits(heal(lm, with_e, pos, stale, chosen))[0], reply)
    return out


def main() -> None:
    key, supporting = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = build_cases(lm, supporting, 0)
    stem = f"e2e_insert_reader_{key}_{tag}"
    settings = {"model": MODELS[key], "budget": BUDGET, "policies": list(POLICIES)}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case))
        if len(rows) % 10 == 0:
            save(stem, {**settings, "rows": rows})
            print(f"{len(rows)} cases", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
