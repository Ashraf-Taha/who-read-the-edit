"""E7: does the score pick the recomputations that help, and how far is it from the best selection?

Cases are supporting-document deletions in E2c's layout with the edited document 5th of ten, the
cached question at each of the three places, and a different seed per place so that every case is
a different question. For each case the suffix S is cut into blocks of BLOCK tokens and two things
are measured against the cold pass, teacher-forced over the re-prefill's reply:

    marginal  recompute one block alone; the KL reduction it buys is compared with each policy's
              score summed over the block (Spearman over the blocks of a case)
    greedy    greedy forward selection of blocks, each step adding the block that lowers the KL
              most, until the 10% token budget is reached (the last block is cut to fit). It
              optimises the evaluation metric itself, with the answer known, so it is a reference
              for how much a selection of this size can do, not a policy.

The token-level policies of E2c are evaluated at the same 10% budget on the same cases.

Usage: e7_mechanism.py <model-key> <cases-per-placement> [tag] [device]
"""
from __future__ import annotations

import sys

import torch
from scipy.stats import spearmanr

from common import MODELS, best_device, resume, save, release
from e2_healing import relative_deviation
from e2c_ablation import Reader, budget_count, compare, policy_scores, reader_attention, reply_of
from e2d_reader_first import first
from mutablekv import LM, Cells, heal
from workloads import READER_PLACES, reader_cases

BLOCK = 8
LEAD = 4
BUDGET = 0.10
POLICIES = ("boundary", "tail", "reader", "exposure", "sum", "bridge", "question_exposure", "segment_bridge", "oracle", "oracle_reader")


def analyse(lm: LM, case) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    with_e = torch.cat((P, E, S))
    groups = torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long()
    original = lm.cold(with_e, groups=groups)
    exposure_by_layer = original.mass[:, nP + nE:, 1]
    cP, with_S = original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))
    del original

    context, start, n = torch.cat((P, S)), nP, nP + nS
    pos = torch.arange(n)
    truth = Cells.cat([cP, lm.run(context, pos, pos >= nP, cP).cells])
    stale = Cells.cat([cP, with_S])
    reply, _ = reply_of(lm, context, Q, truth)
    reader = Reader(lm, context, Q, reply)
    truth_tf, _ = reader.logits(truth)
    stale_tf, mass = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
    reader_by_layer = reader_attention(mass[:, : Q.numel()], start, n)
    oracle = relative_deviation(lm, with_S, truth.take(slice(nP, None))).mean(0)
    scores, _ = policy_scores(lm, case, nS, exposure_by_layer, reader_by_layer, None, oracle)
    lengths = [case.segments[i].numel() for i in range(case.edit + 1, case.fresh_from)]
    starts = [sum(lengths[:j]) for j in range(len(lengths))]
    reader_score = reader_by_layer.mean(0)
    picked = max(range(len(lengths)), key=lambda j: reader_score[starts[j]:starts[j] + lengths[j]].mean().item())
    in_picked = torch.zeros(nS, dtype=torch.bool)
    in_picked[starts[picked]:starts[picked] + lengths[picked]] = True
    scores["segment_bridge"] = first(in_picked, scores["bridge"])
    base = compare(truth_tf, stale_tf, reply)

    def evaluate(selected: torch.Tensor) -> dict:
        chosen = torch.zeros(n, dtype=torch.bool)
        chosen[start:] = selected
        result = compare(truth_tf, reader.logits(heal(lm, context, pos, stale, chosen))[0], reply)
        # MPS keeps freed blocks cached until usage passes ~1.4x the working set, far more than this laptop has free
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return result

    blocks = [(low, min(low + BLOCK, nS)) for low in range(0, nS, BLOCK)]
    marginal = []
    for low, high in blocks:
        selected = torch.zeros(nS, dtype=torch.bool)
        selected[low:high] = True
        marginal.append(evaluate(selected))
    release()
    reduction = [base["kl_tf"] - m["kl_tf"] for m in marginal]
    block_scores = {name: [scores[name][low:high].sum().item() for low, high in blocks] for name in POLICIES}
    correlation = {name: spearmanr(values, reduction).statistic for name, values in block_scores.items()}
    best_blocks = set(sorted(range(len(blocks)), key=lambda b: -reduction[b])[:5])
    top_hit = {name: int(max(range(len(blocks)), key=lambda b: values[b]) in best_blocks) for name, values in block_scores.items()}

    budget = budget_count(BUDGET, nS)
    policies = {}
    for name in POLICIES:
        selected = torch.zeros(nS, dtype=torch.bool)
        selected[scores[name].topk(budget).indices] = True
        policies[name] = evaluate(selected)

    selected, remaining, path = torch.zeros(nS, dtype=torch.bool), set(range(len(blocks))), []
    while int(selected.sum()) < budget and remaining:
        room = budget - int(selected.sum())
        best = None
        for b in sorted(remaining):
            low, high = blocks[b]
            trial = selected.clone()
            trial[low:min(high, low + room)] = True
            result = evaluate(trial)
            if best is None or result["kl_tf"] < best[1]["kl_tf"]:
                best = (b, result, trial)
        selected = best[2]
        remaining.discard(best[0])
        path.append({"block": best[0], "tokens": int(selected.sum()), **best[1]})
        release()
    return {"name": case.name, "place": case.meta["place"], "nP": nP, "nE": nE, "nS": nS, "budget": budget, "blocks": len(blocks),
            "base": base, "marginal": marginal, "reduction": reduction, "block_scores": block_scores, "correlation": correlation,
            "top_hit": top_hit, "policies": policies, "greedy": path[-1] if path else base, "path": path,
            "greedy_blocks": sorted(p["block"] for p in path)}


def main() -> None:
    key, per_place = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = []
    for index, place in enumerate(READER_PLACES):
        for case in reader_cases(lm, per_place, "gold", place, seed=7000 + 100 * index + LEAD, lead=LEAD):
            case.name += f"/lead{LEAD}"
            cases.append(case)
    stem = f"e7_mechanism_{key}_{tag}"
    settings = {"model": MODELS[key], "block": BLOCK, "lead": LEAD, "budget": BUDGET, "policies": list(POLICIES)}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}", flush=True)
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case))
        release()
        save(stem, {**settings, "rows": rows})
        print(f"{len(rows)} cases", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
