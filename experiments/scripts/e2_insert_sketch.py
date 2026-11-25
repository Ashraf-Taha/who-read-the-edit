"""E2b: repairing after an insert, with exposure estimated from kept queries.

After an insert the cached suffix never attended to the new segment, so there is no
recorded exposure. Instead, each suffix token's queries are kept from when it was
encoded and scored against the new segment's keys (`mutablekv.sketch`). Variants by
what is kept per token:

    qs      queries of every head at every layer (as large as the key cache times H/G)
    qs4     every head at four spread layers
    qs4g    one query per key/value group (the group mean) at four spread layers

Compared with reader attention alone (ProphetKV's signal), first-layer exposure times
reader, and the bridge score with the true exposure from the cold pass (not available
in practice). Same cases as e2_healing.py: supporting document, three reader placements.

Usage: e2_insert_sketch.py <model-key> <cases-per-place> [tag] [device]
"""
from __future__ import annotations

import sys

import torch
from scipy.stats import spearmanr, wilcoxon

from common import MODELS, best_device, mean_ci, normalise, resume, save
from e2_healing import BUDGETS, evaluate
from mutablekv import LM, Cells, would_be_exposure
from workloads import READER_PLACES, Case, reader_cases

POLICIES = ["reader", "bridge_l1", "bridge_qs", "bridge_qs4", "bridge_qs4g", "exposure_qs4g", "bridge"]
BUDGETS[:] = [0.05, 0.10, 0.20]


def spread(layers: int, count: int = 4) -> list[int]:
    return sorted({round(layers * (2 * i + 1) / (2 * count)) for i in range(count)})


def analyse(lm: LM, case: Case) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    before_question = sum(s.numel() for s in case.segments[case.edit + 1: case.meta["question_segment"]])
    reader_span = (before_question, before_question + case.segments[case.meta["question_segment"]].numel())

    with_e, without_e = torch.cat((P, E, S)), torch.cat((P, S))
    original = lm.cold(with_e, groups=torch.arange(nP + nE + nS))
    lineage = original.mass[:, nP + nE:]
    exposure_by_layer = lineage[:, :, nP: nP + nE].sum(-1)
    graph = lineage[:, :, nP + nE:].clone()
    graph.diagonal(dim1=1, dim2=2).zero_()
    cells = original.cells
    del original, lineage

    cP, cE = cells.take(slice(0, nP)), cells.take(slice(nP, nP + nE))
    before = lm.run(without_e, torch.arange(nP + nS), torch.arange(nP + nS) >= nP, cP, keep_queries=True)
    queries, lse = before.queries, before.lse
    four = spread(lm.L)
    grouped_q = queries.reshape(lm.L, lm.G, lm.H // lm.G, nS, lm.d_head).mean(2)
    grouped_lse = lse.reshape(lm.L, lm.G, lm.H // lm.G, nS).mean(2)
    estimates = {
        "qs": would_be_exposure(lm, queries, lse, nE, cE.K),
        "qs4": would_be_exposure(lm, queries, lse, nE, cE.K, layers=four),
        "qs4g": would_be_exposure(lm, grouped_q, grouped_lse, nE, cE.K, layers=four),
    }
    truth = exposure_by_layer.mean(0)
    agreement = {name: float(spearmanr(est.numpy(), truth.numpy()).statistic) for name, est in estimates.items()}
    agreement["l1"] = float(spearmanr(exposure_by_layer[0].numpy(), truth.numpy()).statistic)
    del queries, lse, grouped_q, grouped_lse, before

    insert = evaluate(lm, with_e, Q, Cells.cat([cP, cE, lm.run(without_e, torch.arange(nP + nS), torch.arange(nP + nS) >= nP, cP).cells]),
                      cells, nP + nE, exposure_by_layer=exposure_by_layer, graph=graph, edited=E, reader_span=reader_span,
                      policies=POLICIES, extra_exposure=estimates)
    return {"name": case.name, **case.meta, "nP": nP, "nE": nE, "nS": nS, "layers_kept": four,
            "rank_agreement_with_true_exposure": agreement, "insert": insert}


def report(rows: list[dict]) -> None:
    chosen = [r["insert"] for r in rows]
    print(f"\n=== insert a supporting document, all placements / n={len(rows)}")
    for name in ("qs", "qs4", "qs4g", "l1"):
        rho = mean_ci([r["rank_agreement_with_true_exposure"][name] for r in rows])
        print(f"  rank correlation of exposure estimate '{name}' with the true exposure: {rho[0]:.3f} ± {rho[1]:.3f}")
    stale, half = mean_ci([r["results"]["stale"]["0"]["kl_last"] for r in chosen])
    print(f"  no repair: KL {stale:.4f} ± {half:.4f}")
    print("    policy        " + "".join(f"{b:>18.0%}" for b in BUDGETS) + "   answer identical @10%")
    for name in POLICIES:
        line = "".join("  {:.4f} ± {:.4f}".format(*mean_ci([r["results"][name][str(b)]["kl_last"] for r in chosen])) for b in BUDGETS)
        if name in chosen[0]["answers"]:
            m, h = mean_ci([float(normalise(r["answers"][name]) == normalise(r["answers"]["truth"])) for r in chosen])
            line += f"   {m:.3f} ± {h:.3f}"
        print(f"    {name:13s} {line}")
    for ours in ("bridge_qs", "bridge_qs4", "bridge_qs4g"):
        a = [r["results"][ours]["0.1"]["kl_last"] for r in chosen]
        cells = []
        for other in ("reader", "bridge_l1", "bridge"):
            b = [r["results"][other]["0.1"]["kl_last"] for r in chosen]
            cells.append(f"{other} {sum(x < y for x, y in zip(a, b)) / len(a):.2f} (p={wilcoxon(a, b).pvalue:.0e})")
        print(f"  {ours} has lower KL at 10% than: " + "   ".join(cells))
    for place in READER_PLACES:
        part = [r["insert"] for r in rows if r["place"] == place]
        print(f"  {place:9s} KL @10%: " + ", ".join(f"{p} {mean_ci([r['results'][p]['0.1']['kl_last'] for r in part])[0]:.4f}" for p in POLICIES))


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = [case for place in READER_PLACES for case in reader_cases(lm, count, "gold", place)]
    stem, settings = f"e2b_insert_sketch_{key}_{tag}", {"model": MODELS[key], "budgets": list(BUDGETS), "policies": POLICIES}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}")
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case))
        if len(rows) % 10 == 0:
            save(stem, {**settings, "rows": rows})
    print("saved", save(stem, {**settings, "rows": rows}))
    report(rows)


if __name__ == "__main__":
    main()
