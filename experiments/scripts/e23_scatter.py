"""E23: does it cost more to recompute scattered tokens than a contiguous block of the same size?

E6's cases (supporting document deleted, cached question at the end). For each budget, the same
number of suffix tokens is recomputed twice: once spread at random over the suffix, as a
token-level score picks them, and once as one block right after the edit, as a window does.
Timing as in E6 (best of three after a warm-up, current device). This measures our unfused
PyTorch engine only; Appendix A of the paper gives a cost model for a fused one.

Usage: e23_scatter.py <model-key> <cases> [tag] [device]
"""
from __future__ import annotations

import statistics
import sys

import torch

from common import MODELS, best_device, resume, save
from e6_cost import clock
from mutablekv import LM, Cells, heal
from workloads import reader_cases

BUDGETS = (0.01, 0.10, 0.20)


def runs(chosen: torch.Tensor) -> int:
    flags = chosen.int()
    return int((flags[1:] > flags[:-1]).sum()) + int(flags[0])


def measure(lm: LM, case) -> dict:
    P, E, S = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    original = lm.cold(torch.cat((P, E, S)))
    stale = Cells.cat([original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))])
    del original
    kept = torch.cat((P, S))
    n = kept.numel()
    pos = torch.arange(n)
    out = {"name": case.name, "nP": nP, "nS": nS, "device": lm.device}
    generator = torch.Generator().manual_seed(nS)
    for budget in BUDGETS:
        count = max(1, round(budget * nS))
        scattered = torch.zeros(n, dtype=torch.bool)
        scattered[nP + torch.randperm(nS, generator=generator)[:count]] = True
        block = torch.zeros(n, dtype=torch.bool)
        block[nP:nP + count] = True
        out[f"scattered@{budget}"] = clock(lm, lambda: heal(lm, kept, pos, stale, scattered))
        out[f"block@{budget}"] = clock(lm, lambda: heal(lm, kept, pos, stale, block))
        out[f"runs@{budget}"] = runs(scattered[nP:])
        out[f"tokens@{budget}"] = count
    return out


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = reader_cases(lm, count, "gold", "end")
    stem, settings = f"e23_scatter_{key}_{tag}", {"model": MODELS[key], "budgets": list(BUDGETS)}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    for case in cases:
        if case.name not in done:
            rows.append(measure(lm, case))
            save(stem, {**settings, "rows": rows})
    print(f"{MODELS[key]} on {lm.device}, n={len(rows)}; median seconds (best of three):")
    for budget in BUDGETS:
        scattered = statistics.median(r[f"scattered@{budget}"] for r in rows)
        block = statistics.median(r[f"block@{budget}"] for r in rows)
        spread = statistics.median(r[f"runs@{budget}"] for r in rows)
        print(f"  {budget:.0%}: scattered {scattered:.3f} ({spread:.0f} runs), block {block:.3f}, ratio {scattered / block:.2f}")


if __name__ == "__main__":
    main()
