"""E6: wall-clock of repair against re-prefilling the suffix, in the reference engine.

For the E2 cases (supporting document deleted, cached question at the end), times on
the current device:

    reprefill   recompute every suffix token (what a prefix cache does after the edit)
    reader      one pass of the new input over the stale cache, keeping attention mass
    repair@b    recompute the top fraction b of suffix tokens (selection already made)

The bridge policy costs reader + repair@b. Numbers come from an unoptimised PyTorch
forward pass on a laptop GPU and show relative cost only; they say nothing about a
serving engine with fused kernels.

Usage: e6_cost.py <model-key> <cases> [tag] [device]
"""
from __future__ import annotations

import statistics
import sys
import time

import torch

from common import MODELS, best_device, resume, save
from mutablekv import LM, Cells, heal
from workloads import reader_cases

BUDGETS = (0.05, 0.10, 0.20)
REPEATS = 3


def clock(lm: LM, fn) -> float:
    def sync():
        if lm.device == "mps":
            torch.mps.synchronize()
    fn()
    sync()
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - start)
    return best


def measure(lm: LM, case) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    original = lm.cold(torch.cat((P, E, S)))
    stale = Cells.cat([original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))])
    del original
    kept = torch.cat((P, S))
    n = kept.numel()
    pos = torch.arange(n)
    asked, asked_pos = torch.cat((kept, Q)), torch.arange(n + Q.numel())
    out = {"name": case.name, "nP": nP, "nS": nS, "nQ": Q.numel(), "device": lm.device}
    out["reprefill"] = clock(lm, lambda: heal(lm, kept, pos, stale, pos >= nP))
    out["reader"] = clock(lm, lambda: lm.run(asked, asked_pos, asked_pos >= n, stale, groups=asked_pos))
    out["read_only"] = clock(lm, lambda: lm.run(asked, asked_pos, asked_pos >= n, stale))
    generator = torch.Generator().manual_seed(nS)
    for budget in BUDGETS:
        chosen = torch.zeros(n, dtype=torch.bool)
        chosen[nP + torch.randperm(nS, generator=generator)[: max(1, round(budget * nS))]] = True
        out[f"repair@{budget}"] = clock(lm, lambda: heal(lm, kept, pos, stale, chosen))
    return out


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = reader_cases(lm, count, "gold", "end")
    stem, settings = f"e6_cost_{key}_{tag}", {"model": MODELS[key], "budgets": list(BUDGETS), "repeats": REPEATS}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    for case in cases:
        if case.name not in done:
            rows.append(measure(lm, case))
            save(stem, {**settings, "rows": rows})
    median = lambda field: statistics.median(r[field] for r in rows)
    print(f"{MODELS[key]} on {lm.device}, n={len(rows)}, suffix {statistics.fmean(r['nS'] for r in rows):.0f} tokens; median seconds (best of {REPEATS}):")
    print(f"  re-prefill suffix {median('reprefill'):.3f} | read new input {median('read_only'):.3f} | read with attention kept {median('reader'):.3f}")
    for budget in BUDGETS:
        total = median("reader") + median(f"repair@{budget}")
        print(f"  repair {budget:.0%}: {median(f'repair@{budget}'):.3f}; with the reader pass {total:.3f} = {total / median('reprefill'):.2f} of re-prefill")


if __name__ == "__main__":
    main()
