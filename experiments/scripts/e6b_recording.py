"""E6b: cost of recording what repair needs during prefill, in the reference engine.

For the E6 cases (the whole prompt before the new input, about 2,000 tokens), times the cold
prefill on the current device three ways:

    plain     the ordinary forward pass
    exposure  the same pass, also accumulating each token's attention mass on every earlier
              segment (one group per segment, as the runtime records it for deletions)
    sketch    as exposure, also keeping the queries and softmax normalisers that the
              insert-side estimate needs

Like E6 this is an unoptimised PyTorch implementation: the ratios say how much extra work the
recording is here, not what it would cost inside a fused attention kernel.

Usage: e6b_recording.py <model-key> <cases> [tag] [device]
"""
from __future__ import annotations

import statistics
import sys

import torch

from common import MODELS, best_device, resume, save
from e6_cost import REPEATS, clock
from mutablekv import LM
from workloads import reader_cases


def measure(lm: LM, case) -> dict:
    segments = case.segments[: case.fresh_from]
    tokens = torch.cat(segments)
    groups = torch.cat([torch.full((s.numel(),), j) for j, s in enumerate(segments)]).long()
    out = {"name": case.name, "n": tokens.numel(), "segments": len(segments), "device": lm.device}
    out["plain"] = clock(lm, lambda: lm.cold(tokens))
    out["exposure"] = clock(lm, lambda: lm.cold(tokens, groups=groups))
    out["sketch"] = clock(lm, lambda: lm.cold(tokens, groups=groups, keep_queries=True))
    return out


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = reader_cases(lm, count, "gold", "end")
    stem, settings = f"e6b_recording_{key}_{tag}", {"model": MODELS[key], "repeats": REPEATS}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    for case in cases:
        if case.name not in done:
            rows.append(measure(lm, case))
            save(stem, {**settings, "rows": rows})
    median = lambda field: statistics.median(r[field] for r in rows)
    print(f"{MODELS[key]} on {lm.device}, n={len(rows)}, prompt {statistics.fmean(r['n'] for r in rows):.0f} tokens, "
          f"{statistics.fmean(r['segments'] for r in rows):.0f} segments; median seconds (best of {REPEATS}):")
    print(f"  plain {median('plain'):.3f} | with exposure {median('exposure'):.3f} ({median('exposure') / median('plain'):.2f}x) | "
          f"with exposure and sketch {median('sketch'):.3f} ({median('sketch') / median('plain'):.2f}x)")


if __name__ == "__main__":
    main()
