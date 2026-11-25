"""E18: how far apart are two correct implementations in bfloat16? (calibrates E9's engine-vs-library check)

E9 found our bfloat16 forward pass of Qwen2.5-7B choosing the same next token as the library's
bfloat16 pass at 92% of 512 positions. To read that number, the same check is run here on the
models that fit in float32: on the first 512 tokens of four of E2c's prompts, logits of

    engine32, library32   our forward pass and the library's, float32
    engine16, library16   the same in bfloat16

compared pairwise (largest absolute logit gap, share of positions with the same argmax).

Usage: e18_reference.py <model-key> [tag] [device]
"""
from __future__ import annotations

import sys

import torch

from common import MODELS, best_device, save
from e2c_ablation import build_cases
from mutablekv import LM

PAIRS = (("engine32", "library32"), ("engine16", "library16"), ("library16", "library32"), ("engine16", "library32"), ("engine16", "engine32"))


@torch.inference_mode()
def logits(lm: LM, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    ours = lm.logits(lm.cold(tokens).hidden).float().cpu()
    theirs = lm.hf(tokens[None].to(lm.device)).logits[0].float().cpu()
    return {"engine": ours, "library": theirs}


def main() -> None:
    key = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "main"
    device = sys.argv[3] if len(sys.argv) > 3 else best_device()
    wide = LM(MODELS[key], device, torch.float32)
    prompts = [torch.cat(case.segments)[:512] for case in build_cases(wide, 1, 0)[:4]]
    out32 = [logits(wide, p) for p in prompts]
    del wide
    narrow = LM(MODELS[key], device, torch.bfloat16)
    out16 = [logits(narrow, p) for p in prompts]
    rows = []
    for p, a, b in zip(prompts, out32, out16):
        named = {"engine32": a["engine"], "library32": a["library"], "engine16": b["engine"], "library16": b["library"]}
        rows.append({"tokens": int(p.numel()), **{f"{x}_vs_{y}": {"max_gap": (named[x] - named[y]).abs().max().item(),
                                                                    "same_argmax": (named[x].argmax(-1) == named[y].argmax(-1)).float().mean().item()}
                                                  for x, y in PAIRS}})
    print("saved", save(f"e18_reference_{key}_{tag}", {"model": MODELS[key], "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
