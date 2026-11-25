"""E20: our forward pass against the library's in bfloat16 on every prompt of the main experiment.

E9 checked one prompt of Qwen2.5-7B (same next token at 92% of 512 positions, largest logit gap
23.8). Here every one of the main experiment's prompts is checked (first 512 tokens): per prompt,
the share of positions with the same argmax, the largest absolute logit gap and where it sits,
and, at the positions where the two disagree, the library's margin between its first and second
choice, so that disagreements at near-ties can be told apart from real ones.

Usage: e20_reference.py <model-key> [tag] [dtype] [device]
"""
from __future__ import annotations

import sys

import torch

from common import MODELS, best_device, resume, save, release
from e19_family import matched_cases
from mutablekv import LM


@torch.inference_mode()
def compare(lm: LM, tokens: torch.Tensor) -> dict:
    ours = lm.logits(lm.cold(tokens).hidden).float().cpu()
    theirs = lm.hf(tokens[None].to(lm.device)).logits[0].float().cpu()
    gap = (ours - theirs).abs()
    disagree = (ours.argmax(-1) != theirs.argmax(-1)).nonzero().flatten()
    top2 = theirs.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1])
    worst = int(gap.max(-1).values.argmax())
    log_p, log_q = torch.log_softmax(theirs, -1), torch.log_softmax(ours, -1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    return {"tokens": int(tokens.numel()), "same_argmax": 1.0 - disagree.numel() / tokens.numel(),
            "max_gap": gap.max().item(), "max_gap_position": worst, "max_gap_at_position_0": worst == 0,
            "max_gap_without_position_0": gap[1:].max().item() if tokens.numel() > 1 else 0.0,
            "margin_at_disagreements": [round(margin[i].item(), 4) for i in disagree.tolist()],
            "median_margin": margin.median().item(), "kl_mean": kl.mean().item(), "kl_max": kl.max().item()}


def main() -> None:
    key = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "main"
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[3] if len(sys.argv) > 3 else "bf16"]
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, dtype)
    stem = f"e20_reference_{key}_{tag}"
    settings = {"model": MODELS[key], "dtype": str(dtype)}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    for case in matched_cases(lm, 12):
        if case.name in done:
            continue
        rows.append({"name": case.name, **compare(lm, torch.cat(case.segments)[:512])})
        release()
        if len(rows) % 12 == 0:
            save(stem, {**settings, "rows": rows})
            print(f"{len(rows)} prompts", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
