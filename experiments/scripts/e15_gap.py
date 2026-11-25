"""E15: how much does a gap in positions change the output, as a function of its size?

Section 4.3's cases (HotpotQA, ten documents, the question asked after the documents as new
input), with nothing deleted: every token is kept and every cell is computed exactly, but the
positions of everything after one document boundary (the boundary where E1 deletes a document)
jump by `g`. This isolates the geometry of a hole from staleness and from the content removed.
For each g, KL(contiguous cold pass || cold pass with the gap) at the first answer token, and
whether the 16-token greedy answer is unchanged. g = 0 is the reference.

Usage: e15_gap.py <model-key> <cases-per-kind> [tag] [device]
"""
from __future__ import annotations

import sys

import torch

from common import MODELS, best_device, kl_rows, normalise, resume, save
from mutablekv import LM, greedy
from workloads import hotpot_cases

GAPS = (1, 4, 16, 64, 256, 1024)


def answer(lm: LM, tokens: torch.Tensor, pos: torch.Tensor) -> str:
    stop = {lm.tok.eos_token_id, lm.tok.convert_tokens_to_ids("<|im_end|>")}
    cells = lm.cold(tokens[:-1], pos[:-1]).cells
    return lm.tok.decode(greedy(lm, tokens, pos, cells, 16, stop)).strip().split("\n")[0]


def analyse(lm: LM, case) -> dict:
    tokens = torch.cat(case.segments)
    boundary = sum(s.numel() for s in case.segments[:case.edit])
    n = tokens.numel()
    contiguous = torch.arange(n)
    reference = lm.logits(lm.cold(tokens, contiguous).hidden[-1:]).float().cpu()
    reference_answer = answer(lm, tokens, contiguous)
    out = {"name": case.name, "kind": case.meta["kind"], "n": n, "boundary": boundary, "results": {}}
    for g in GAPS:
        pos = contiguous + g * (contiguous >= boundary).long()
        logits = lm.logits(lm.cold(tokens, pos).hidden[-1:]).float().cpu()
        out["results"][str(g)] = {"kl": kl_rows(reference, logits)[0].item(),
                                  "same_answer": normalise(answer(lm, tokens, pos)) == normalise(reference_answer)}
    return out


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = hotpot_cases(lm, count, "distractor") + hotpot_cases(lm, count, "gold")
    stem = f"e15_gap_{key}_{tag}"
    settings = {"model": MODELS[key], "gaps": list(GAPS)}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case))
        if len(rows) % 10 == 0:
            save(stem, {**settings, "rows": rows})
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
