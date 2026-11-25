"""E17: how far apart are two exact computations in 16-bit arithmetic?

On E2c's cases (same seeds, so the same contexts and replies), the re-prefill after the deletion
is computed three ways:

    fp32         prefix cells from the cold pass with the document, suffix recomputed (as in E2c)
    bf16         the same in bfloat16 (normalisation and softmax in float32, as in E9)
    bf16_oneshot one bfloat16 cold pass over the edited context, without reading a cached prefix

and compared over the teacher-forced fp32 reply: KL per position. In exact arithmetic all three
are the same; the differences are the noise floor that repair differences must be read against.

Usage: e17_precision.py <model-key> <supporting-per-cell> [tag] [device]
"""
from __future__ import annotations

import sys

import torch

from common import MODELS, best_device, kl_rows, resume, save
from e2c_ablation import Reader, build_cases, reply_of
from mutablekv import LM, Cells


def reprefill(lm: LM, case) -> tuple[torch.Tensor, Cells, Cells]:
    P, E, S = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from)
    nP = P.numel()
    cP = lm.cold(torch.cat((P, E, S))).cells.take(slice(0, nP))
    context = torch.cat((P, S))
    pos = torch.arange(context.numel())
    return context, Cells.cat([cP, lm.run(context, pos, pos >= nP, cP).cells]), lm.cold(context).cells


def main() -> None:
    key, supporting = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    wide = LM(MODELS[key], device, torch.float32)
    narrow = LM(MODELS[key], device, torch.bfloat16)
    cases = build_cases(wide, supporting, 0)
    stem = f"e17_precision_{key}_{tag}"
    rows = resume(stem, model=MODELS[key]).get("rows", [])
    done = {r["name"] for r in rows}
    for case in cases:
        if case.name in done:
            continue
        Q = case.part(case.fresh_from, None)
        context, exact32, _ = reprefill(wide, case)
        reply, _ = reply_of(wide, context, Q, exact32)
        logits32, _ = Reader(wide, context, Q, reply).logits(exact32)
        del exact32
        _, exact16, oneshot16 = reprefill(narrow, case)
        reader16 = Reader(narrow, context, Q, reply)
        logits16, _ = reader16.logits(exact16)
        oneshot, _ = reader16.logits(oneshot16)
        rows.append({"name": case.name, "place": case.meta["place"], "lead": case.meta["lead"], "reply_tokens": reply.numel(),
                     "fp32_vs_bf16": [round(x, 7) for x in kl_rows(logits32, logits16).tolist()],
                     "bf16_vs_oneshot": [round(x, 7) for x in kl_rows(logits16, oneshot).tolist()],
                     "fp32_vs_oneshot": [round(x, 7) for x in kl_rows(logits32, oneshot).tolist()],
                     "top1_fp32_bf16": (logits32.argmax(-1) == logits16.argmax(-1)).float().mean().item()})
        if len(rows) % 12 == 0:
            save(stem, {"model": MODELS[key], "rows": rows})
    print("saved", save(stem, {"model": MODELS[key], "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
