"""E1p: the fresh-question and cached-question contrast of Section 4, on the same cases.

E1 built its fresh-question and cached-question groups from different shuffles, so they held
different questions. Here every case is read both ways. The token sequence is the same (system
prompt, ten documents, the question, the assistant opener); only the split between cache and new
input differs:

    fresh    the cache holds the documents; the question and the opener are new input
    cached   the cache also holds the question; only the opener is new input

A supporting document (not the first or last) is deleted. For each split: KL at the first answer
token between the re-prefill and the relocated stale cache, top-1 agreement there, and whether the
greedy 16-token answer is the re-prefill's. The effect of the deletion itself (re-prefill against
the cold pass that keeps the document) is the same for both splits and is recorded once.

Usage: e1p_paired.py <model-key> <cases> [tag] [device]
"""
from __future__ import annotations

import sys

import torch

from common import MODELS, best_device, kl_rows, normalise, resume, save
from mutablekv import LM, Cells, greedy
from workloads import hotpot_cases


def first_token(lm: LM, tokens: torch.Tensor, cached: Cells) -> torch.Tensor:
    pos = torch.arange(tokens.numel())
    return lm.logits(lm.run(tokens, pos, pos >= len(cached), cached).hidden[-1:]).float().cpu()


def answer(lm: LM, tokens: torch.Tensor, cached: Cells) -> str:
    stop = {lm.tok.eos_token_id, lm.tok.convert_tokens_to_ids("<|im_end|>")}
    return lm.tok.decode(greedy(lm, tokens, torch.arange(tokens.numel()), cached, 16, stop)).strip().split("\n")[0]


def analyse(lm: LM, case) -> dict:
    question_at = case.fresh_from
    P, E = case.part(0, case.edit), case.segments[case.edit]
    docs, question, opener = case.part(case.edit + 1, question_at), case.segments[question_at], case.part(question_at + 1, None)
    nP, nE, nD, nQ = P.numel(), E.numel(), docs.numel(), question.numel()
    with_e = torch.cat((P, E, docs, question))
    original = lm.cold(with_e).cells
    cP, stale_docs_question = original.take(slice(0, nP)), original.take(slice(nP + nE, None))
    without = torch.cat((P, docs, question))
    pos = torch.arange(without.numel())
    closed = lm.run(without, pos, pos >= nP, cP).cells
    truth_all = Cells.cat([cP, closed])
    stale_all = Cells.cat([cP, stale_docs_question])
    full = torch.cat((without, opener))
    kept = torch.cat((with_e, opener))

    splits = {"fresh": nP + nD, "cached": nP + nD + nQ}
    out = {"name": case.name, "answer": case.meta["answer"], "nP": nP, "nE": nE, "nD": nD, "nQ": nQ}
    truth_kept = first_token(lm, kept, original)
    truth = first_token(lm, full, truth_all.take(slice(0, splits["fresh"])))
    reference = answer(lm, full, truth_all.take(slice(0, splits["fresh"])))
    out["deletion"] = {"kl": kl_rows(truth, truth_kept)[0].item(), "same_top1": bool(truth.argmax() == truth_kept.argmax())}
    out["truth_answer"], out["kept_answer"] = reference, answer(lm, kept, original)
    for split, cut in splits.items():
        stale = stale_all.take(slice(0, cut))
        logits = first_token(lm, full, stale)
        reply = answer(lm, full, stale)
        out[split] = {"kl": kl_rows(truth, logits)[0].item(), "same_top1": bool(truth.argmax() == logits.argmax()),
                      "answer": reply, "same_answer": normalise(reply) == normalise(reference)}
    exact_cached = first_token(lm, full, truth_all)
    out["reprefill_consistency"] = kl_rows(truth, exact_cached)[0].item()
    return out


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = hotpot_cases(lm, count, "gold") + hotpot_cases(lm, count, "gold", seed=1)
    stem = f"e1p_paired_{key}_{tag}"
    rows = resume(stem, model=MODELS[key]).get("rows", [])
    done = {r["name"] for r in rows}
    for case in cases:
        if case.name in done:
            continue
        done.add(case.name)
        rows.append(analyse(lm, case))
        if len(rows) % 10 == 0:
            save(stem, {"model": MODELS[key], "rows": rows})
    print("saved", save(stem, {"model": MODELS[key], "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
