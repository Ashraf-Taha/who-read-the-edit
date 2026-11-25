"""E9: the repair policies on a 7B model, on E2c's cases.

Same supporting-document deletions as E2c (edited document 2nd, 5th or 8th of ten; cached
question right after it, in the middle or at the end; 12 questions per position). Qwen2.5-7B
shares the tokenizer of the smaller Qwen models, so the cases are the same token for token.
The model runs in bfloat16 (it does not fit in 24 GB in float32). Policies at 10% of the suffix:
no repair, boundary and tail windows, reader attention, recorded exposure, reciprocal-rank
fusion, the bridge score, the cached question (given) then exposure, the reader segment (found
from reader attention) then exposure, and true deviation x reader attention (ProphetKV's ideal);
also the cached question alone. Metrics as in E2c: teacher-forced KL over the first 16 reply
tokens and KL at the first token. In bfloat16 two exact computations differ by rounding: the
`floor` entry compares a cold pass over the edited context with the re-prefill that reads the
prefix from cache, which are the same in exact arithmetic.

Usage: e9_scale.py <model-key> <supporting-per-cell> [tag] [dtype] [device]
"""
from __future__ import annotations

import sys

import torch

from common import MODELS, best_device, resume, save, release
from e2_healing import decode
from e2c_ablation import BUDGETS, Reader, budget_count, build_cases, compare, policy_scores, reader_attention, reply_of
from e2d_reader_first import first
from mutablekv import LM, Cells, heal

POLICIES = ("boundary", "tail", "reader", "exposure", "rrf", "bridge", "question_exposure", "segment_exposure", "oracle_reader")
DECODED = ("bridge", "segment_exposure")
BUDGET = 0.10


def deviation(lm: LM, a: Cells, b: Cells) -> torch.Tensor:
    ka, kb = lm.unrotated(a).float(), lm.unrotated(b).float()
    va, vb = a.V.float(), b.V.float()
    num = (va - vb).pow(2).sum((1, 3)) + (ka - kb).pow(2).sum((1, 3))
    den = vb.pow(2).sum((1, 3)) + kb.pow(2).sum((1, 3))
    return (num / den).cpu()


@torch.inference_mode()
def reference_check(lm: LM, case, length: int = 512) -> dict:
    """Our forward pass against the library's on the first `length` tokens of a prompt, in the model's dtype."""
    tokens = torch.cat(case.segments)[:length]
    ours = lm.logits(lm.cold(tokens).hidden).float().cpu()
    theirs = lm.hf(tokens[None].to(lm.device)).logits[0].float().cpu()
    return {"tokens": int(tokens.numel()), "max_abs_logit_gap": (ours - theirs).abs().max().item(),
            "same_argmax": (ours.argmax(-1) == theirs.argmax(-1)).float().mean().item()}


def analyse(lm: LM, case) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    groups = torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long()
    original = lm.cold(torch.cat((P, E, S)), groups=groups)
    exposure_by_layer = original.mass[:, nP + nE:, 1]
    cP, with_S = original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))
    del original

    context, start, n = torch.cat((P, S)), nP, nP + nS
    pos = torch.arange(n)
    truth = Cells.cat([cP, lm.run(context, pos, pos >= nP, cP).cells])
    stale = Cells.cat([cP, with_S])
    reply, truth_answer = reply_of(lm, context, Q, truth)
    reader = Reader(lm, context, Q, reply)
    truth_tf, _ = reader.logits(truth)
    stale_tf, mass = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
    floor_tf, _ = reader.logits(lm.cold(context).cells)
    reader_by_layer = reader_attention(mass[:, : Q.numel()], start, n)
    oracle = deviation(lm, with_S, truth.take(slice(nP, None))).mean(0)
    scores, (q0, q1) = policy_scores(lm, case, nS, exposure_by_layer, reader_by_layer, None, oracle)
    lengths = [case.segments[i].numel() for i in range(case.edit + 1, case.fresh_from)]
    starts = [sum(lengths[:j]) for j in range(len(lengths))]
    reader_score = reader_by_layer.mean(0)
    picked = max(range(len(lengths)), key=lambda j: reader_score[starts[j]:starts[j] + lengths[j]].mean().item())
    in_picked = torch.zeros(nS, dtype=torch.bool)
    in_picked[starts[picked]:starts[picked] + lengths[picked]] = True
    scores["segment_exposure"] = first(in_picked, scores["exposure"])
    question = case.meta["question_segment"] - (case.edit + 1)

    out = {"name": case.name, **{k: v for k, v in case.meta.items() if k != "edit_text"}, "nP": nP, "nE": nE, "nS": nS,
           "picked_is_question": picked == question, "results": {"stale": compare(truth_tf, stale_tf, reply), "floor": compare(truth_tf, floor_tf, reply)},
           "answers": {"truth": truth_answer, "stale": decode(lm, torch.cat((context, Q)), torch.arange(n + Q.numel()), stale)}}
    for name in POLICIES:
        chosen = torch.zeros(n, dtype=torch.bool)
        chosen[start + scores[name].topk(budget_count(BUDGET, nS)).indices] = True
        healed = heal(lm, context, pos, stale, chosen)
        out["results"][name] = compare(truth_tf, reader.logits(healed)[0], reply)
        if name in DECODED:
            out["answers"][name] = decode(lm, reader.tokens[: reader.n + reader.nQ], reader.pos[: reader.n + reader.nQ], healed)
    chosen = torch.zeros(n, dtype=torch.bool)
    chosen[start + q0: start + q1] = True
    out["results"]["question_only"] = compare(truth_tf, reader.logits(heal(lm, context, pos, stale, chosen))[0], reply)
    return out


def main() -> None:
    key, supporting = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[sys.argv[4] if len(sys.argv) > 4 else "bf16"]
    device = sys.argv[5] if len(sys.argv) > 5 else best_device()
    lm = LM(MODELS[key], device, dtype)
    cases = build_cases(lm, supporting, 0)
    stem = f"e9_scale_{key}_{tag}"
    settings = {"model": MODELS[key], "dtype": str(dtype), "budget": BUDGET, "policies": list(POLICIES)}
    saved = resume(stem, **settings)
    rows = saved.get("rows", [])
    check = saved.get("reference_check") or reference_check(lm, cases[0])
    print("engine vs reference implementation:", check, flush=True)
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}", flush=True)
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case))
        release()
        if len(rows) % 5 == 0:
            save(stem, {**settings, "reference_check": check, "rows": rows})
            print(f"{len(rows)} cases", flush=True)
    print("saved", save(stem, {**settings, "reference_check": check, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
