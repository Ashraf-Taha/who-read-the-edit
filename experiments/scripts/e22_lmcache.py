"""E22: CacheBlend's selection exactly as LMCache runs it, next to our first-layer criterion.

LMCache's in-process blender (github.com/LMCache/LMCache_inprocess, commit 0aebefd,
lmcache/inprocess/v1/compute/blend/blender.py) ranks tokens at its check layer (index 1 by default)
by the squared difference between the new keys and the cached ones, summed over the whole key vector,
and recomputes the top share in every later layer. Our `probe1` (E2c) also counts the values and divides
by the norm of the cells. This run scores both on E2c's supporting-document deletions at 10% of the
suffix, together with the bridge score, which must reproduce E2c's numbers exactly.

Keys are compared without rotation: relocation is exact in our engine, so this equals comparing keys
rotated to the same positions.

Usage: e22_lmcache.py <model-key> [tag] [device]
"""
from __future__ import annotations

import sys

import torch

from common import MODELS, best_device, release, resume, save
from e2_healing import relative_deviation, shallow
from e2c_ablation import Reader, budget_count, build_cases, compare, reader_attention, reply_of
from mutablekv import LM, Cells, heal

BUDGET = 0.10


def analyse(lm: LM, case) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
    groups = torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long()
    original = lm.cold(torch.cat((P, E, S)), groups=groups)
    exposure = original.mass[:, nP + nE:, 1].mean(0)
    cP, with_S = original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))
    del original
    context, n = torch.cat((P, S)), nP + nS
    pos = torch.arange(n)
    truth = Cells.cat([cP, lm.run(context, pos, pos >= nP, cP).cells])
    stale = Cells.cat([cP, with_S])
    reply, _ = reply_of(lm, context, Q, truth)
    reader = Reader(lm, context, Q, reply)
    truth_tf, _ = reader.logits(truth)
    stale_tf, mass = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
    reader_score = reader_attention(mass[:, : Q.numel()], nP, n).mean(0)
    probe_cells = lm.run(context, pos, pos >= nP, shallow(cP), depth=2).cells
    ours = relative_deviation(lm, shallow(with_S), probe_cells)[1]
    lmcache = (lm.unrotated(shallow(with_S)).float() - lm.unrotated(probe_cells).float()).pow(2).sum((1, 3))[1].cpu()
    scores = {"probe1": ours, "probe1_lmcache": lmcache, "bridge": exposure * reader_score}
    count = budget_count(BUDGET, nS)
    results = {"stale": compare(truth_tf, stale_tf, reply)}
    picked = {}
    for name, score in scores.items():
        top = score.topk(count).indices
        picked[name] = set(top.tolist())
        chosen = torch.zeros(n, dtype=torch.bool)
        chosen[nP + top] = True
        results[name] = compare(truth_tf, reader.logits(heal(lm, context, pos, stale, chosen))[0], reply)
    return {"name": case.name, "nS": nS, "budget_tokens": count, "results": results,
            "overlap": len(picked["probe1"] & picked["probe1_lmcache"]) / count}


def main() -> None:
    key = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "main"
    device = sys.argv[3] if len(sys.argv) > 3 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = [case for case in build_cases(lm, 12, 3) if case.meta.get("kind") == "gold"]
    stem = f"e22_lmcache_{key}_{tag}"
    settings = {"model": MODELS[key], "budget": BUDGET, "source": "LMCache_inprocess@0aebefd"}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}", flush=True)
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case))
        save(stem, {**settings, "rows": rows})
        release()
        print(f"{len(rows)} cases", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
