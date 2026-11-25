"""Fill a cell E2 left empty: answers after repairing by reader attention at 10% on Qwen2.5-0.5B.

E2's run on Qwen2.5-0.5B decoded answers for fewer policies than the later runs. This rebuilds
exactly E2's supporting-document cases (seed 0, edited document third), recomputes the
reader-attention repair at 10% with E2's token count, checks that the first-token KL reproduces
the value E2 saved for the same case, and decodes the answer.

Usage: e2_fill.py <model-key> <supporting-per-place> [tag] [device]
"""
from __future__ import annotations

import json
import math
import sys

import torch

from common import MODELS, RESULTS, best_device, kl_rows, save
from e2_healing import decode
from mutablekv import LM, Cells, heal
from workloads import READER_PLACES, reader_cases


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    saved = {r["name"]: r for r in json.loads((RESULTS / f"e2_healing_{key}_main.json").read_text())["rows"]}
    rows = []
    for place in READER_PLACES:
        for case in reader_cases(lm, count, "gold", place):
            P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
            nP, nE, nS = P.numel(), E.numel(), S.numel()
            cells = lm.cold(torch.cat((P, E, S))).cells
            cP, with_S = cells.take(slice(0, nP)), cells.take(slice(nP + nE, None))
            context, n = torch.cat((P, S)), nP + nS
            pos = torch.arange(n)
            truth = Cells.cat([cP, lm.run(context, pos, pos >= nP, cP).cells])
            stale = Cells.cat([cP, with_S])
            asked, asked_pos = torch.cat((context, Q)), torch.arange(n + Q.numel())
            read = lambda c, groups=None: lm.run(asked, asked_pos, asked_pos >= n, c, groups=groups)
            truth_logits = lm.logits(read(truth).hidden).float().cpu()
            reader = read(stale, groups=asked_pos).mass[:, :, nP:n].mean((0, 1))
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[nP + reader.topk(max(1, math.ceil(0.10 * nS))).indices] = True
            healed = heal(lm, context, pos, stale, chosen)
            kl = kl_rows(truth_logits, lm.logits(read(healed).hidden).float().cpu())[-1].item()
            stored = saved[case.name]["delete"]["results"]["reader"]["0.1"]["kl_last"]
            rows.append({"name": case.name, "place": place, "kl_last": kl, "stored_kl_last": stored,
                         "answer": decode(lm, asked, asked_pos, healed), "truth": decode(lm, asked, asked_pos, truth)})
            print(f"{case.name}: KL {kl:.6f} (saved {stored:.6f})", flush=True)
    print("saved", save(f"e2_fill_{key}_{tag}", {"model": MODELS[key], "rows": rows}))
    print("largest difference from the saved KL:", max(abs(r["kl_last"] - r["stored_kl_last"]) for r in rows))


if __name__ == "__main__":
    main()
