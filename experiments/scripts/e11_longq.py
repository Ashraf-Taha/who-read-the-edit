"""E11: a long new input. Can reader attention be taken from a few of its tokens?

E2c's supporting-document deletions with the edited document 5th of ten (same seed, so the same
cases as E2c's), but the new input is long: WikiText paragraphs of about `L` tokens, appended to
the user turn as notes, followed by the assistant opener. Measuring reader attention over the
stale cache with the whole new input costs a pass of all its tokens before the repair, and the
answer then needs a second pass. Estimates of reader attention compared here:

    full        every token of the new input, over the stale cache (the reference)
    first64     its first 64 tokens only (exact for those tokens; costs 64 tokens)
    last16      its last 16 tokens, run alone at their true positions without the rest of the
                new input in context (an approximation; costs 16 tokens)
    last64      the same with the last 64 tokens

Each estimate drives two repairs at 10% of the suffix: the reader segment it finds then exposure,
and the bridge score. Metric: teacher-forced KL over the first 16 reply tokens after the full new
input. Also recorded: the rank correlation of each estimate with the full one and whether it finds
the same segment.

Usage: e11_longq.py <model-key> <questions-per-place> [tag] [lengths, comma separated] [device]
"""
from __future__ import annotations

import random
import sys

import torch
from scipy.stats import spearmanr

from common import MODELS, best_device, resume, save, wikitext_articles
from e2c_ablation import Reader, budget_count, compare, reply_of
from e2d_reader_first import first
from mutablekv import LM, Cells, heal
from workloads import READER_PLACES, reader_cases

LEAD = 4
BUDGET = 0.10
LENGTHS = (512, 2048)
ESTIMATES = ("full", "first64", "last16", "last64")


def notes(lm: LM, target: int, seed: str) -> torch.Tensor:
    rng = random.Random(seed)
    articles = wikitext_articles()
    pieces, used = [lm.encode("\nNotes from an earlier tool call:\n")], 0
    while used < target:
        for line in rng.choice(articles).splitlines():
            if len(line) > 200 and not line.startswith(" ="):
                piece = lm.encode(line.strip() + "\n")
                pieces.append(piece)
                used += piece.numel()
                if used >= target:
                    break
    return torch.cat(pieces)


def attention_of(lm: LM, context: torch.Tensor, cells: Cells, new: torch.Tensor, new_pos: torch.Tensor, start: int) -> torch.Tensor:
    """Mean attention of the tokens `new` (at positions `new_pos`) on context[start:], over layers, heads and tokens."""
    tokens = torch.cat((context, new))
    pos = torch.cat((torch.arange(context.numel()), new_pos))
    result = lm.run(tokens, pos, torch.arange(tokens.numel()) >= context.numel(), cells, groups=torch.arange(tokens.numel()))
    return result.mass[:, :, start:context.numel()].mean((0, 1))


def analyse(lm: LM, case, length: int) -> dict:
    P, E, S = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from)
    opener = case.part(case.fresh_from, None)
    Q = torch.cat((notes(lm, length, case.name), opener))
    nP, nE, nS, nQ = P.numel(), E.numel(), S.numel(), Q.numel()
    groups = torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long()
    original = lm.cold(torch.cat((P, E, S)), groups=groups)
    exposure = original.mass[:, nP + nE:, 1].mean(0)
    cP, with_S = original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))
    del original

    context, start, n = torch.cat((P, S)), nP, nP + nS
    pos = torch.arange(n)
    truth = Cells.cat([cP, lm.run(context, pos, pos >= nP, cP).cells])
    stale = Cells.cat([cP, with_S])
    reply, _ = reply_of(lm, context, Q, truth)
    reader = Reader(lm, context, Q, reply)
    truth_tf, _ = reader.logits(truth)
    stale_tf, _ = reader.logits(stale)
    q_pos = n + torch.arange(nQ)
    estimates = {
        "full": attention_of(lm, context, stale, Q, q_pos, start),
        "first64": attention_of(lm, context, stale, Q[:64], q_pos[:64], start),
        "last16": attention_of(lm, context, stale, Q[-16:], q_pos[-16:], start),
        "last64": attention_of(lm, context, stale, Q[-64:], q_pos[-64:], start),
    }
    lengths = [case.segments[i].numel() for i in range(case.edit + 1, case.fresh_from)]
    starts = [sum(lengths[:j]) for j in range(len(lengths))]
    question = case.meta["question_segment"] - (case.edit + 1)
    mask = lambda j: (torch.arange(nS) >= starts[j]) & (torch.arange(nS) < starts[j] + lengths[j])
    out = {"name": case.name, "place": case.meta["place"], "length": length, "nS": nS, "nQ": nQ,
           "results": {"stale": compare(truth_tf, stale_tf, reply)}, "agreement": {}, "picked_is_question": {}}
    picks = {}
    for name, score in estimates.items():
        picks[name] = max(range(len(lengths)), key=lambda j: score[starts[j]:starts[j] + lengths[j]].mean().item())
        out["picked_is_question"][name] = picks[name] == question
        out["agreement"][name] = {"spearman": spearmanr(score, estimates["full"]).statistic, "same_segment": picks[name] == picks["full"]}
        for policy, priority in (("segment_exposure", first(mask(picks[name]), exposure)), ("bridge", exposure * score)):
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[start + priority.topk(budget_count(BUDGET, nS)).indices] = True
            out["results"][f"{policy}/{name}"] = compare(truth_tf, reader.logits(heal(lm, context, pos, stale, chosen))[0], reply)
    return out


def main() -> None:
    key, per_place = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    lengths = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else LENGTHS
    device = sys.argv[5] if len(sys.argv) > 5 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = []
    for place in READER_PLACES:
        for case in reader_cases(lm, per_place, "gold", place, seed=1000 + LEAD, lead=LEAD):
            case.name += f"/lead{LEAD}"
            cases.append(case)
    stem = f"e11_longq_{key}_{tag}"
    settings = {"model": MODELS[key], "budget": BUDGET, "estimates": list(ESTIMATES)}
    rows = resume(stem, **settings).get("rows", [])
    done = {(r["name"], r["length"]) for r in rows}
    for length in lengths:
        for case in cases:
            if (case.name, length) in done:
                continue
            rows.append(analyse(lm, case, length))
            if len(rows) % 6 == 0:
                save(stem, {**settings, "rows": rows})
                print(f"{len(rows)} runs", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
