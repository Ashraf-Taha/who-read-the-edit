"""E3: needle erasure, after the protocol of KVEraser (arXiv 2606.17034, Sec. 5 and App. C).

Two needles with the same key and different values are placed in unrelated text:

    One of the special magic numbers for <key> is: <value>.

The earlier needle is erased. The question, which is already cached, asks for the
number(s) for the key. Correct means the reply contains the retained value and no
other number. A reply that still contains the erased value is a ghost read.

Differences from KVEraser's setup, to be stated wherever these numbers are shown:
background text is WikiText-103 (theirs: Paul Graham essays); models are 0.5B to 1.7B
(theirs: 8B); contexts are 1K to 2K tokens (theirs: 1K to 32K).

Usage: e3_needle.py <model-key> <cases-per-size> [tag] [sizes, comma separated]
"""
from __future__ import annotations

import math
import random
import re
import sys

import torch

from common import MODELS, best_device, kl_rows, mean_ci, resume, save, wikitext_articles, release
from mutablekv import LM, Cells, greedy, heal
from e2c_ablation import ranks
from e2d_reader_first import first
from workloads import Case, chat_frame

SIZES = [1000, 2000]
PLAN = {
    "boundary": (0.05, 0.15),
    "tail": (0.05, 0.15),
    "probe1": (0.05, 0.15),
    "reader": (0.02, 0.05, 0.15),
    "exposure": (0.02, 0.05, 0.15),
    "bridge": (0.01, 0.02, 0.05, 0.15),
    "oracle": (0.02, 0.05, 0.15),
}
# Run with a tag starting with "reader": reader-first policies only (the cached question, or the suffix
# segment with the most reader attention per token, first; then exposure or the bridge score).
READER_PLAN = {
    "boundary": (0.02,),
    "tail": (0.02,),
    "probe1": (0.02,),
    "question_exposure": (0.02, 0.05),
    "question_bridge": (0.02, 0.05),
    "segment_exposure": (0.02, 0.05),
    "segment_bridge": (0.02, 0.05),
    "bridge": (0.02, 0.05),
}
# Run with a tag starting with "rrf": reciprocal-rank fusion of exposure and reader attention, with the
# single signals and the bridge score at the same budgets for reference (RRF was not run on this task before).
RRF_PLAN = {
    "reader": (0.02, 0.05),
    "exposure": (0.02, 0.05),
    "bridge": (0.02, 0.05),
    "rrf": (0.02, 0.05),
}
# Run with a tag starting with "fresh": the question is asked after the erasure, as new input, as in
# KVEraser; the stale suffix then holds the retained needle as the reader of the erased one.
FRESH_PLAN = {
    "boundary": (0.02, 0.05, 0.15),
    "tail": (0.05, 0.15),
    "reader": (0.02, 0.05),
    "exposure": (0.02, 0.05),
    "bridge": (0.02, 0.05),
    "segment_exposure": (0.02, 0.05),
    "segment_bridge": (0.02, 0.05),
}
ADJECTIVES = ("amber", "brisk", "cobalt", "dusty", "eager", "fabled", "gilded", "hollow", "ivory", "jagged", "keen", "lunar",
              "mellow", "nimble", "opal", "placid", "quiet", "russet", "silver", "tidal", "umber", "velvet", "wistful", "zesty")
NOUNS = ("anchor", "beacon", "canyon", "dynamo", "ember", "falcon", "glacier", "harbor", "island", "jungle", "kettle", "lantern",
         "meadow", "nebula", "orchard", "pebble", "quarry", "river", "saddle", "thicket", "utensil", "valley", "willow", "zephyr")
NUMBER = re.compile(r"\d+")


def needle_cases(lm: LM, count: int, target_tokens: int, seed: int = 0, fresh: bool = False) -> list[Case]:
    articles = wikitext_articles()
    rng = random.Random(f"{seed}/{target_tokens}")
    head, tail = chat_frame(lm)
    cases = []
    while len(cases) < count:
        article = rng.choice(articles)
        paragraphs, used = [], 0
        for line in article.splitlines():
            if len(line) > 200 and not line.startswith(" ="):
                piece = lm.encode(line.strip() + "\n")
                if used + piece.numel() > target_tokens:
                    break
                paragraphs.append(piece)
                used += piece.numel()
        if len(paragraphs) < 4 or used < 0.7 * target_tokens:
            continue
        first, second = sorted(rng.sample(range(1, len(paragraphs) + 1), 2))
        key = f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}"
        erased, retained = (str(rng.randrange(1_000_000, 10_000_000)) for _ in range(2))
        if erased == retained:
            continue
        needle = lambda value: lm.encode(f"One of the special magic numbers for {key} is: {value}.\n")
        question = lm.encode(f"\nWhat are all the special magic numbers for {key} mentioned in the text above? "
                             "Reply with the number(s) only, separated by commas.")
        between, after = torch.cat(paragraphs[first:second]), paragraphs[second:]
        segments = [torch.cat([lm.encode(head)] + paragraphs[:first]), needle(erased), between, needle(retained)]
        segments += ([torch.cat(after)] if after else []) + [question, lm.encode(tail)]
        retained_start = between.numel()
        question_start = retained_start + segments[3].numel() + (segments[4].numel() if after else 0)
        cases.append(Case(
            name=f"needle/{target_tokens}/{len(cases)}", segments=segments, edit=1, fresh_from=len(segments) - (2 if fresh else 1),
            meta={"size": target_tokens, "key": key, "erased": erased, "retained": retained,
                  "retained_span": (retained_start, retained_start + segments[3].numel()),
                  "question_span": (question_start, question_start + question.numel())},
        ))
    return cases


def relative_deviation(lm: LM, a: Cells, b: Cells) -> torch.Tensor:
    ka, kb = lm.unrotated(a), lm.unrotated(b)
    num = (a.V - b.V).pow(2).sum((1, 3)) + (ka - kb).pow(2).sum((1, 3))
    den = b.V.pow(2).sum((1, 3)) + kb.pow(2).sum((1, 3))
    return (num / den).float().cpu()


def shallow(cells: Cells, depth: int = 2) -> Cells:
    return Cells(cells.tokens, cells.enc_pos, cells.K[:depth], cells.V[:depth])


def grade(text: str, retained: str, erased: str) -> dict:
    """exact: the retained value is the only number in the reply. ghost: the erased value is in it."""
    numbers = set(NUMBER.findall(text))
    return {"text": text, "exact": numbers == {retained}, "ghost": erased in numbers, "has_retained": retained in numbers}


def analyse(lm: LM, case: Case, plan: dict = PLAN) -> dict:
    P, E, S, Q = case.part(0, 1), case.segments[1], case.part(2, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS, nQ = P.numel(), E.numel(), S.numel(), Q.numel()
    span = torch.arange
    stop = {lm.tok.eos_token_id, lm.tok.convert_tokens_to_ids("<|im_end|>")}
    retained, erased = case.meta["retained"], case.meta["erased"]

    def reply(tokens, pos, cells) -> dict:
        text = lm.tok.decode(greedy(lm, tokens, pos, cells, 24, stop)).strip()
        return grade(text, retained, erased)

    groups = torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long()
    original = lm.cold(torch.cat((P, E, S)), groups=groups)
    exposure = original.mass[:, nP + nE:, 1].mean(0)
    cP, stale_S = original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))

    kept, n1 = torch.cat((P, S)), nP + nS
    kept_pos, in_suffix = span(n1), span(n1) >= nP
    truth = Cells.cat([cP, lm.run(kept, kept_pos, in_suffix, cP).cells])
    stale = Cells.cat([cP, stale_S])
    asked, asked_pos = torch.cat((kept, Q)), span(n1 + nQ)
    new_input = asked_pos >= n1

    def first_token(cells: Cells, pos: torch.Tensor = asked_pos, groups: torch.Tensor | None = None):
        result = lm.run(asked, pos, new_input, cells, groups=groups)
        return lm.logits(result.hidden).float().cpu(), result.mass

    truth_logits, _ = first_token(truth)
    stale_logits, reader_mass = first_token(stale, groups=asked_pos)
    reader = reader_mass[:, :, nP:n1].mean((0, 1))
    divergence = lambda logits: kl_rows(truth_logits, logits)[-1].item()

    probe = lm.run(kept, kept_pos, in_suffix, shallow(cP), depth=2).cells
    order = span(nS).float()
    deviation = relative_deviation(lm, stale_S, truth.take(slice(nP, None))).mean(0)
    retained_span, question_span = case.meta["retained_span"], case.meta["question_span"]
    if question_span[1] > nS:
        question_span = (nS, nS)
    spans = {"before": (0, retained_span[0]), "retained": retained_span, "after": (retained_span[1], min(question_span[0], nS)), "question": question_span}
    spans = {name: bounds for name, bounds in spans.items() if bounds[1] > bounds[0]}
    picked = max(spans, key=lambda name: reader[spans[name][0]:spans[name][1]].mean().item())
    mask = lambda bounds: (order >= bounds[0]) & (order < bounds[1])
    bridge = exposure * reader
    scores = {
        "boundary": -order, "tail": order, "reader": reader, "exposure": exposure, "bridge": bridge,
        "rrf": 1 / (60 + ranks(exposure)) + 1 / (60 + ranks(reader)),
        "probe1": relative_deviation(lm, shallow(stale_S), probe)[1], "oracle": deviation,
        "question_exposure": first(mask(question_span), exposure), "question_bridge": first(mask(question_span), bridge),
        "segment_exposure": first(mask(spans[picked]), exposure), "segment_bridge": first(mask(spans[picked]), bridge),
    }

    hole_pos = torch.cat((span(nP), nP + nE + span(nS + nQ)))
    with_e = torch.cat((P, E, S, Q))
    out = {
        "both_present": reply(with_e, span(with_e.numel()), original.cells),
        "full_recompute": {**reply(asked, asked_pos, truth), "kl": 0.0},
        "no_repair": {**reply(asked, asked_pos, stale), "kl": divergence(stale_logits)},
        "hole_no_repair": {**reply(asked, hole_pos, stale), "kl": divergence(first_token(stale, hole_pos)[0])},
    }
    inside = lambda index, bounds: ((index >= bounds[0]) & (index < bounds[1])).float().mean().item()
    picks = {}
    for name, budgets in plan.items():
        for budget in budgets:
            top = scores[name].topk(max(1, math.ceil(budget * nS))).indices
            chosen = torch.zeros(n1, dtype=torch.bool)
            chosen[nP + top] = True
            healed = heal(lm, kept, kept_pos, stale, chosen)
            label = f"{name}@{budget}"
            out[label] = {**reply(asked, asked_pos, healed), "kl": divergence(first_token(healed)[0]), "tokens": int(top.numel())}
            picks[label] = {"on_retained_needle": inside(top, retained_span), "on_question": inside(top, question_span)}

    total = deviation.sum().item()
    return {
        "name": case.name, **case.meta, "nP": nP, "nE": nE, "nS": nS, "nQ": nQ, "out": out, "picks": picks, "picked_segment": picked,
        "deviation_share": {"retained_needle": deviation[retained_span[0]:retained_span[1]].sum().item() / total,
                            "question": deviation[question_span[0]:question_span[1]].sum().item() / total,
                            "first_5_percent": deviation[: max(1, nS // 20)].sum().item() / total},
        "exposure_share": {"retained_needle": exposure[retained_span[0]:retained_span[1]].sum().item() / exposure.sum().item(),
                           "question": exposure[question_span[0]:question_span[1]].sum().item() / exposure.sum().item()},
        "token_share": {"retained_needle": (retained_span[1] - retained_span[0]) / nS, "question": (question_span[1] - question_span[0]) / nS},
    }


def report(rows: list[dict]) -> None:
    for size in sorted({r["size"] for r in rows}):
        chosen = [r for r in rows if r["size"] == size]
        print(f"\n=== about {size} tokens of background / n={len(chosen)} (suffix {mean_ci([r['nS'] for r in chosen])[0]:.0f} tokens)")
        both = mean_ci([float(r["out"]["both_present"]["ghost"] and r["out"]["both_present"]["has_retained"]) for r in chosen])[0]
        print(f"  before the erasure the reply lists both numbers in {both:.0%} of cases")
        print(f"    {'condition':18s} {'exact':>14s} {'ghost read':>12s} {'has retained':>13s} {'KL first token':>16s} {'tokens':>7s}")
        for label in chosen[0]["out"]:
            if label == "both_present":
                continue
            exact, half = mean_ci([float(r["out"][label]["exact"]) for r in chosen])
            ghost = mean_ci([float(r["out"][label]["ghost"]) for r in chosen])[0]
            has = mean_ci([float(r["out"][label]["has_retained"]) for r in chosen])[0]
            kl = mean_ci([r["out"][label]["kl"] for r in chosen])[0]
            tokens = mean_ci([r["out"][label].get("tokens", 0) for r in chosen])[0]
            print(f"    {label:18s} {exact:>7.2f} ± {half:.2f} {ghost:>12.2f} {has:>13.2f} {kl:>16.4f} {tokens:>7.0f}")
        share = lambda field, part: mean_ci([r[field][part] for r in chosen])[0]
        print(f"  the retained needle is {share('token_share', 'retained_needle'):.1%} of suffix tokens and holds "
              f"{share('deviation_share', 'retained_needle'):.1%} of cell deviation and {share('exposure_share', 'retained_needle'):.1%} of recorded attention on the erased needle")
        print(f"  the question is {share('token_share', 'question'):.1%} of suffix tokens and holds "
              f"{share('deviation_share', 'question'):.1%} of cell deviation and {share('exposure_share', 'question'):.1%} of recorded attention on the erased needle")
        print(f"  the first 5% of the suffix holds {share('deviation_share', 'first_5_percent'):.1%} of cell deviation")
        for label in ("boundary@0.15", "tail@0.15", "reader@0.05", "exposure@0.05", "bridge@0.02", "bridge@0.05"):
            if label not in chosen[0]["picks"]:
                continue
            needle = mean_ci([r["picks"][label]["on_retained_needle"] for r in chosen])[0]
            question = mean_ci([r["picks"][label]["on_question"] for r in chosen])[0]
            print(f"    picks of {label:14s}: {needle:.0%} on the retained needle, {question:.0%} on the question")


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    sizes = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else SIZES
    device = sys.argv[5] if len(sys.argv) > 5 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    cases = [case for size in sizes for case in needle_cases(lm, count, size, fresh=tag.startswith("fresh"))]
    plan = (READER_PLAN if tag.startswith("reader") else FRESH_PLAN if tag.startswith("fresh")
            else RRF_PLAN if tag.startswith("rrf") else PLAN)
    stem, settings = f"e3_needle_{key}_{tag}", {"model": MODELS[key], "plan": plan}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}")
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case, plan))
        release()
        if len(rows) % 10 == 0:
            save(stem, {**settings, "rows": rows})
    print("saved", save(stem, {**settings, "rows": rows}))
    report(rows)


if __name__ == "__main__":
    main()
