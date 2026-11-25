"""E10: repair in long contexts (4K to 16K tokens).

Each case is one HotpotQA question whose ten documents are padded with documents of other
questions up to a target length. The edited document is a supporting document placed after about
a quarter of the documents; the cached question sits right after it or after all documents; only
the assistant opener is new input. The supporting document is deleted. Policies, at 1%, 3% and 10%
of the suffix: boundary window, reader attention, recorded exposure, the bridge score, the cached
question (given) then exposure, the reader segment (found from reader attention) then exposure,
and true deviation x reader attention; also the cached question alone and no repair. Metrics as
in E2c: teacher-forced KL over the first 16 reply tokens and KL at the first reply token.

Usage: e10_long.py <model-key> <questions-per-length> [tag] [lengths, comma separated] [device]
"""
from __future__ import annotations

import random
import sys

import torch

from common import MODELS, best_device, resume, save, release
from e2c_ablation import Reader, budget_count, compare, ranks, reader_attention, reply_of
from e2d_reader_first import first
from e9_scale import deviation
from mutablekv import LM, Cells, heal
from workloads import Case, chat_frame

LENGTHS = (4000, 8000, 16000)
BUDGETS = (0.01, 0.03, 0.10)
PLACES = ("adjacent", "end")
POLICIES = ("boundary", "reader", "exposure", "bridge", "question_exposure", "segment_exposure", "oracle_reader")
RRF_POLICIES = ("reader", "bridge", "rrf", "segment_exposure")  # tag "rrf...": adds rank fusion, with references


def long_cases(lm: LM, count: int, target: int, place: str, seed: int) -> list[Case]:
    from datasets import load_dataset

    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    head, tail = chat_frame(lm)
    text = lambda title, sentences: f"Title: {title}\n{''.join(sentences).strip()}\n\n"
    cases, cursor = [], 0
    for row_index in order:
        if len(cases) == count:
            break
        row = data[row_index]
        titles, sentences = row["context"]["title"], row["context"]["sentences"]
        gold = set(row["supporting_facts"]["title"])
        candidates = [i for i in range(len(titles)) if titles[i] in gold]
        if len(titles) < 8 or not candidates:
            continue
        rng = random.Random(f"{seed}/{row['id']}/{target}")
        edit = rng.choice(candidates)
        docs = [lm.encode(text(titles[i], sentences[i])) for i in range(len(titles)) if i != edit]
        seen = set(titles)
        question = lm.encode(f"Question: {row['question']}\n\n")
        budget = target - sum(d.numel() for d in docs) - question.numel() - 64
        while budget > 0:
            other = data[order[cursor % len(order)]]
            cursor += 1
            for title, sents in zip(other["context"]["title"], other["context"]["sentences"]):
                if title in seen or budget <= 0:
                    continue
                seen.add(title)
                piece = lm.encode(text(title, sents))
                docs.append(piece)
                budget -= piece.numel()
        rng.shuffle(docs)
        total, lead = 0, 0
        while lead < len(docs) and total < 0.25 * target:
            total += docs[lead].numel()
            lead += 1
        after = docs[lead:]
        slot = 0 if place == "adjacent" else len(after)
        segments = [lm.encode(head + "Documents:\n\n")] + docs[:lead] + [lm.encode(text(titles[edit], sentences[edit]))]
        segments += after[:slot] + [question] + after[slot:] + [lm.encode(tail)]
        cases.append(Case(name=f"long/{target}/{row['id']}/{place}", segments=segments, edit=1 + lead, fresh_from=len(segments) - 1,
                          meta={"target": target, "place": place, "answer": row["answer"], "question_segment": 2 + lead + slot,
                                "tokens": sum(s.numel() for s in segments)}))
    return cases


def analyse(lm: LM, case: Case, policies: tuple = POLICIES) -> dict:
    P, E, S, Q = case.part(0, case.edit), case.segments[case.edit], case.part(case.edit + 1, case.fresh_from), case.part(case.fresh_from, None)
    nP, nE, nS = P.numel(), E.numel(), S.numel()
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
    stale_tf, mass = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
    reader_score = reader_attention(mass[:, : Q.numel()], start, n).mean(0)
    oracle = deviation(lm, with_S, truth.take(slice(nP, None))).mean(0)
    del truth

    lengths = [case.segments[i].numel() for i in range(case.edit + 1, case.fresh_from)]
    starts = [sum(lengths[:j]) for j in range(len(lengths))]
    question = case.meta["question_segment"] - (case.edit + 1)
    picked = max(range(len(lengths)), key=lambda j: reader_score[starts[j]:starts[j] + lengths[j]].mean().item())
    mask = lambda j: (torch.arange(nS) >= starts[j]) & (torch.arange(nS) < starts[j] + lengths[j])
    order = torch.arange(nS).float()
    bridge = exposure * reader_score
    scores = {"boundary": -order, "reader": reader_score, "exposure": exposure, "bridge": bridge,
              "rrf": 1 / (60 + ranks(exposure)) + 1 / (60 + ranks(reader_score)),
              "question_exposure": first(mask(question), exposure), "segment_exposure": first(mask(picked), exposure),
              "oracle_reader": oracle * reader_score}
    out = {"name": case.name, **case.meta, "nP": nP, "nE": nE, "nS": nS, "question_tokens": lengths[question],
           "picked_is_question": picked == question, "results": {"stale": compare(truth_tf, stale_tf, reply)}}
    for name in policies:
        out["results"][name] = {}
        for budget in BUDGETS:
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[start + scores[name].topk(budget_count(budget, nS)).indices] = True
            out["results"][name][str(budget)] = compare(truth_tf, reader.logits(heal(lm, context, pos, stale, chosen))[0], reply)
    chosen = torch.zeros(n, dtype=torch.bool)
    chosen[start + starts[question]: start + starts[question] + lengths[question]] = True
    out["results"]["question_only"] = compare(truth_tf, reader.logits(heal(lm, context, pos, stale, chosen))[0], reply)
    return out


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    lengths = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else LENGTHS
    device = sys.argv[5] if len(sys.argv) > 5 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    stem = f"e10_long_{key}_{tag}"
    policies = RRF_POLICIES if tag.startswith("rrf") else POLICIES
    settings = {"model": MODELS[key], "budgets": list(BUDGETS), "policies": list(policies)}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["name"] for r in rows}
    for target in lengths:
        for place in PLACES:
            for case in long_cases(lm, count, target, place, seed=10_000):
                if case.name in done:
                    continue
                rows.append(analyse(lm, case, policies))
                release()
                save(stem, {**settings, "rows": rows})
                print(f"{target} {place}: {sum(r['target'] == target and r['place'] == place for r in rows)} cases", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
