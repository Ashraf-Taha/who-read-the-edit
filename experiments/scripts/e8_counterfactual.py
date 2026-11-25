"""E8: a counterfactual replacement that the answer must follow.

The layout is E2's (edited document after `LEAD` others, cached question right after it, in the
middle of the remaining documents or after all of them; only the assistant opener is new input),
with a different seed per placement so that every case is a different question. The edited
document E is the one supporting document that contains the answer string; the answer appears
nowhere else in the prompt. A case is kept only if the model, with E as written, answers with
the dataset's answer.

E is then replaced by E', in which every occurrence of the answer is swapped for another answer
from the dataset with the same shape (same number of words; per word: digits of the same length,
capitalised or lower case), one that occurs nowhere in the prompt. The cache holds the suffix S
as encoded after E. Policies rank the tokens of S, the top fraction is recomputed, and the reply is
decoded. A repair succeeds when its reply contains the new answer, as the re-prefill's does.

Exposure is recorded during the original prefill twice: the mass each suffix token put on E as a
whole (what the runtime keeps per segment), and the mass it put on the answer's tokens in E (what a
finer-grained record, per sentence or span, would keep).

Usage: e8_counterfactual.py <model-key> <cases-per-placement> [tag] [device]
"""
from __future__ import annotations

import random
import re
import sys

import torch

from common import MODELS, best_device, normalise, resume, save
from e2_healing import decode, relative_deviation
from e2c_ablation import Reader, budget_count, compare, reader_attention, reply_of
from e2d_reader_first import first
from mutablekv import LM, Cells, heal
from workloads import READER_PLACES, chat_frame

LEAD = 2
BUDGETS = (0.05, 0.10, 0.20)
POLICIES = ("boundary", "tail", "reader", "exposure", "span_exposure", "sum", "bridge", "bridge_span", "question_bridge", "segment_bridge", "oracle_reader")
MAX_TOKENS = 2200


def shape(answer: str) -> tuple:
    return tuple(f"d{len(w)}" if w.isdigit() else ("C" if w[:1].isupper() else "l") for w in answer.split())


def build(lm: LM, row, place: str, pool: dict) -> dict | None:
    answer = row["answer"].strip()
    if len(answer) < 3 or answer.lower() in ("yes", "no"):
        return None
    titles, sentences = row["context"]["title"], row["context"]["sentences"]
    gold = set(row["supporting_facts"]["title"])
    text = lambda i: f"Title: {titles[i]}\n{''.join(sentences[i]).strip()}\n\n"
    holders = [i for i in range(len(titles)) if answer in text(i)]
    if len(titles) < 8 or len(holders) != 1 or titles[holders[0]] not in gold or answer in row["question"]:
        return None
    edit = holders[0]
    prompt_text = "".join(text(i) for i in range(len(titles))) + row["question"]
    options = [a for a in pool.get(shape(answer), []) if a != answer and a not in prompt_text]
    if not options:
        return None
    substitute = random.Random(row["id"]).choice(options)
    others = [i for i in range(len(titles)) if i != edit]
    head, tail = chat_frame(lm)
    after = [lm.encode(text(i)) for i in others[LEAD:]]
    slot = {"adjacent": 0, "middle": len(after) // 2, "end": len(after)}[place]
    P = torch.cat([lm.encode(head + "Documents:\n\n")] + [lm.encode(text(i)) for i in others[:LEAD]])
    S = torch.cat(after[:slot] + [lm.encode(f"Question: {row['question']}\n\n")] + after[slot:])
    Q = lm.encode(tail)
    edited = text(edit)
    encoded = lm.tok(edited, add_special_tokens=False, return_offsets_mapping=True)
    E = torch.tensor(encoded["input_ids"], dtype=torch.long)
    spans = [(m.start(), m.end()) for m in re.finditer(re.escape(answer), edited)]
    in_span = torch.tensor([any(a < e and s < b for s, e in spans) for a, b in encoded["offset_mapping"]])
    E2 = lm.encode(edited.replace(answer, substitute))
    if P.numel() + E.numel() + S.numel() + Q.numel() > MAX_TOKENS or not in_span.any():
        return None
    q_start = sum(t.numel() for t in after[:slot])
    q_len = lm.encode(f"Question: {row['question']}\n\n").numel()
    lengths = [t.numel() for t in after[:slot]] + [q_len] + [t.numel() for t in after[slot:]]
    return {"name": f"counterfactual/{row['id']}/{place}", "place": place, "answer": answer, "substitute": substitute,
            "P": P, "E": E, "E2": E2, "S": S, "Q": Q, "in_span": in_span, "question": (q_start, q_start + q_len), "lengths": lengths}


def analyse(lm: LM, case: dict) -> dict | None:
    P, E, E2, S, Q = case["P"], case["E"], case["E2"], case["S"], case["Q"]
    nP, nE, nE2, nS = P.numel(), E.numel(), E2.numel(), S.numel()
    with_e = torch.cat((P, E, S))
    groups = torch.cat((torch.zeros(nP), 1 + case["in_span"].float(), torch.full((nS,), 3))).long()
    original = lm.cold(with_e, groups=groups)
    reply_before = decode(lm, torch.cat((with_e, Q)), torch.arange(with_e.numel() + Q.numel()), original.cells)
    if normalise(case["answer"]) not in normalise(reply_before):
        return None
    mass = original.mass[:, nP + nE:]
    exposure_by_layer, span_by_layer = mass[:, :, 1] + mass[:, :, 2], mass[:, :, 2]
    cP, with_S = original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))
    del original

    context = torch.cat((P, E2, S))
    start, n = nP + nE2, nP + nE2 + nS
    pos = torch.arange(n)
    cE2 = lm.run(torch.cat((P, E2)), torch.arange(nP + nE2), torch.arange(nP + nE2) >= nP, cP).cells
    truth = Cells.cat([cP, cE2, lm.run(context, pos, pos >= start, Cells.cat([cP, cE2])).cells])
    stale = Cells.cat([cP, cE2, with_S])
    ids, truth_answer = reply_of(lm, context, Q, truth)
    reader = Reader(lm, context, Q, ids)
    truth_tf, _ = reader.logits(truth)
    stale_tf, attention = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
    reader_score = reader_attention(attention[:, : Q.numel()], start, n).mean(0)
    exposure, span = exposure_by_layer.mean(0), span_by_layer.mean(0)
    oracle = relative_deviation(lm, with_S, truth.take(slice(start, None))).mean(0)
    order = torch.arange(nS).float()
    rank = lambda x: x.argsort(descending=True).argsort().float()
    bridge = exposure * reader_score
    starts = [sum(case["lengths"][:j]) for j in range(len(case["lengths"]))]
    picked = max(range(len(starts)), key=lambda j: reader_score[starts[j]:starts[j] + case["lengths"][j]].mean().item())
    in_picked, in_question = torch.zeros(nS, dtype=torch.bool), torch.zeros(nS, dtype=torch.bool)
    in_picked[starts[picked]:starts[picked] + case["lengths"][picked]] = True
    in_question[case["question"][0]:case["question"][1]] = True
    scores = {"boundary": -order, "tail": order, "reader": reader_score, "exposure": exposure, "span_exposure": span,
              "sum": -(rank(exposure) + rank(reader_score)), "bridge": bridge, "bridge_span": span * reader_score,
              "question_bridge": first(in_question, bridge), "segment_bridge": first(in_picked, bridge), "oracle_reader": oracle * reader_score}
    follows = lambda text: normalise(case["substitute"]) in normalise(text)
    keeps = lambda text: normalise(case["answer"]) in normalise(text)
    stale_answer = decode(lm, torch.cat((context, Q)), torch.arange(n + Q.numel()), stale)
    out = {"name": case["name"], "place": case["place"], "answer": case["answer"], "substitute": case["substitute"],
           "nP": nP, "nE": nE, "nE2": nE2, "nS": nS, "span_tokens": int(case["in_span"].sum()), "picked_is_question": bool(in_picked[in_question].all()),
           "reply_before": reply_before, "truth": {"answer": truth_answer, "follows": follows(truth_answer), "keeps_old": keeps(truth_answer)},
           "stale": {"answer": stale_answer, "follows": follows(stale_answer), "keeps_old": keeps(stale_answer), **compare(truth_tf, stale_tf, ids)},
           "policies": {}}
    for name in POLICIES:
        out["policies"][name] = {}
        for budget in BUDGETS:
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[start + scores[name].topk(budget_count(budget, nS)).indices] = True
            healed = heal(lm, context, pos, stale, chosen)
            answer = decode(lm, torch.cat((context, Q)), torch.arange(n + Q.numel()), healed)
            out["policies"][name][str(budget)] = {"answer": answer, "follows": follows(answer), "keeps_old": keeps(answer),
                                                  **compare(truth_tf, reader.logits(healed)[0], ids)}
    q0, q1 = case["question"]
    chosen = torch.zeros(n, dtype=torch.bool)
    chosen[start + q0: start + q1] = True
    healed = heal(lm, context, pos, stale, chosen)
    answer = decode(lm, torch.cat((context, Q)), torch.arange(n + Q.numel()), healed)
    out["question_only"] = {"answer": answer, "follows": follows(answer), "keeps_old": keeps(answer), "tokens": q1 - q0,
                            **compare(truth_tf, reader.logits(healed)[0], ids)}
    return out


def main() -> None:
    from datasets import load_dataset

    key, per_place = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    pool: dict = {}
    for answer in sorted(set(data["answer"])):
        if len(answer) >= 3 and answer.lower() not in ("yes", "no"):
            pool.setdefault(shape(answer), []).append(answer)
    stem = f"e8_counterfactual_{key}_{tag}"
    settings = {"model": MODELS[key], "lead": LEAD, "budgets": list(BUDGETS), "policies": list(POLICIES)}
    saved = resume(stem, **settings)
    rows, tried = saved.get("rows", []), saved.get("tried", {})
    for index, place in enumerate(READER_PLACES):
        order = list(range(len(data)))
        random.Random(8000 + 100 * index).shuffle(order)
        kept = sum(r["place"] == place for r in rows)
        for row_index in order:
            if kept >= per_place:
                break
            row = data[row_index]
            name = f"counterfactual/{row['id']}/{place}"
            if name in tried:
                continue
            case = build(lm, row, place, pool)
            result = analyse(lm, case) if case else None
            tried[name] = "kept" if result else ("unanswered" if case else "unsuitable")
            if result:
                rows.append(result)
                kept += 1
                save(stem, {**settings, "rows": rows, "tried": tried})
                print(f"{place}: {kept} cases ({sum(v != 'unsuitable' for k, v in tried.items() if k.endswith(place))} answerable candidates tried)", flush=True)
    save(stem, {**settings, "rows": rows, "tried": tried})
    follow = lambda part, key, budget: sum(r["policies"][key][budget]["follows"] for r in part)
    adopting = [r for r in rows if r["truth"]["follows"]]
    print(f"{len(rows)} cases; the re-prefill follows the edit in {len(adopting)}; no repair follows in {sum(r['stale']['follows'] for r in adopting)} of those")
    for name in POLICIES:
        print(f"  {name:14s} " + "  ".join(f"{budget}: {follow(adopting, name, str(budget))}" for budget in BUDGETS))


if __name__ == "__main__":
    main()
