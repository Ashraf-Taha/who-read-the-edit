"""E5: out-of-order arrival and branch merge, driven through the runtime layer.

Part A, arrival. A question is followed by k documents that have a canonical order
(the order in which they were requested) but arrive in a random order. Compared:

    wait        encode nothing until every document is in, then prefill in canonical order
    append      encode each document when it arrives, in arrival order (exact, but a different context)
    speculate   encode each document when it arrives, at its canonical place; documents that were
                already encoded after that place keep their cells and are marked as missing a read
      + repair  recompute a share of the stale tokens, ranked by reader attention times the
                estimated attention each token would have paid to content it never saw
                ("sketch", from kept queries) or times a flat prior ("flat", which ranks by
                reader attention alone, ProphetKV's signal)
      + sweep   recompute stale segments left to right until the state is exact

Fidelity is measured against the cold pass over the canonical order. Work is counted in
tokens that must be encoded after the last document arrives and before the first output token.

Part B, merge. Two branches fork from a shared prefix; each reads its own documents and
writes a short note. The branches are merged and a question is asked.

Usage: e5_streams.py <model-key> <cases> [tag] [device]
"""
from __future__ import annotations

import random
import sys

import torch

from common import MODELS, answer_f1, best_device, kl_rows, mean_ci, normalise, resume, save
from mutablekv import LM, Context, greedy
from workloads import chat_frame

DOCS = 6
REPAIR_SHARES = (0.1, 0.2)
ESTIMATES = ("sketch", "flat")
READER_FIRST = (False, True)
NOTE_TOKENS = 40


def hotpot_rows(count: int, seed: int = 0, docs: int = DOCS):
    from datasets import load_dataset

    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    picked = []
    for index in order:
        row = data[index]
        titles, sentences = row["context"]["title"], row["context"]["sentences"]
        gold = [i for i, t in enumerate(titles) if t in set(row["supporting_facts"]["title"])]
        other = [i for i in range(len(titles)) if i not in gold]
        if len(gold) != 2 or len(other) < docs - 2:
            continue
        chosen = gold + other[: docs - 2]
        random.Random(f"{seed}/{row['id']}").shuffle(chosen)
        picked.append({"id": row["id"], "question": row["question"], "answer": row["answer"],
                       "docs": [f"Title: {titles[i]}\n{''.join(sentences[i]).strip()}\n\n" for i in chosen],
                       "gold": [chosen.index(i) for i in gold]})
        if len(picked) == count:
            break
    return picked


def inversions(order: list[int]) -> int:
    return sum(a > b for i, a in enumerate(order) for b in order[i + 1:])


def outcome(reference: torch.Tensor, logits: torch.Tensor) -> dict:
    divergence = kl_rows(reference.float().cpu(), logits.float().cpu())
    return {"kl_last": divergence[-1].item(), "top1_last": bool(reference[-1].argmax() == logits[-1].argmax())}


def answer(lm: LM, context: Context, reader: torch.Tensor) -> str:
    stop = {lm.tok.eos_token_id, lm.tok.convert_tokens_to_ids("<|im_end|>")}
    tokens = torch.cat((context.tokens(), reader))
    return lm.tok.decode(greedy(lm, tokens, torch.arange(tokens.numel()), context.cells(), 16, stop)).strip().split("\n")[0]


def arrival_case(lm: LM, row: dict, seed: int) -> dict:
    head, tail = chat_frame(lm)
    opening = lm.encode(head + f"Question: {row['question']}\n\nDocuments:\n\n")
    docs = [lm.encode(d) for d in row["docs"]]
    reader = lm.encode(tail)
    arrival = list(range(len(docs)))
    random.Random(f"{seed}/{row['id']}").shuffle(arrival)

    def build(order: list[int], canonical: bool) -> Context:
        context = Context(lm)
        context.put("opening", -1, opening)
        for step, doc in enumerate(order):
            context.put(f"doc{doc}", doc if canonical else step, docs[doc])
        return context

    truth = build(sorted(arrival), canonical=True)
    assert truth.frontier() == len(docs) + 1
    truth_logits, _ = truth.read(reader)
    truth_answer = answer(lm, truth, reader)

    appended = build(arrival, canonical=False)
    assert appended.frontier() == len(docs) + 1

    speculative = build(arrival, canonical=True)
    assert speculative.keys() == truth.keys()
    stale_tokens = speculative.stale_tokens()
    total = sum(d.numel() for d in docs)
    result = {
        "id": row["id"], "answer": row["answer"], "arrival": arrival, "inversions": inversions(arrival),
        "doc_tokens": total, "last_doc_tokens": docs[arrival[-1]].numel(), "stale_tokens": stale_tokens,
        "doc_lengths": [d.numel() for d in docs], "reader_tokens": reader.numel(), "opening_tokens": opening.numel(),
        "stale_segments": len(speculative.segments) - speculative.frontier(),
        "truth_answer": truth_answer,
        "append": {**outcome(truth_logits, appended.read(reader)[0]), "answer": answer(lm, appended, reader)},
        "speculate": {**outcome(truth_logits, speculative.read(reader)[0]), "answer": answer(lm, speculative, reader)},
    }
    for share in REPAIR_SHARES:
        for estimate in ESTIMATES:
            for first in READER_FIRST:
                repaired = speculative.fork()
                spent = repaired.repair(reader, share, estimate=estimate, reader_first=first)
                result[f"repair_{estimate}{'_first' if first else ''}@{share}"] = {
                    **outcome(truth_logits, repaired.read(reader)[0]), "answer": answer(lm, repaired, reader), "tokens": spent}
    swept = speculative.fork()
    spent = swept.sweep()
    result["sweep"] = {**outcome(truth_logits, swept.read(reader)[0]), "answer": answer(lm, swept, reader), "tokens": spent}
    if result["sweep"]["kl_last"] > 1e-4:
        raise AssertionError(f"a full sweep should reproduce the cold pass, KL={result['sweep']['kl_last']}")
    return result


def merge_case(lm: LM, row: dict) -> dict:
    head, tail = chat_frame(lm)
    stop = {lm.tok.eos_token_id, lm.tok.convert_tokens_to_ids("<|im_end|>")}
    base = Context(lm)
    base.put("opening", 0, lm.encode(head + "Two researchers each read some documents and wrote a note.\n\n"))
    half = len(row["docs"]) // 2
    branches = []
    for b, (name, doc_ids) in enumerate((("A", range(half)), ("B", range(half, len(row["docs"]))))):
        branch = base.fork()
        for doc in doc_ids:
            branch.put(f"doc{doc}", 10 * (b + 1) + doc, lm.encode(row["docs"][doc]))
        prompt = lm.encode(f"Note from researcher {name}, one sentence on what these documents say:")
        tokens = torch.cat((branch.tokens(), prompt))
        written = greedy(lm, tokens, torch.arange(tokens.numel()), branch.cells(), NOTE_TOKENS, stop | {int(lm.encode("\n")[-1])})
        branch.put(f"note{name}", 10 * (b + 1) + 9, torch.cat((prompt, torch.tensor(written, dtype=torch.long), lm.encode("\n\n"))))
        branches.append(branch)

    merged = branches[0].fork()
    merged.merge(branches[1])
    reader = lm.encode(f"Question: {row['question']}" + tail)
    truth = merged.fork()
    truth.sweep()
    truth_logits, _ = truth.read(reader)
    truth_answer = answer(lm, truth, reader)
    result = {
        "id": row["id"], "answer": row["answer"], "tokens": len(merged), "stale_tokens": merged.stale_tokens(), "reader_tokens": reader.numel(),
        "shared_segments": sum(a is b for a, b in zip(merged.segments, branches[0].segments)),
        "truth_answer": truth_answer,
        "merge": {**outcome(truth_logits, merged.read(reader)[0]), "answer": answer(lm, merged, reader)},
    }
    for share in REPAIR_SHARES:
        for estimate in ESTIMATES:
            for first in READER_FIRST:
                repaired = merged.fork()
                spent = repaired.repair(reader, share, estimate=estimate, reader_first=first)
                result[f"repair_{estimate}{'_first' if first else ''}@{share}"] = {
                    **outcome(truth_logits, repaired.read(reader)[0]), "answer": answer(lm, repaired, reader), "tokens": spent}
    return result


def line(rows: list[dict], key: str) -> str:
    kl, half = mean_ci([r[key]["kl_last"] for r in rows])
    top = mean_ci([float(r[key]["top1_last"]) for r in rows])[0]
    same = mean_ci([float(normalise(r[key]["answer"]) == normalise(r["truth_answer"])) for r in rows])[0]
    f1 = mean_ci([answer_f1(r[key]["answer"], r["truth_answer"]) for r in rows])[0]
    return f"KL {kl:.4f} ± {half:.4f}   first token agrees {top:.2f}   answer identical {same:.2f}   answer F1 {f1:.2f}"


def report(arrival: list[dict], merge: list[dict]) -> None:
    if arrival:
        mean = lambda field: mean_ci([r[field] for r in arrival])[0]
        print(f"\n=== A. {DOCS} documents arriving out of order / n={len(arrival)} "
              f"(documents {mean('doc_tokens'):.0f} tokens in total, mean inversions {mean('inversions'):.1f} of {DOCS * (DOCS - 1) // 2})")
        print(f"  tokens to encode after the last arrival: wait {mean('doc_tokens'):.0f} | append or speculate {mean('last_doc_tokens'):.0f} "
              f"| stale when the last document lands {mean('stale_tokens'):.0f}")
        for key in ("append", "speculate", *[f"repair_{e}{'_first' if f else ''}@{s}" for s in REPAIR_SHARES for e in ESTIMATES for f in READER_FIRST], "sweep"):
            extra = f"   (+{mean_ci([r[key]['tokens'] for r in arrival])[0]:.0f} tokens)" if "tokens" in arrival[0][key] else ""
            print(f"    {key:12s} {line(arrival, key)}{extra}")
        low = [r for r in arrival if r["inversions"] <= 5]
        high = [r for r in arrival if r["inversions"] > 9]
        for name, part in (("few inversions (<=5)", low), ("many inversions (>9)", high)):
            if len(part) >= 4:
                print(f"    speculate, {name}, n={len(part)}: KL {mean_ci([r['speculate']['kl_last'] for r in part])[0]:.4f}; append: {mean_ci([r['append']['kl_last'] for r in part])[0]:.4f}")
    if merge:
        mean = lambda field: mean_ci([r[field] for r in merge])[0]
        print(f"\n=== B. merge of two branches / n={len(merge)} (merged context {mean('tokens'):.0f} tokens, {mean('stale_tokens'):.0f} stale after the merge)")
        for key in ("merge", *[f"repair_{e}{'_first' if f else ''}@{s}" for s in REPAIR_SHARES for e in ESTIMATES for f in READER_FIRST]):
            extra = f"   (+{mean_ci([r[key]['tokens'] for r in merge])[0]:.0f} tokens)" if "tokens" in merge[0][key] else ""
            print(f"    {key:12s} {line(merge, key)}{extra}")


def main() -> None:
    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    rows = hotpot_rows(count)
    stem, settings = f"e5_streams_{key}_{tag}", {"model": MODELS[key], "docs": DOCS, "repair_shares": list(REPAIR_SHARES), "estimates": list(ESTIMATES),
                     "reader_first": list(READER_FIRST)}
    saved = resume(stem, **settings)
    arrival, merge = saved.get("arrival", []), saved.get("merge", [])
    done = {r["id"] for r in arrival} & {r["id"] for r in merge}
    arrival, merge = [r for r in arrival if r["id"] in done], [r for r in merge if r["id"] in done]
    print(f"{len(done)} of {len(rows)} cases already saved in {stem}")
    for row in rows:
        if row["id"] in done:
            continue
        arrival.append(arrival_case(lm, row, seed=0))
        merge.append(merge_case(lm, row))
        if len(arrival) % 10 == 0:
            save(stem, {**settings, "arrival": arrival, "merge": merge})
    print("saved", save(stem, {**settings, "arrival": arrival, "merge": merge}))
    report(arrival, merge)


if __name__ == "__main__":
    main()
