"""Segmented contexts with one segment marked for editing.

A case is a list of token segments. `edit` is the segment that gets deleted (or,
read the other way round, arrives late and is inserted). Everything from
`fresh_from` on is new input that arrives after the edit and is always computed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from common import wikitext_articles

SYSTEM = "You answer questions using the documents provided. Reply with the answer only, as briefly as possible."
MARKER = "[[CONTENT]]"


@dataclass
class Case:
    name: str
    segments: list[torch.Tensor]
    edit: int
    fresh_from: int
    meta: dict = field(default_factory=dict)

    def part(self, start: int, stop: int) -> torch.Tensor:
        chosen = self.segments[start:stop]
        return torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long)


def chat_frame(lm) -> tuple[str, str]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": MARKER}]
    head, tail = lm.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).split(MARKER)
    return head, tail


def hotpot_cases(lm, count: int, kind: str, question_is_fresh: bool = True, seed: int = 0, max_tokens: int = 2200):
    """Multi-document QA prompts. `kind` picks which document is edited: 'gold' or 'distractor'."""
    from datasets import load_dataset

    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    head, tail = chat_frame(lm)
    cases = []
    for row_index in order:
        if len(cases) == count:
            break
        row = data[row_index]
        titles, sentences = row["context"]["title"], row["context"]["sentences"]
        gold = set(row["supporting_facts"]["title"])
        inner = [i for i in range(1, len(titles) - 1) if (titles[i] in gold) == (kind == "gold")]
        if len(titles) < 6 or not inner:
            continue
        edit_doc = random.Random(f"{seed}/{row['id']}").choice(inner)
        segments = [lm.encode(head + "Documents:\n\n")]
        segments += [lm.encode(f"Title: {t}\n{''.join(s).strip()}\n\n") for t, s in zip(titles, sentences)]
        segments += [lm.encode(f"Question: {row['question']}"), lm.encode(tail)]
        if sum(s.numel() for s in segments) > max_tokens or segments[1 + edit_doc].numel() < 16:
            continue
        cases.append(Case(
            name=f"hotpot/{row['id']}/{kind}/doc{edit_doc}",
            segments=segments,
            edit=1 + edit_doc,
            fresh_from=len(segments) - (2 if question_is_fresh else 1),
            meta={"workload": "hotpot", "kind": kind, "question_is_fresh": question_is_fresh, "answer": row["answer"],
                  "question": row["question"], "docs": len(titles), "edit_doc": edit_doc,
                  "gold_after_edit": sum(t in gold for t in titles[edit_doc + 1:])},
        ))
    return cases


def wikitext_cases(lm, count: int, seed: int = 0, head_tokens: int = 400, suffix_tokens: int = 600, probe: int = 32):
    """One coherent article; a middle paragraph is edited; the probe is the text that follows."""
    articles = wikitext_articles()
    random.Random(seed).shuffle(articles)
    cases = []
    for number, article in enumerate(articles):
        if len(cases) == count:
            break
        paragraphs = [lm.encode(line) for line in article.splitlines(keepends=True) if len(line) > 200 and " = " not in line[:4]]
        segments, used, i = [], 0, 0
        while i < len(paragraphs) and used < head_tokens:
            segments.append(paragraphs[i])
            used += paragraphs[i].numel()
            i += 1
        if i >= len(paragraphs) or not 40 <= paragraphs[i].numel() <= 300:
            continue
        edit = len(segments)
        segments.append(paragraphs[i])
        i, used = i + 1, 0
        while i < len(paragraphs) and used < suffix_tokens:
            segments.append(paragraphs[i])
            used += paragraphs[i].numel()
            i += 1
        if used < suffix_tokens // 2 or i >= len(paragraphs) or paragraphs[i].numel() < probe:
            continue
        segments.append(paragraphs[i][:probe])
        cases.append(Case(
            name=f"wikitext/{seed}/{number}",
            segments=segments,
            edit=edit,
            fresh_from=len(segments) - 1,
            meta={"workload": "wikitext", "kind": "paragraph"},
        ))
    return cases


READER_PLACES = ("adjacent", "middle", "end")


def reader_cases(lm, count: int, kind: str, place: str, seed: int = 0, max_tokens: int = 2200, lead: int = 2, only_ids: set | None = None):
    """Multi-document QA where the question is already cached and sits at a chosen place after the edited document.

    Documents are reordered so the edited one comes after `lead` others. The question
    then goes right after it ('adjacent'), halfway through the remaining documents
    ('middle'), or after all of them ('end'). Only the assistant turn opener is new input.
    `only_ids` restricts the cases to the given HotpotQA ids, so that a model with another
    tokenizer can be given exactly the questions selected for an earlier one.
    """
    from datasets import load_dataset

    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    head, tail = chat_frame(lm)
    cases = []
    for row_index in order:
        if len(cases) == count:
            break
        row = data[row_index]
        if only_ids is not None and row["id"] not in only_ids:
            continue
        titles, sentences = row["context"]["title"], row["context"]["sentences"]
        gold = set(row["supporting_facts"]["title"])
        candidates = [i for i in range(len(titles)) if (titles[i] in gold) == (kind == "gold")]
        if len(titles) < 8 or not candidates:
            continue
        edit_doc = random.Random(f"{seed}/{row['id']}").choice(candidates)
        others = [i for i in range(len(titles)) if i != edit_doc]
        text = lambda i: f"Title: {titles[i]}\n{''.join(sentences[i]).strip()}\n\n"
        document = lambda i: lm.encode(text(i))
        after = [document(i) for i in others[lead:]]
        slot = {"adjacent": 0, "middle": len(after) // 2, "end": len(after)}[place]
        question = lm.encode(f"Question: {row['question']}\n\n")
        segments = [lm.encode(head + "Documents:\n\n")] + [document(i) for i in others[:lead]] + [document(edit_doc)]
        segments += after[:slot] + [question] + after[slot:] + [lm.encode(tail)]
        order_of_titles = [titles[i] for i in others[:lead]] + [titles[edit_doc]] + [titles[i] for i in others[lead:]]
        segment_titles = [None] + order_of_titles[: lead + 1 + slot] + [None] + order_of_titles[lead + 1 + slot:] + [None]
        if sum(s.numel() for s in segments) > max_tokens or segments[1 + lead].numel() < 16:
            continue
        cases.append(Case(
            name=f"reader/{row['id']}/{kind}/{place}",
            segments=segments,
            edit=1 + lead,
            fresh_from=len(segments) - 1,
            meta={"workload": "reader", "kind": kind, "place": place, "answer": row["answer"], "question": row["question"],
                  "question_segment": 2 + lead + slot, "lead": lead, "segment_titles": segment_titles,
                  "gold_titles": sorted(gold), "edit_text": text(edit_doc)},
        ))
    return cases
