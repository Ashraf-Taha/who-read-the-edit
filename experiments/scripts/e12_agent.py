"""E12: agent traces, where the reader is not one cached question, and chained edits.

Each case is a synthetic tool-use trace built from one HotpotQA question: a system prompt, the
user's question, and three rounds of a templated thought, a search action and an observation
(the two supporting documents, each with two distractors, then three distractors), ending with a
thought that the agent has enough information. The new input asks for the final answer. Agents
clear old tool results to save room; the edit replaces one observation by a placeholder. The
user's question comes before every observation, so it is never stale: the stale readers are the
later thoughts and observations.

Part 1, single edit (`clear1`: the first observation; `clear2`: the second). Policies at 5%, 10%
and 20% of the stale tokens, as in E2c: boundary window, reader attention, recorded exposure,
rank fusion, the bridge score, the stale segment with the most reader attention per token then
exposure, and true deviation x reader attention. Teacher-forced KL over the first 16 reply tokens.

Part 2, chained edits through the runtime (`mutablekv.Context`): clear the first observation,
repair 10% for the answer turn, then clear the second observation before any sweep and repair
10% again; compare with no repair, with a sweep between the edits, and with a sweep at the end.
Repairs use the runtime's own records (exposure as debt, reader attention) with reader-first on
and off.

Usage: e12_agent.py <model-key> <cases> [tag] [device]
"""
from __future__ import annotations

import random
import sys

import torch

from common import MODELS, best_device, kl_rows, resume, save
from e2c_ablation import Reader, budget_count, compare, reply_of
from e2d_reader_first import first
from e9_scale import deviation
from mutablekv import LM, Cells, Context, heal

BUDGETS = (0.05, 0.10, 0.20)
POLICIES = ("boundary", "reader", "exposure", "rrf", "bridge", "segment_exposure", "oracle_reader")
SYSTEM = "You are a research assistant. You may call search[query] to look things up. Answer the user's question from the observations."
PLACEHOLDER = "Observation: [tool result cleared to save space]\n"


def trace(lm: LM, row, rng: random.Random, distractors: list[tuple[str, str]]) -> tuple[list[torch.Tensor], list[str]] | None:
    titles, sentences = row["context"]["title"], row["context"]["sentences"]
    gold = [t for t in dict.fromkeys(row["supporting_facts"]["title"]) if t in titles]
    if len(gold) != 2:
        return None
    text = {t: "".join(s).strip() for t, s in zip(titles, sentences)}
    others = [t for t in titles if t not in gold]
    if len(others) < 4:
        return None
    rng.shuffle(others)
    a, b = gold
    extra = rng.sample(distractors, 3)
    doc = lambda title, body: f"[{title}] {body}\n"
    observation = lambda docs: "Observation:\n" + "".join(doc(t, body) for t, body in docs)
    turn = lambda role, content: f"<|im_start|>{role}\n{content}<|im_end|>\n"
    pieces = [
        ("frame", turn("system", SYSTEM) + turn("user", row["question"])),
        ("thought1", turn("assistant", f"Thought: I should look up {a}.\nAction: search[{a}]")),
        ("observation1", turn("user", observation([(a, text[a]), (others[0], text[others[0]]), (others[1], text[others[1]])]))),
        ("thought2", turn("assistant", f"Thought: The search returned information about {a}. Next I will look up {b}.\nAction: search[{b}]")),
        ("observation2", turn("user", observation([(b, text[b]), (others[2], text[others[2]]), (others[3], text[others[3]])]))),
        ("thought3", turn("assistant", f"Thought: Now I know about {b}. One more source to be sure.\nAction: search[{extra[0][0]}]")),
        ("observation3", turn("user", observation(extra))),
        ("thought4", turn("assistant", "Thought: I have enough information to answer.")),
        ("ask", turn("user", "What is the final answer? Reply with the answer only.") + "<|im_start|>assistant\n"),
    ]
    return [lm.encode(content) for _, content in pieces], [name for name, _ in pieces]


def cleared(lm: LM) -> torch.Tensor:
    return lm.encode(f"<|im_start|>user\n{PLACEHOLDER}<|im_end|>\n")


def single_edit(lm: LM, segments: list[torch.Tensor], names: list[str], target: str, effect: bool = False) -> dict:
    """One edit: clear `target`, then score every policy in POLICIES at each budget against the re-prefill.

    With `effect`, also the edit's own effect: the re-prefill of the edited context against the cold
    pass over the unedited one, teacher-forced over the same reply.
    """
    edit = names.index(target)
    P, E, S, Q = torch.cat(segments[:edit]), segments[edit], torch.cat(segments[edit + 1:-1]), segments[-1]
    E2 = cleared(lm)
    nP, nE, nE2, nS = P.numel(), E.numel(), E2.numel(), S.numel()
    groups = torch.cat((torch.zeros(nP), torch.ones(nE), torch.full((nS,), 2))).long()
    original = lm.cold(torch.cat((P, E, S)), groups=groups)
    exposure = original.mass[:, nP + nE:, 1].mean(0)
    cP, with_S = original.cells.take(slice(0, nP)), original.cells.take(slice(nP + nE, None))
    unedited = original.cells if effect else None
    del original
    context = torch.cat((P, E2, S))
    start, n = nP + nE2, nP + nE2 + nS
    pos = torch.arange(n)
    cE2 = lm.run(torch.cat((P, E2)), torch.arange(start), torch.arange(start) >= nP, cP).cells
    truth = Cells.cat([cP, cE2, lm.run(context, pos, pos >= start, Cells.cat([cP, cE2])).cells])
    stale = Cells.cat([cP, cE2, with_S])
    reply, answer = reply_of(lm, context, Q, truth)
    reader = Reader(lm, context, Q, reply)
    truth_tf, _ = reader.logits(truth)
    stale_tf, mass = reader.logits(stale, groups=torch.arange(reader.tokens.numel()))
    reader_score = mass[:, : Q.numel(), start:n].mean((0, 1))
    if effect:
        unedited_tf, _ = Reader(lm, torch.cat((P, E, S)), Q, reply).logits(unedited)
        del unedited
    oracle = deviation(lm, with_S, truth.take(slice(start, None))).mean(0)
    lengths = [s.numel() for s in segments[edit + 1:-1]]
    starts = [sum(lengths[:j]) for j in range(len(lengths))]
    picked = max(range(len(lengths)), key=lambda j: reader_score[starts[j]:starts[j] + lengths[j]].mean().item())
    in_picked = (torch.arange(nS) >= starts[picked]) & (torch.arange(nS) < starts[picked] + lengths[picked])
    in_last = torch.arange(nS) >= starts[-1]
    rank = lambda x: x.argsort(descending=True).argsort().float()
    scores = {"boundary": -torch.arange(nS).float(), "tail": torch.arange(nS).float(), "reader": reader_score, "exposure": exposure,
              "rrf": 1 / (60 + rank(exposure)) + 1 / (60 + rank(reader_score)), "bridge": exposure * reader_score,
              "segment_exposure": first(in_picked, exposure), "latest_exposure": first(in_last, exposure), "oracle_reader": oracle * reader_score}
    out = {"edit": target, "nS": nS, "picked": names[edit + 1 + picked], "answer": answer,
           "segment_share": {names[edit + 1 + j]: reader_score[starts[j]:starts[j] + lengths[j]].sum().item() / reader_score.sum().item() for j in range(len(lengths))},
           "results": {"stale": compare(truth_tf, stale_tf, reply)}}
    if effect:
        out["results"]["edit_effect"] = compare(truth_tf, unedited_tf, reply)
    for name in POLICIES:
        out["results"][name] = {}
        for budget in BUDGETS:
            chosen = torch.zeros(n, dtype=torch.bool)
            chosen[start + scores[name].topk(budget_count(budget, nS)).indices] = True
            out["results"][name][str(budget)] = compare(truth_tf, reader.logits(heal(lm, context, pos, stale, chosen))[0], reply)
    return out


def chained(lm: LM, segments: list[torch.Tensor], names: list[str]) -> dict:
    ask = segments[-1]

    def build() -> Context:
        context = Context(lm)
        for order, (name, tokens) in enumerate(zip(names[:-1], segments[:-1])):
            context.put(name, order, tokens)
        return context

    def distance(context: Context) -> float:
        truth = Context(lm, list(context.segments))
        truth.sweep()
        target = truth.read(ask)[0][-1]
        return kl_rows(target[None], context.read(ask)[0][-1][None])[0].item()

    out = {}
    for label, plan in (("positional", (None, None)), ("repair_repair", ("repair", "repair")), ("repair_repair_reader_first", ("first", "first")),
                        ("sweep_repair", ("sweep", "repair")), ("sweep_repair_reader_first", ("sweep", "first"))):
        context = build()
        steps = []
        for target, action in zip(("observation1", "observation2"), plan):
            context.swap(target, cleared(lm))
            if action == "repair":
                context.repair(ask, 0.10, estimate="flat")
            elif action == "first":
                context.repair(ask, 0.10, estimate="flat", reader_first=True)
            elif action == "sweep":
                context.sweep()
            steps.append({"kl": distance(context), "frontier": context.frontier(), "stale_tokens": context.stale_tokens()})
        out[label] = steps
    return out


def main() -> None:
    from datasets import load_dataset

    key, count = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    lm = LM(MODELS[key], device, torch.float32)
    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    order = list(range(len(data)))
    random.Random(12_000).shuffle(order)
    pool = [(t, "".join(s).strip()) for i in order[-400:] for t, s in zip(data[i]["context"]["title"], data[i]["context"]["sentences"])]
    stem = f"e12_agent_{key}_{tag}"
    settings = {"model": MODELS[key], "budgets": list(BUDGETS), "policies": list(POLICIES)}
    rows = resume(stem, **settings).get("rows", [])
    done = {r["id"] for r in rows}
    for row_index in order[:-400]:
        if len(rows) >= count:
            break
        row = data[row_index]
        if row["id"] in done or row["answer"].lower() in ("yes", "no"):
            continue
        built = trace(lm, row, random.Random(row["id"]), pool)
        if built is None:
            continue
        segments, names = built
        if sum(s.numel() for s in segments) > 2400:
            continue
        rows.append({"id": row["id"], "answer": row["answer"], "tokens": sum(s.numel() for s in segments),
                     "clear1": single_edit(lm, segments, names, "observation1"), "clear2": single_edit(lm, segments, names, "observation2"),
                     "chained": chained(lm, segments, names)})
        save(stem, {**settings, "rows": rows})
        print(f"{len(rows)} traces", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
