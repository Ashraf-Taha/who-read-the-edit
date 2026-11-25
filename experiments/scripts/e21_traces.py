"""E21: real agent transcripts. An old tool result is cleared while the newest one arrives.

E12's traces are templated. These are AgentInstruct trajectories [Zeng et al., 2023, AgentTuning]:
actions written by a model, interleaved with the outputs of real environments (a Linux shell, a
MySQL database, a knowledge-base API, the WebShop site and the ALFWorld simulator). A case cuts a
trajectory at one of its tool results: everything before that result is cached, and the result
itself, followed by the opening of the assistant's next turn, is the new input. The edit clears the
task's first or second tool result to a placeholder, as context editing does to save room. The
single-edit protocol, policies and budgets are E12's (`single_edit`), unchanged.

Selection, fixed before the run: per environment, trajectories in a seeded shuffled order that
have at least MIN_RESULTS tool results after the task statement; the cut is the latest tool result
for which the whole context, new input included, stays within MAX_TOKENS tokens of Qwen2.5's
tokenizer. Selection and cut use that tokenizer for every model, so all models see the same cases.
Only five database trajectories have four tool results, so that environment contributes five.

A tag starting with "extra" runs "latest turn first, then exposure" (no reader pass needed) and the
bridge score, and records the edit's own effect (the re-prefill of the edited context against the
cold pass over the unedited one).

A tag starting with "tail" runs only the tail window (the tokens right before the new input) and
the bridge score, the latter to check that the run reproduces the main one. The tail window was
added after a first look at the main run suggested that reader-first repair's advantage on these
transcripts might be recency.

Usage: e21_traces.py <model-key> <per-environment> [tag] [device]
"""
from __future__ import annotations

import random
import sys

import torch

from common import MODELS, best_device, release, resume, save
import e12_agent
from e12_agent import single_edit
from e19_family import TokenizerOnly
from mutablekv import LM

REVISION = "e252cf78ced8a0ea5f62cfd591784cdbbddbac8a"
FILES = {"os": "os-00000-of-00001-971539c34fcc7500", "db": "db-00000-of-00001-916a87c4725da8c0",
         "kg": "kg-00000-of-00001-9e159f6d0557d229", "webshop": "webshop-00000-of-00001-9f2ae60445e11b4e",
         "alfworld": "alfworld-00000-of-00001-302ad687bb3817a4"}
SYSTEM = "You are a helpful assistant."
MAX_TOKENS = 3000
MIN_RESULTS = 4
ROLE = {"human": "user", "gpt": "assistant"}


def load_split(name: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("THUDM/AgentInstruct", f"data/{FILES[name]}.parquet", repo_type="dataset", revision=REVISION)
    return pq.read_table(path).to_pylist()


def task_start(env: str, turns: list[dict]) -> int | None:
    """Index of the user turn that states the task, after the instructions and any worked example."""
    if env == "os":
        marks = [i for i, t in enumerate(turns) if t["from"] == "human" and t["value"].startswith("Now, I will start a new problem")]
        return marks[-1] if marks else None
    return 2 if len(turns) > 2 and turns[2]["from"] == "human" else None


def wrap(role: str, text: str) -> str:
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def trace(lm, env: str, turns: list[dict], new_result: int | None = None) -> tuple[list[torch.Tensor], list[str], int] | None:
    """Segments and names of a case. With `new_result` (1-based) the cut is that tool result; without it, the latest that fits."""
    start = task_start(env, turns)
    if start is None:
        return None
    results = [i for i in range(start + 1, len(turns)) if turns[i]["from"] == "human" and turns[i - 1]["from"] == "gpt"]
    if len(results) < MIN_RESULTS:
        return None
    frame = wrap("system", SYSTEM) + "".join(wrap(ROLE[t["from"]], t["value"]) for t in turns[:start + 1])
    for k in ([new_result - 1] if new_result else range(len(results) - 1, MIN_RESULTS - 2, -1)):
        cut = results[k]
        segments, names = [lm.encode(frame)], ["frame"]
        for i in range(start + 1, cut):
            segments.append(lm.encode(wrap(ROLE[turns[i]["from"]], turns[i]["value"])))
            names.append(f"observation{results.index(i) + 1}" if i in results else f"turn{i}")
        segments.append(lm.encode(wrap("user", turns[cut]["value"]) + "<|im_start|>assistant\n"))
        names.append("new_result")
        if new_result or sum(s.numel() for s in segments) <= MAX_TOKENS:
            return segments, names, k + 1
    return None


def main() -> None:
    key, per_env = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    device = sys.argv[4] if len(sys.argv) > 4 else best_device()
    if tag.startswith("tail"):
        e12_agent.POLICIES = ("tail", "bridge")
    effect = tag.startswith("extra")
    if effect:
        e12_agent.POLICIES = ("latest_exposure", "bridge")
    lm = LM(MODELS[key], device, torch.float32)
    reference = TokenizerOnly(MODELS["qwen1.5b"])
    stem = f"e21_traces_{key}_{tag}"
    settings = {"model": MODELS[key], "dataset": f"THUDM/AgentInstruct@{REVISION}", "max_tokens": MAX_TOKENS, "min_results": MIN_RESULTS}
    if tag.startswith(("tail", "extra")):
        settings["policies"] = list(e12_agent.POLICIES)
    if effect:
        settings["effect"] = True
    rows = resume(stem, **settings).get("rows", [])
    done = {r["id"] for r in rows}
    for env in FILES:
        data = load_split(env)
        order = list(range(len(data)))
        random.Random(21_000).shuffle(order)
        for index in order:
            if sum(r["env"] == env for r in rows) >= per_env:
                break
            name = f"{env}/{data[index]['id']}"
            if name in done:
                continue
            chosen = trace(reference, env, data[index]["conversations"])
            if chosen is None:
                continue
            segments, names, new_result = trace(lm, env, data[index]["conversations"], chosen[2])
            rows.append({"id": name, "env": env, "tokens": sum(s.numel() for s in segments), "new_result": new_result,
                         "cleared_tokens": {t: segments[names.index(t)].numel() for t in ("observation1", "observation2")},
                         "clear1": single_edit(lm, segments, names, "observation1", effect=effect),
                         "clear2": single_edit(lm, segments, names, "observation2", effect=effect)})
            save(stem, {**settings, "rows": rows})
            release()
            print(f"{len(rows)} traces ({env})", flush=True)
    print("saved", save(stem, {**settings, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
