from __future__ import annotations

import collections
import datetime
import json
import math
import pathlib
import platform
import re
import statistics
import string

import torch
import transformers

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smol1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
}
TITLE = re.compile(r"^ = [^=].*[^=] = \n?$")


def best_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def wikitext_articles(min_chars: int = 6000) -> list[str]:
    from datasets import load_dataset

    lines = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"]
    articles, current = [], []
    for line in lines:
        if TITLE.match(line) and current:
            articles.append("".join(current))
            current = []
        current.append(line)
    articles.append("".join(current))
    return [a.strip() for a in articles if len(a) >= min_chars]


def kl_rows(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """KL(reference || other) in nats for each row of two logit matrices."""
    p = torch.log_softmax(reference.cpu().double(), -1)
    q = torch.log_softmax(other.cpu().double(), -1)
    return (p.exp() * (p - q)).sum(-1).clamp_min(0.0)


def normalise(text: str) -> str:
    """SQuAD-style answer normalisation."""
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(word for word in text.split() if word not in ("a", "an", "the"))


def answer_f1(prediction: str, gold: str) -> float:
    predicted, wanted = normalise(prediction).split(), normalise(gold).split()
    common = sum((collections.Counter(predicted) & collections.Counter(wanted)).values())
    if not predicted or not wanted or not common:
        return float(predicted == wanted)
    precision, recall = common / len(predicted), common / len(wanted)
    return 2 * precision * recall / (precision + recall)


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of a normal-approximation 95% interval."""
    if len(values) < 2:
        return (values[0] if values else math.nan), math.nan
    return statistics.fmean(values), 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def save(name: str, payload: dict) -> pathlib.Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "written": datetime.datetime.now().isoformat(timespec="seconds"),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "machine": platform.platform(),
            "processor": platform.processor(),
        },
        **payload,
    }
    path = RESULTS / f"{name}.json"
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(payload, indent=1))
    partial.replace(path)
    return path


def resume(name: str, **settings) -> dict:
    """The saved payload of an earlier run with the same name, or {} if there is none.

    Refuses to continue a run that was written with different settings, so that one
    file never mixes cases computed under two configurations.
    """
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return {}
    saved = json.loads(path.read_text())
    for field, value in json.loads(json.dumps(settings)).items():
        if saved.get(field) != value:
            raise SystemExit(f"{path.name} was written with {field}={saved.get(field)!r}, not {value!r}; use another tag")
    return saved


def release() -> None:
    """Hand cached GPU memory back to the system between cases, so other programs keep room on a shared 24 GB machine."""
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
