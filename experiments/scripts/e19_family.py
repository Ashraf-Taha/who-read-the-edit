"""E19: the main experiment on a second 7B family (Mistral-7B-Instruct-v0.3, bfloat16).

The cases are exactly the main experiment's questions, edited documents and layouts: they are
selected with the Qwen tokenizer, as in E2c and E9, and then encoded with this model's tokenizer,
so their token counts differ. Policies and metrics as in E9 (10% of the suffix, teacher-forced KL
over the first 16 reply tokens, a bfloat16 floor, greedy answers for two policies).

Usage: e19_family.py <model-key> <supporting-per-cell> [tag] [dtype] [device]
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoTokenizer

from common import MODELS, best_device, resume, save, release
from e2c_ablation import LEADS, build_cases
from e9_scale import BUDGET, POLICIES, analyse, reference_check
from mutablekv import LM
from workloads import READER_PLACES, reader_cases


class TokenizerOnly:
    """What the case builders need from a model: its tokenizer and `encode`."""

    def __init__(self, repo: str):
        self.tok = AutoTokenizer.from_pretrained(repo)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tok(text, add_special_tokens=False)["input_ids"], dtype=torch.long)


def matched_cases(lm: LM, supporting: int) -> list:
    """The main experiment's cases, rebuilt with `lm`'s tokenizer."""
    reference = build_cases(TokenizerOnly(MODELS["qwen1.5b"]), supporting, 0)
    wanted = {case.name for case in reference}
    cases = []
    for lead in LEADS:
        ids = {name.split("/")[1] for name in wanted if name.endswith(f"/lead{lead}")}
        for place in READER_PLACES:
            for case in reader_cases(lm, len(ids), "gold", place, seed=1000 + lead, lead=lead, max_tokens=10**6, only_ids=ids):
                case.name += f"/lead{lead}"
                if case.name in wanted:
                    cases.append(case)
    missing = wanted - {case.name for case in cases}
    if missing:
        raise SystemExit(f"{len(missing)} cases could not be rebuilt, e.g. {sorted(missing)[:3]}")
    return cases


def main() -> None:
    key, supporting = sys.argv[1], int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "main"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[sys.argv[4] if len(sys.argv) > 4 else "bf16"]
    device = sys.argv[5] if len(sys.argv) > 5 else best_device()
    lm = LM(MODELS[key], device, dtype)
    cases = matched_cases(lm, supporting)
    stem = f"e19_family_{key}_{tag}"
    settings = {"model": MODELS[key], "dtype": str(dtype), "budget": BUDGET, "policies": list(POLICIES)}
    saved = resume(stem, **settings)
    rows = saved.get("rows", [])
    check = saved.get("reference_check") or reference_check(lm, cases[0])
    print("engine vs reference implementation:", check, flush=True)
    done = {r["name"] for r in rows}
    print(f"{len(done)} of {len(cases)} cases already saved in {stem}", flush=True)
    for case in cases:
        if case.name in done:
            continue
        rows.append(analyse(lm, case))
        release()
        if len(rows) % 5 == 0:
            save(stem, {**settings, "reference_check": check, "rows": rows})
            print(f"{len(rows)} cases", flush=True)
    print("saved", save(stem, {**settings, "reference_check": check, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
