"""E0: is the engine exact, and does rewriting cached keys in place drift?

Part A compares the engine with the library and checks the relocation algebra.
Part B stores keys in a 16-bit format and applies chains of in-place rotations,
the way a key-rewriting splice does on every upstream edit, against leaving the
stored keys alone and rotating the query.

Usage: e0_exactness.py [exact] [drift]
"""
from __future__ import annotations

import sys

import torch

from common import MODELS, kl_rows, mean_ci, save, wikitext_articles
from mutablekv import LM, Cells

CONTEXT, TAIL = 480, 32
CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
STORAGE = {"bf16": torch.bfloat16, "fp16": torch.float16}
PASSAGES, SEEDS, MAX_STEP = 4, 2, 64


def gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.cpu().double() - b.cpu().double()).abs().max().item()


def exactness(repo: str, device: str, dtype: torch.dtype, text: str) -> dict:
    lm = LM(repo, device, dtype)
    tokens = lm.encode(text)[: CONTEXT + TAIL]
    n, a, b = tokens.numel(), 160, CONTEXT
    index = pos = torch.arange(n)
    whole = lm.cold(tokens)
    reference = lm.logits(whole.hidden)
    row = {"model": repo, "device": device, "dtype": str(dtype).removeprefix("torch."), "tokens": n}

    if dtype == torch.float32:
        with torch.inference_mode():
            library = lm.hf(tokens[None].to(device)).logits[0]
        row["vs_library_logit_gap"] = gap(reference, library)
        row["vs_library_argmax_agree"] = (reference.argmax(-1) == library.argmax(-1)).float().mean().item()

    for shift in (1000, 20000):
        row[f"uniform_shift_{shift}_logit_gap"] = gap(lm.logits(lm.cold(tokens, pos + shift).hidden), reference)

    first = lm.cold(tokens[:a]).cells
    second = lm.run(tokens[:b], pos[:b] + 1000, fresh=index[:b] >= a, cached=first).cells
    cached = Cells.cat([first, second])
    for mode in ("query", "rewrite"):
        tail = lm.run(tokens, pos, fresh=index >= b, cached=cached, relocate=mode)
        row[f"relocated_{mode}_logit_gap"] = gap(lm.logits(tail.hidden), reference[b:])

    scrambled = torch.cat((pos[:a], pos[a:320] + 37, pos[320:b] + 42, pos[b:] + 64))
    by_query = lm.run(tokens, scrambled, index >= b, whole.cells.take(index < b), relocate="query")
    by_rewrite = lm.run(tokens, scrambled, index >= b, whole.cells.take(index < b), relocate="rewrite")
    row["query_vs_rewrite_logit_gap"] = gap(lm.logits(by_query.hidden), lm.logits(by_rewrite.hidden))
    return row


def stored(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype).to(torch.float32)


def drift(repo: str, texts: list[str]) -> list[dict]:
    lm = LM(repo, "cpu", torch.float32)
    rows = []
    for passage, text in enumerate(texts):
        tokens = lm.encode(text)[: CONTEXT + TAIL]
        n = tokens.numel()
        index = pos = torch.arange(n)
        fresh = index >= CONTEXT
        exact = lm.cold(tokens[:CONTEXT]).cells
        full_precision = lm.logits(lm.run(tokens, pos, fresh, exact).hidden)

        for storage, dtype in STORAGE.items():
            base = Cells(exact.tokens, exact.enc_pos, stored(exact.K, dtype), stored(exact.V, dtype))
            reference = lm.logits(lm.run(tokens, pos, fresh, base).hidden)
            floor = kl_rows(full_precision, reference).mean().item()
            norm = base.K.double().norm().item()

            for seed in range(SEEDS):
                generator = torch.Generator().manual_seed(1000 * passage + seed)
                keys, total = base.K.clone(), 0
                for edit in range(1, CHECKPOINTS[-1] + 1):
                    step = int(torch.randint(1, MAX_STEP + 1, (1,), generator=generator))
                    if total > 4 * MAX_STEP and torch.rand((), generator=generator) < 0.5:
                        step = -step
                    total += step
                    keys = stored(lm.rotate(keys, torch.full((CONTEXT,), step)), dtype)
                    if edit not in CHECKPOINTS:
                        continue

                    ideal = lm.rotate(base.K.double(), torch.full((CONTEXT,), total))
                    moved = pos + total
                    rewritten = Cells(base.tokens, moved[:CONTEXT], keys, base.V)
                    arms = {
                        "rewrite_in_place": lm.run(tokens, moved, fresh, rewritten),
                        "rotate_query": lm.run(tokens, moved, fresh, base),
                    }
                    for arm, result in arms.items():
                        logits = lm.logits(result.hidden)
                        divergence = kl_rows(reference, logits)
                        rows.append({
                            "model": repo, "storage": storage, "passage": passage, "seed": seed, "edits": edit,
                            "total_shift": total, "arm": arm,
                            "key_rel_err": (keys.double() - ideal).norm().item() / norm if arm == "rewrite_in_place" else 0.0,
                            "kl_mean": divergence.mean().item(), "kl_max": divergence.max().item(),
                            "flip_rate": (logits.argmax(-1) != reference.argmax(-1)).float().mean().item(),
                            "storage_floor_kl": floor,
                        })
    return rows


def main() -> None:
    parts = set(sys.argv[1:]) or {"exact", "drift"}
    articles = wikitext_articles()
    if "exact" in parts:
        configs = [(MODELS["qwen0.5b"], "cpu", torch.float64)]
        configs += [(repo, "cpu", torch.float32) for repo in MODELS.values()]
        if torch.backends.mps.is_available():
            configs += [(repo, "mps", torch.float32) for repo in MODELS.values()]
        rows = []
        for repo, device, dtype in configs:
            rows.append(exactness(repo, device, dtype, articles[0]))
            if device == "mps":
                torch.mps.empty_cache()
        print("saved", save("e0_exactness", {"rows": rows}))
        for row in rows:
            print({k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in row.items()})

    if "drift" in parts:
        rows = []
        for repo in MODELS.values():
            rows += drift(repo, articles[1 : 1 + PASSAGES])
            save("e0_drift", {"context": CONTEXT, "tail": TAIL, "max_step": MAX_STEP, "rows": rows})
        print("saved", save("e0_drift", {"context": CONTEXT, "tail": TAIL, "max_step": MAX_STEP, "rows": rows}))
        for repo in MODELS.values():
            for storage in STORAGE:
                print(f"\n{repo} | keys stored as {storage}")
                print(f"{'edits':>6} {'key rel err':>12} {'KL rewrite':>22} {'flips rewrite':>14} {'KL rotate-query':>22} {'flips query':>12}")
                for edits in CHECKPOINTS:
                    pick = lambda arm, field: [r[field] for r in rows if (r["model"], r["storage"], r["edits"], r["arm"]) == (repo, storage, edits, arm)]
                    kr, kr_ci = mean_ci(pick("rewrite_in_place", "kl_mean"))
                    kq, kq_ci = mean_ci(pick("rotate_query", "kl_mean"))
                    err, _ = mean_ci(pick("rewrite_in_place", "key_rel_err"))
                    fr, _ = mean_ci(pick("rewrite_in_place", "flip_rate"))
                    fq, _ = mean_ci(pick("rotate_query", "flip_rate"))
                    print(f"{edits:>6} {err:>12.2e} {kr:>12.2e} ±{kr_ci:>8.1e} {fr:>14.3f} {kq:>12.2e} ±{kq_ci:>8.1e} {fq:>12.3f}")


if __name__ == "__main__":
    main()
