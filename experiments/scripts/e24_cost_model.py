"""E24: what would repair cost in a fused engine? A roofline estimate, not a measurement.

Each pass over m tokens is charged max(FLOPs / peak compute, bytes / peak bandwidth), the time an
ideal kernel at peak would need. The model counts:

- weights: every layer's projection and MLP weights in bfloat16, read once per pass (one pass per
  CHUNK tokens, as chunked prefill does); the LM head is not needed for any of these passes;
- matmul FLOPs: 2 per weight per token;
- attention: 4 * heads * head_dim FLOPs per layer per (query, key) pair, keys up to the query's
  own position; query rows are packed into tiles of TILE rows, and each tile reads the keys and
  values of every token up to the last position in the tile;
- KV writes: keys and values of every recomputed token, all layers, bfloat16.

Three passes are compared, all over a context of PREFIX tokens before the edit and n after it:
re-prefilling the suffix; the reader pass, NEW_INPUT tokens of new input over the stale cache;
and the repair of a share of the suffix, either spread evenly over it (as a token score picks
them) or as one block right after the edit. Real kernels run below peak and small passes pay
launch and scheduling costs, so the estimate flatters small passes: the measured ratios in a real
engine will be higher. The reader pass also needs a kernel that returns per-key attention sums,
which standard fused attention does not.

Usage: e24_cost_model.py [tag]
"""
from __future__ import annotations

import math
import sys

from transformers import AutoConfig

from common import MODELS, save

GPUS = {"a": (312e12, 2.0e12), "h": (989e12, 3.35e12)}
SUFFIXES = (1024, 4096, 16384)
BUDGETS = (0.01, 0.10)
PREFIX, NEW_INPUT, TILE, CHUNK, BYTES = 1024, 16, 128, 2048, 2


def shape(key: str) -> dict:
    c = AutoConfig.from_pretrained(MODELS[key])
    h, layers, heads, groups = c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads
    d = getattr(c, "head_dim", None) or h // heads
    weights = h * heads * d + 2 * h * groups * d + heads * d * h + 3 * h * c.intermediate_size
    return {"layers": layers, "heads": heads, "groups": groups, "head_dim": d, "weights_per_layer": weights,
            "kv_bytes_per_token": layers * 2 * groups * d * BYTES}


def pass_time(m: dict, positions: list[int], gpu: tuple[float, float]) -> float:
    """Ideal time to compute the tokens at `positions` (sorted), reading everything before them from cache."""
    flops_peak, bandwidth = gpu
    total = 0.0
    for start in range(0, len(positions), CHUNK):
        chunk = positions[start:start + CHUNK]
        flops = 2 * m["layers"] * m["weights_per_layer"] * len(chunk)
        flops += 4 * m["layers"] * m["heads"] * m["head_dim"] * sum(p + 1 for p in chunk)
        tiles = [chunk[i:i + TILE] for i in range(0, len(chunk), TILE)]
        kv_read = sum((tile[-1] + 1) * m["kv_bytes_per_token"] for tile in tiles)
        weights = m["layers"] * m["weights_per_layer"] * BYTES
        kv_write = len(chunk) * m["kv_bytes_per_token"]
        total += max(flops / flops_peak, (weights + kv_read + kv_write) / bandwidth)
    return total


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    rows = []
    for key in ("qwen1.5b", "qwen7b"):
        m = shape(key)
        for gpu_name, gpu in GPUS.items():
            for n in SUFFIXES:
                suffix = list(range(PREFIX, PREFIX + n))
                reprefill = pass_time(m, suffix, gpu)
                reader = pass_time(m, list(range(PREFIX + n, PREFIX + n + NEW_INPUT)), gpu)
                row = {"model": key, "gpu": gpu_name, "suffix": n, "reprefill": reprefill, "reader": reader}
                for budget in BUDGETS:
                    count = max(1, round(budget * n))
                    spread = sorted({PREFIX + math.floor(i * n / count) for i in range(count)})
                    row[f"scattered@{budget}"] = pass_time(m, spread, gpu)
                    row[f"block@{budget}"] = pass_time(m, suffix[:count], gpu)
                rows.append(row)
        print(key, m)
    save(f"e24_cost_model_{tag}", {"gpus": GPUS, "prefix": PREFIX, "new_input": NEW_INPUT, "tile": TILE, "chunk": CHUNK, "rows": rows})
    for r in rows:
        print(f"{r['model']:9s} {r['gpu']} n={r['suffix']:5d} re-prefill {1e3 * r['reprefill']:7.2f} ms | reader {1e3 * r['reader']:.2f} | "
              + " | ".join(f"{b:.0%}: scattered {1e3 * r[f'scattered@{b}']:.2f} block {1e3 * r[f'block@{b}']:.2f} "
                           f"ratio {(r['reader'] + r[f'scattered@{b}']) / r['reprefill']:.3f}" for b in BUDGETS))


if __name__ == "__main__":
    main()
