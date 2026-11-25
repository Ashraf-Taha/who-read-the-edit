"""E14: what offset grouping and online exposure cost inside one attention call (micro-benchmarks).

Not a fused kernel: plain PyTorch on the current device, timed in isolation, so the numbers are
relative costs of the arithmetic and memory access patterns, not serving latency.

Part A, query-side relocation. One attention call of `m` new queries over `n` cached keys
(Qwen2.5-1.5B shapes: 12 query heads, 2 key/value heads, head dim 128) whose offsets take `D`
distinct values. Variants:
    plain      keys already at their current positions (what a rewrite buys), one attention
    rewrite    rotate every cached key to its current position, then one attention
    gather     per offset value, gather its keys, score them with the query rotated once for
               that offset, scatter the scores back (what our engine does)
    select     score every key with each of the D rotated queries and keep, per key, the score
               for its own offset; no gather, D times the score arithmetic
Offsets are laid out either as D contiguous groups or scattered over runs of `run` keys
(fragmentation; the number of runs is n / run).

Part B, exposure inside tiled attention. A FlashAttention-style pass over key tiles with an
online softmax, with and without one extra accumulator per segment that is rescaled like the
output; checks that the online per-segment mass equals the mass from materialised weights.

Usage: e14_kernels.py [tag] [device]
"""
from __future__ import annotations

import sys
import time

import torch

from common import best_device, save

H, G, DH = 12, 2, 128
BASE = 1e6


def inv_freq(device) -> torch.Tensor:
    return 1.0 / BASE ** (torch.arange(0, DH, 2, dtype=torch.float32, device=device) / DH)


def rotate(x: torch.Tensor, pos: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
    angle = pos.float()[:, None] * freq[None, :]
    angle = torch.cat((angle, angle), dim=-1)
    half = x.shape[-1] // 2
    turned = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * angle.cos() + turned * angle.sin()


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def clock(fn, device: str, repeats: int = 20) -> float:
    fn()
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


def layout(n: int, D: int, run: int | None, device) -> torch.Tensor:
    """Offset value per key: D values in D contiguous groups, or cycling over runs of `run` keys."""
    if D == 1:
        return torch.zeros(n, dtype=torch.long, device=device)
    if run is None:
        return (torch.arange(n, device=device) * D // n) * 37
    return ((torch.arange(n, device=device) // run) % D) * 37


def part_a(device: str) -> list[dict]:
    torch.manual_seed(0)
    freq, rows = inv_freq(device), []
    for n in (4096, 16384):
        for m in (1, 64):
            q = torch.randn(H, m, DH, device=device)
            k_enc = torch.randn(G, n, DH, device=device)
            v = torch.randn(G, n, DH, device=device)
            qpos = torch.arange(n, n + m, device=device)
            for D in (1, 2, 4, 8, 16, 32):
                for run in ((None,) if D == 1 else (None, 64, 8)):
                    offset = layout(n, D, run, device)
                    enc_pos = torch.arange(n, device=device) - offset
                    k_enc_rot = rotate(k_enc, enc_pos, freq)
                    k_now = rotate(k_enc, torch.arange(n, device=device), freq)
                    values = offset.unique().tolist()
                    groups = [(value, (offset == value).nonzero().squeeze(1)) for value in values]
                    q_at = {value: rotate(q, qpos - value, freq).view(G, H // G, m, DH) for value in values}
                    scale = DH ** -0.5

                    def attend(scores):
                        return torch.softmax(scores * scale, dim=-1) @ v[:, None]

                    def plain():
                        return attend(rotate(q, qpos, freq).view(G, H // G, m, DH) @ k_now[:, None].transpose(-1, -2))

                    def rewrite():
                        k = rotate(k_enc_rot, offset, freq)
                        return attend(rotate(q, qpos, freq).view(G, H // G, m, DH) @ k[:, None].transpose(-1, -2))

                    def gather():
                        scores = q.new_empty(G, H // G, m, n)
                        for value, columns in groups:
                            scores[..., columns] = q_at[value] @ k_enc_rot[:, None, columns].transpose(-1, -2)
                        return attend(scores)

                    def select():
                        scores = q.new_zeros(G, H // G, m, n)
                        for value, _ in groups:
                            part = q_at[value] @ k_enc_rot[:, None].transpose(-1, -2)
                            scores = torch.where((offset == value)[None, None, None, :], part, scores)
                        return attend(scores)

                    reference = plain()
                    errors = {name: (fn() - reference).abs().max().item() for name, fn in (("rewrite", rewrite), ("gather", gather), ("select", select))}
                    runs = int((offset[1:] != offset[:-1]).sum().item()) + 1
                    row = {"n": n, "m": m, "D": D, "runs": runs, "layout": "contiguous" if run is None else f"runs of {run}",
                           **{name: clock(fn, device) for name, fn in (("plain", plain), ("rewrite", rewrite), ("gather", gather), ("select", select))},
                           "max_error": max(errors.values())}
                    rows.append(row)
                    print({k: (round(v * 1e3, 3) if isinstance(v, float) and k not in ("max_error",) else v) for k, v in row.items()}, flush=True)
    return rows


def tiled(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, segment: torch.Tensor | None, n_segments: int, tile: int):
    """Online-softmax attention over key tiles; optionally accumulates the attention mass per segment."""
    Hq, m, d = q.shape
    scale = d ** -0.5
    running_max = q.new_full((Hq, m, 1), -float("inf"))
    running_sum = q.new_zeros(Hq, m, 1)
    out = q.new_zeros(Hq, m, d)
    mass = q.new_zeros(Hq, m, n_segments) if segment is not None else None
    for start in range(0, k.shape[1], tile):
        kt, vt = k[:, start:start + tile], v[:, start:start + tile]
        scores = (q @ kt.transpose(-1, -2)) * scale
        new_max = torch.maximum(running_max, scores.amax(-1, keepdim=True))
        rescale = (running_max - new_max).exp()
        p = (scores - new_max).exp()
        running_sum = running_sum * rescale + p.sum(-1, keepdim=True)
        out = out * rescale + p @ vt
        if mass is not None:
            mass = mass * rescale
            mass.index_add_(2, segment[start:start + tile], p)
        running_max = new_max
    out = out / running_sum
    return out, (mass / running_sum if mass is not None else None)


def part_b(device: str) -> list[dict]:
    torch.manual_seed(1)
    rows = []
    for n in (4096, 16384):
        for n_segments in (10, 50):
            for m in (64, 512):
                q = torch.randn(H, m, DH, device=device)
                k = torch.randn(H, n, DH, device=device)
                v = torch.randn(H, n, DH, device=device)
                segment = torch.sort(torch.randint(0, n_segments, (n,), device=device)).values
                _, online = tiled(q, k, v, segment, n_segments, 1024)
                weights = torch.softmax((q @ k.transpose(-1, -2)) * DH ** -0.5, dim=-1)
                exact = torch.zeros(H, m, n_segments, device=device).index_add_(2, segment, weights)
                row = {"n": n, "segments": n_segments, "m": m,
                       "without": clock(lambda: tiled(q, k, v, None, n_segments, 1024), device, 10),
                       "with": clock(lambda: tiled(q, k, v, segment, n_segments, 1024), device, 10),
                       "max_error": (online - exact).abs().max().item()}
                rows.append(row)
                print({key: (round(val * 1e3, 3) if key in ("without", "with") else val) for key, val in row.items()}, flush=True)
    return rows


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    device = sys.argv[2] if len(sys.argv) > 2 else best_device()
    payload = {"device": device, "shapes": {"H": H, "G": G, "head_dim": DH}}
    payload["relocation"] = part_a(device)
    payload["exposure"] = part_b(device)
    print("saved", save(f"e14_kernels_{tag}", payload))


if __name__ == "__main__":
    main()
