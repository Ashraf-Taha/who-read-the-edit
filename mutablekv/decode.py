from __future__ import annotations

import torch

from mutablekv.lm import LM, Cells


def greedy(lm: LM, tokens: torch.Tensor, pos: torch.Tensor, cached: Cells | None, max_new: int, stop: set[int]) -> list[int]:
    """Greedy continuation of a sequence whose first len(cached) tokens are read from cache.

    Every step recomputes all tokens after the cache. That is a few dozen tokens,
    and it avoids copying the whole cache once per generated token.
    """
    held = 0 if cached is None else len(cached)
    produced: list[int] = []
    for _ in range(max_new + 1):
        step = lm.run(tokens, pos, torch.arange(tokens.numel()) >= held, cached)
        choice = int(lm.logits(step.hidden[-1:]).argmax(-1))
        if choice in stop or len(produced) == max_new:
            break
        produced.append(choice)
        tokens = torch.cat((tokens, torch.tensor([choice])))
        pos = torch.cat((pos, pos[-1:] + 1))
    return produced
