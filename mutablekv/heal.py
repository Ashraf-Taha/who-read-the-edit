from __future__ import annotations

import torch

from mutablekv.lm import LM, Cells


def heal(lm: LM, tokens: torch.Tensor, pos: torch.Tensor, cells: Cells, chosen: torch.Tensor) -> Cells:
    """Recompute the tokens selected by the boolean mask `chosen`, reading every other cell as cached.

    The result is exact for the chosen tokens only if every cell before them is
    exact; otherwise it is an approximation that removes their direct reads of
    whatever is no longer in the sequence.
    """
    if not chosen.any():
        return cells
    fresh = lm.run(tokens, pos, chosen, cells.take(~chosen) if (~chosen).any() else None).cells
    return cells.replaced(chosen, fresh)


def sweep(lm: LM, tokens: torch.Tensor, pos: torch.Tensor, cells: Cells, start: int, stop: int | None = None, chunk: int = 256) -> Cells:
    """Recompute [start, stop) left to right in chunks. If cells before `start` are exact, so is the result up to `stop`."""
    index = torch.arange(tokens.numel())
    stop = tokens.numel() if stop is None else stop
    for low in range(start, stop, chunk):
        cells = heal(lm, tokens, pos, cells, (index >= low) & (index < min(low + chunk, stop)))
    return cells
