from __future__ import annotations

from typing import Any

import torch

from mutablekv.context import Context


class LateEvent(Exception):
    """An event arrived for a part of the context that a watermark already sealed."""


class Stream:
    """Event-time ordering over a Context.

    Events are applied when they arrive, in any order; each lands at the place its
    event time gives it. A watermark promises that nothing earlier will arrive.
    Everything before it is then brought to the exact state once and never changes.
    """

    def __init__(self, context: Context):
        self.context = context
        self.watermark: Any = None

    def _check(self, order) -> None:
        if self.watermark is not None and order < self.watermark:
            raise LateEvent(f"event time {order!r} is behind the watermark {self.watermark!r}")

    def offer(self, key, order, tokens: torch.Tensor) -> None:
        self._check(order)
        self.context.put(key, order, tokens)

    def retract(self, key) -> None:
        self._check(self.context.segments[self.context._index(key)].order)
        self.context.drop(key)

    def advance(self, watermark) -> int:
        """Move the watermark forward and make everything before it exact. Returns tokens recomputed."""
        if self.watermark is not None and watermark < self.watermark:
            raise ValueError("a watermark cannot move backwards")
        self.watermark = watermark
        sealed = sum(s.order < watermark for s in self.context.segments)
        spent = 0
        while (index := self.context.frontier()) < sealed:
            spent += self.context.sweep(budget=len(self.context.segments[index]))
        return spent

    def sealed(self) -> int:
        if self.watermark is None:
            return 0
        return sum(s.order < self.watermark for s in self.context.segments)
