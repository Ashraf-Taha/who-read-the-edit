from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, replace
from typing import Any

import torch

from mutablekv.heal import heal
from mutablekv.lm import LM, Cells
from mutablekv.sketch import spread, would_be_logits

_patches = itertools.count(1)


def content_version(tokens: torch.Tensor, saw: tuple) -> str:
    """Identity of an encoding: its tokens and exactly what it read, at which relative offsets.

    Cells are a deterministic function of these, so equal versions mean equal cells.
    """
    digest = hashlib.blake2b(digest_size=12)
    digest.update(tokens.cpu().numpy().tobytes())
    digest.update(repr(saw).encode())
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class Segment:
    """One encoding of one keyed span of the context. Never modified after creation.

    version: content address of the encoding (see `content_version`), or a unique
        id for cells produced by a targeted repair, which are approximate.
    saw: for every segment that preceded it when it was encoded, that segment's
        version and its start relative to this one.
    lineage: attention mass each token put on each of those segments (layer mean).
    debt: attention mass each token put on content that has since been removed.
    missing: content has since been inserted before it that it never read.
    sketch_q, sketch_lse: the tokens' queries and softmax normalisers at a few layers,
        one per key/value group, kept from encoding so that the attention they would
        have paid to content inserted later can be estimated (see `Context.missing_mass`).
    sketch_start: the position of the first token when the sketch was taken.
    unsettled: 0 for tokens recomputed by a repair since the segment was encoded.
    """

    key: Any
    order: Any
    cells: Cells
    version: str
    saw: tuple[tuple[str, int], ...]
    lineage: torch.Tensor
    debt: torch.Tensor
    missing: bool = False
    patched: bool = False
    sketch_q: torch.Tensor | None = None
    sketch_lse: torch.Tensor | None = None
    sketch_start: int = 0
    unsettled: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.cells)


class Context:
    """A mutable, ordered set of segments over immutable cached cells.

    Mutations change which segments are in the sequence and where; they never
    write to a cell. Cells are only ever created: by encoding on arrival, by a
    targeted repair, or by the sweep that restores exactness.
    """

    def __init__(self, lm: LM, segments: list[Segment] | None = None):
        self.lm = lm
        self.segments: list[Segment] = list(segments or [])
        self.sketch_layers = spread(lm.L)

    def __len__(self) -> int:
        return sum(len(s) for s in self.segments)

    def keys(self) -> list:
        return [s.key for s in self.segments]

    def starts(self) -> list[int]:
        return list(itertools.accumulate((len(s) for s in self.segments), initial=0))[:-1]

    def tokens(self) -> torch.Tensor:
        return torch.cat([s.cells.tokens for s in self.segments]) if self.segments else torch.empty(0, dtype=torch.long)

    def cells(self) -> Cells | None:
        return Cells.cat([s.cells for s in self.segments]) if self.segments else None

    def _signature(self, index: int) -> tuple[tuple[str, int], ...]:
        starts = self.starts()
        return tuple((self.segments[j].version, starts[j] - starts[index]) for j in range(index))

    def frontier(self) -> int:
        """Number of leading segments whose cells equal a cold pass over the current sequence."""
        for index, segment in enumerate(self.segments):
            if segment.patched or segment.saw != self._signature(index):
                return index
        return len(self.segments)

    def stale_tokens(self) -> int:
        return sum(len(s) for s in self.segments[self.frontier():])

    def _index(self, key) -> int:
        for index, segment in enumerate(self.segments):
            if segment.key == key:
                return index
        raise KeyError(key)

    def _encoded_after_all(self, key, order, tokens: torch.Tensor) -> Segment:
        """Encode `tokens` as the segment that follows everything currently in this context."""
        before, total = self.segments, len(self)
        sequence = torch.cat((self.tokens(), tokens))
        groups = torch.cat([torch.full((len(s),), j) for j, s in enumerate(before)] + [torch.full((tokens.numel(),), len(before))])
        result = self.lm.run(sequence, torch.arange(sequence.numel()), torch.arange(sequence.numel()) >= total, self.cells(),
                             groups=groups, keep_queries=True)
        saw = tuple((s.version, start - total) for s, start in zip(before, self.starts()))
        layers, m = self.sketch_layers, tokens.numel()
        G, r, d = self.lm.G, self.lm.H // self.lm.G, self.lm.d_head
        return Segment(key, order, result.cells, content_version(tokens, saw), saw, result.mass.mean(0)[:, : len(before)],
                       torch.zeros(m),
                       sketch_q=result.queries[layers].reshape(len(layers), G, r, m, d).mean(2),
                       sketch_lse=result.lse[layers].reshape(len(layers), G, r, m).mean(2),
                       sketch_start=total)

    def _flag_missing(self) -> None:
        """A segment is missing reads when something now precedes it that it never saw."""
        flagged = []
        for index, segment in enumerate(self.segments):
            known = {version for version, _ in segment.saw}
            lacks = any(earlier.version not in known for earlier in self.segments[:index])
            flagged.append(segment if lacks == segment.missing else replace(segment, missing=lacks))
        self.segments = flagged

    def put(self, key, order, tokens: torch.Tensor) -> None:
        """Add a segment at its place in `order`, encoding it against what precedes it now."""
        if key in self.keys():
            raise KeyError(f"segment {key!r} already present")
        index = sum(s.order <= order for s in self.segments)
        segment = Context(self.lm, self.segments[:index])._encoded_after_all(key, order, tokens)
        self.segments = self.segments[:index] + [segment] + self.segments[index:]
        self._flag_missing()

    def drop(self, key) -> None:
        """Remove a segment. Later segments keep their cells and record what they had read of it."""
        index = self._index(key)
        gone = self.segments[index]
        followers = []
        for segment in self.segments[index + 1:]:
            column = next((c for c, (version, _) in enumerate(segment.saw) if version == gone.version), None)
            followers.append(segment if column is None else replace(segment, debt=segment.debt + segment.lineage[:, column]))
        self.segments = self.segments[:index] + followers

    def swap(self, key, tokens: torch.Tensor) -> None:
        order = self.segments[self._index(key)].order
        self.drop(key)
        self.put(key, order, tokens)

    def fork(self) -> Context:
        return Context(self.lm, self.segments)

    def merge(self, other: Context) -> None:
        """Take every segment `other` has that this context lacks, placed by `order`, keeping its cells."""
        mine = {s.key: s for s in self.segments}
        for segment in other.segments:
            if segment.key in mine and not torch.equal(mine[segment.key].cells.tokens, segment.cells.tokens):
                raise ValueError(f"segment {segment.key!r} differs between the two contexts")
        incoming = [s for s in other.segments if s.key not in mine]
        self.segments = sorted(self.segments + incoming, key=lambda s: s.order)
        self._flag_missing()

    def read(self, tokens: torch.Tensor, groups: torch.Tensor | None = None):
        """Run new input over the current state without storing it. Returns (logits, attention mass or None)."""
        prior = self.tokens()
        sequence = torch.cat((prior, tokens))
        fresh = torch.arange(sequence.numel()) >= prior.numel()
        result = self.lm.run(sequence, torch.arange(sequence.numel()), fresh, self.cells(), groups=groups)
        return self.lm.logits(result.hidden), result.mass

    def sweep(self, budget: int | None = None) -> int:
        """Re-encode stale segments left to right while `budget` tokens allow. Returns tokens recomputed."""
        spent = 0
        while (index := self.frontier()) < len(self.segments):
            segment = self.segments[index]
            if budget is not None and spent + len(segment) > budget:
                break
            fresh = Context(self.lm, self.segments[:index])._encoded_after_all(segment.key, segment.order, segment.cells.tokens)
            self.segments = self.segments[:index] + [fresh] + self.segments[index + 1:]
            spent += len(segment)
        return spent

    def missing_mass(self, index: int) -> torch.Tensor:
        """Estimated attention each token of segment `index` would pay to content now before it that it never read.

        Scores the segment's kept query sketch against the keys of every earlier segment
        it did not see, moved to their current positions. An estimate: the queries come
        from the stale hidden states, and one averaged query stands for each group of heads.
        """
        segment = self.segments[index]
        known = {version for version, _ in segment.saw}
        unseen = [j for j in range(index) if self.segments[j].version not in known]
        if not unseen or segment.sketch_q is None:
            return torch.zeros(len(segment))
        starts = self.starts()
        shift = starts[index] - segment.sketch_start
        total = torch.zeros(len(segment), dtype=torch.float64)
        for row, layer in enumerate(self.sketch_layers):
            logits = []
            for j in unseen:
                other = self.segments[j]
                now = starts[j] + torch.arange(len(other))
                keys = self.lm.rotate(other.cells.K[layer], now - other.cells.enc_pos)
                logits.append(would_be_logits(self.lm, segment.sketch_q[row], shift, keys))
            new = torch.logsumexp(torch.cat(logits, dim=-1), dim=-1)
            total += torch.sigmoid(new - segment.sketch_lse[row]).mean(0).cpu().double()
        mass = (total / len(self.sketch_layers)).float()
        return mass if segment.unsettled is None else mass * segment.unsettled

    def repair(self, reader: torch.Tensor, fraction: float, estimate: str = "sketch", reader_first: bool = False) -> int:
        """Recompute the stale tokens that matter most to `reader`. Returns tokens recomputed.

        Score = (what the token read of removed content, its debt, plus what it would
        have read of inserted content) times how much the reader attends to it.
        estimate: "sketch" estimates the second term from kept queries (`missing_mass`);
        "flat" gives every token of a segment with missing reads the mean recorded debt.
        reader_first: recompute first the stale segment on which `reader` puts the most
        attention per token, then the tokens with the most debt and missing reads.
        """
        if estimate not in ("sketch", "flat"):
            raise ValueError(f"unknown estimate {estimate!r}")
        first = self.frontier()
        if first == len(self.segments):
            return 0
        starts, total = self.starts(), len(self)
        low = starts[first]
        _, mass = self.read(reader, groups=torch.arange(total + reader.numel()))
        attention = mass[:, :, low:total].mean((0, 1))
        stale = list(enumerate(self.segments[first:], start=first))
        debt = torch.cat([s.debt for _, s in stale])
        if estimate == "sketch":
            lacking = torch.cat([self.missing_mass(i) if s.missing else torch.zeros(len(s)) for i, s in stale])
        else:
            prior = debt[debt > 0].mean().item() if (debt > 0).any() else 1.0
            lacking = torch.cat([torch.full((len(s),), prior * float(s.missing)) * (1 if s.unsettled is None else s.unsettled)
                                 for _, s in stale])
        score = attention * (debt + lacking)
        if reader_first:
            lengths = [len(s) for _, s in stale]
            offsets = [sum(lengths[:k]) for k in range(len(lengths))]
            best = max(range(len(stale)), key=lambda k: attention[offsets[k]:offsets[k] + lengths[k]].mean().item())
            exposure = debt + lacking
            score = exposure.clone()
            score[offsets[best]:offsets[best] + lengths[best]] += exposure.max() - exposure.min() + 1.0
        count = min(total - low, max(1, math.ceil(round(fraction * (total - low), 9))))
        chosen = torch.zeros(total, dtype=torch.bool)
        chosen[low + score.topk(count).indices] = True
        healed = heal(self.lm, self.tokens(), torch.arange(total), self.cells(), chosen)
        rebuilt = self.segments[:first]
        for index in range(first, len(self.segments)):
            segment, window = self.segments[index], slice(starts[index], starts[index] + len(self.segments[index]))
            if chosen[window].any():
                keep = (~chosen[window]).float()
                unsettled = keep if segment.unsettled is None else segment.unsettled * keep
                segment = replace(segment, cells=healed.take(window), version=f"patch-{next(_patches)}", patched=True,
                                  debt=segment.debt * keep, unsettled=unsettled)
            rebuilt.append(segment)
        self.segments = rebuilt
        return count
