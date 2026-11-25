from __future__ import annotations

import torch

from mutablekv.lm import LM


def spread(layers: int, count: int = 4) -> list[int]:
    """`count` layer indices spread evenly over a model with `layers` layers."""
    return sorted({min(layers - 1, round(layers * (2 * i + 1) / (2 * count))) for i in range(count)})


def would_be_logits(lm: LM, queries: torch.Tensor, shift: int, keys: torch.Tensor) -> torch.Tensor:
    """Scaled logits of kept queries (H, m, d) against new keys (G, n, d) of one layer.

    The queries were rotated at the positions their tokens had when encoded; `shift`
    is how far those tokens have moved since. Returns (H, m, n).
    """
    H, m, d = queries.shape
    G = keys.shape[0]
    moved = lm.rotate(queries, torch.full((m,), shift)) if shift else queries
    return (moved.reshape(G, H // G, m, d) @ keys[:, None].transpose(-1, -2)).reshape(H, m, -1) * d**-0.5


def would_be_exposure(lm: LM, queries: torch.Tensor, lse: torch.Tensor, shift: int, keys: torch.Tensor,
                      layers: list[int] | None = None) -> torch.Tensor:
    """Attention mass each token would have put on new content, averaged over heads and `layers`.

    queries (L, H, m, d) and lse (L, H, m) are kept from when the tokens were encoded,
    keys (L, G, n, d) belong to the new content. Per head the mass is
    exp(new) / (exp(lse) + exp(new)) with new = logsumexp over the new keys. The
    queries are those of the stale hidden states, so beyond the first layer this is an
    estimate, not the attention the cold pass would compute.
    """
    layers = list(range(queries.shape[0])) if layers is None else layers
    total = torch.zeros(queries.shape[2], dtype=torch.float64)
    for layer in layers:
        new = torch.logsumexp(would_be_logits(lm, queries[layer], shift, keys[layer]), dim=-1)
        total += torch.sigmoid(new - lse[layer]).mean(0).cpu().double()
    return (total / len(layers)).float()
