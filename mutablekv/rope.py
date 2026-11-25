import torch

Phase = tuple[torch.Tensor, torch.Tensor]


def phases(pos: torch.Tensor, inv_freq: torch.Tensor, like: torch.Tensor) -> Phase:
    """cos and sin tables for integer positions `pos` (may be negative).

    Angles are formed in float64 so that composing two rotations and rotating once by
    the sum differ only by rounding in the dtype of `like`.
    """
    angle = pos.cpu().to(torch.float64)[:, None] * inv_freq[None, :]
    angle = torch.cat((angle, angle), dim=-1)
    return angle.cos().to(device=like.device, dtype=like.dtype), angle.sin().to(device=like.device, dtype=like.dtype)


def spin(x: torch.Tensor, phase: Phase) -> torch.Tensor:
    """Rotate x[..., n, d] by a phase table, in the half-split layout of Hugging Face models."""
    cos, sin = phase
    half = x.shape[-1] // 2
    quarter_turn = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + quarter_turn * sin


def rotate(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    return spin(x, phases(pos, inv_freq, x))
