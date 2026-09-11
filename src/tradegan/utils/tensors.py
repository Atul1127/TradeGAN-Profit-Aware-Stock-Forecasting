"""Small tensor helpers extracted from the original TradeGAN implementation."""

from __future__ import annotations

import torch


def combine_vectors(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Concatenate two tensors along ``dim`` and return float tensors."""
    combined = torch.cat([x, y], dim=dim)
    return combined.to(torch.float)


__all__ = ["combine_vectors"]
