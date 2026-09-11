"""Trading metrics extracted from the original TradeGAN implementation."""

from __future__ import annotations

import torch


def getPnL(predicted, real, nsamp):
    """Return signed PnL in basis points for ``nsamp`` observations."""
    if nsamp <= 0:
        raise ValueError("nsamp must be positive")
    sgn_fake = torch.sign(predicted)
    pnl = torch.sum(sgn_fake * real)
    return 10000 * pnl / nsamp


def getSR(predicted, real):
    """Return sample Sharpe ratio; zero-volatility series map to zero."""
    signed_returns = torch.sign(predicted) * real
    std = torch.std(signed_returns)
    mean = torch.mean(signed_returns)
    return torch.where(std > 0, mean / std, torch.zeros_like(mean))


__all__ = ["getPnL", "getSR"]
