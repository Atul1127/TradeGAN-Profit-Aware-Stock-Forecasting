"""Trading and forecasting metrics used by TradeGAN evaluation."""

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


def directional_accuracy(predicted: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Return the fraction of observations with the correct return direction.

    Zero predictions are treated as incorrect unless the realized return is also
    exactly zero, matching the sign-based trading convention used elsewhere.
    """
    predicted = predicted.reshape(-1)
    real = real.reshape(-1)
    if predicted.numel() == 0 or predicted.numel() != real.numel():
        raise ValueError("predicted and real must be non-empty and have equal length")
    return (torch.sign(predicted) == torch.sign(real)).to(torch.float32).mean()


def pnl_std(pnl: torch.Tensor) -> torch.Tensor:
    """Return sample standard deviation of a PnL series."""
    pnl = pnl.reshape(-1)
    if pnl.numel() < 2:
        raise ValueError("at least two PnL observations are required")
    return torch.std(pnl)


def percent_reduction(new: float, baseline: float) -> float:
    """Return percentage reduction of ``new`` relative to ``baseline``."""
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return 100.0 * (1.0 - (new / baseline))


def fold_change(new: float, baseline: float) -> float:
    """Return ``new / baseline`` as a fold-change metric."""
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return float(new / baseline)


__all__ = [
    "getPnL",
    "getSR",
    "directional_accuracy",
    "pnl_std",
    "percent_reduction",
    "fold_change",
]
