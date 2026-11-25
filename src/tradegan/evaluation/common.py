"""Evaluation helpers shared by the GAN and LSTM evaluators.

Half-day layout
---------------
The excess-return series interleaves log prices as ``open_0, close_0, open_1,
close_1, ...`` and takes differences, so the return at global index ``k`` is

* intraday (open -> close of day ``k // 2``) when ``k`` is even, and
* overnight (close of day ``(k - 1) // 2`` -> next open) when ``k`` is odd.

A trading day is therefore an overnight return followed by that day's intraday
return: global indices ``(2d - 1, 2d)``. When the global index of the first
evaluated target is known, PnL is paired on those day boundaries and the two
halves are labelled correctly. When it is unknown, the legacy behaviour (pair
from position 0, no half labels) is kept.
"""
from __future__ import annotations

import math

import numpy as np
import torch

ANNUALISATION = math.sqrt(252.0)


def is_overnight(global_index: int) -> bool:
    return global_index % 2 == 1


def day_aligned(pnl: torch.Tensor, start: int | None) -> torch.Tensor:
    """Drop a leading intraday return whose overnight half is not in the sample."""
    if start is not None and not is_overnight(start):
        return pnl[1:]
    return pnl


def paired_pnl(pnl: torch.Tensor, start: int | None = None) -> torch.Tensor:
    """Sum consecutive half-day PnLs into daily PnL."""
    pnl = day_aligned(pnl, start)
    usable = 2 * (pnl.numel() // 2)
    if usable == 0:
        raise ValueError("At least two observations are required for paired PnL evaluation")
    return pnl[:usable].reshape(-1, 2).sum(dim=1)


def annualised_sharpe(daily: torch.Tensor) -> float:
    std = daily.std()
    if daily.numel() < 2 or not torch.isfinite(std) or std.item() == 0.0:
        return 0.0
    return float((daily.mean() / std).item() * ANNUALISATION)


def safe_corr(prediction: torch.Tensor, real: torch.Tensor) -> float:
    x = prediction.detach().cpu().numpy().reshape(-1)
    y = real.detach().cpu().numpy().reshape(-1)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def half_day_split(pnl: torch.Tensor, start: int | None):
    """Return ``(overnight, intraday)`` PnL arrays, or ``(even, odd)`` if unlabelled."""
    usable = 2 * (pnl.numel() // 2)
    if start is None:
        return pnl[:usable:2], pnl[1:usable:2]
    indices = torch.arange(start, start + pnl.numel(), device=pnl.device)
    overnight_mask = indices % 2 == 1
    return pnl[overnight_mask], pnl[~overnight_mask]


def position_diagnostics(position: torch.Tensor, real: torch.Tensor, start: int | None,
                         suffix: str = "") -> dict[str, float]:
    """Constant-position baselines and direction-bias diagnostics.

    ``position`` is the traded sign (+1 long the stock vs the index, -1 short).
    A model only shows timing skill if it beats the best constant position; the
    "timing" PnL removes the model's average position, so a model that is
    simply short most of the time scores close to zero here.
    """
    position = position.to(real.dtype)
    always_long = paired_pnl(10000 * real, start)
    always_short = -always_long
    timing = paired_pnl(10000 * (position - position.mean()) * real, start)
    model = paired_pnl(10000 * position * real, start)
    best_constant = max(float(always_long.mean()), float(always_short.mean()))

    out = {
        f"Long share{suffix}": float((position > 0).float().mean()),
        f"Always-long PnL_w{suffix}": float(always_long.mean()),
        f"Always-long SR scaled{suffix}": annualised_sharpe(always_long),
        f"Always-short PnL_w{suffix}": float(always_short.mean()),
        f"Always-short SR scaled{suffix}": annualised_sharpe(always_short),
        f"PnL_w over best constant{suffix}": float(model.mean()) - best_constant,
        f"Timing PnL_w{suffix}": float(timing.mean()),
        f"Timing SR scaled{suffix}": annualised_sharpe(timing),
    }
    if start is not None:
        overnight, intraday = half_day_split(10000 * position * real, start)
        out[f"Overnight PnL mean{suffix}"] = float(overnight.mean()) if overnight.numel() else 0.0
        out[f"Intraday PnL mean{suffix}"] = float(intraday.mean()) if intraday.numel() else 0.0
        out[f"Overnight SR scaled{suffix}"] = annualised_sharpe(overnight)
        out[f"Intraday SR scaled{suffix}"] = annualised_sharpe(intraday)
    return out


__all__ = [
    "ANNUALISATION",
    "annualised_sharpe",
    "day_aligned",
    "half_day_split",
    "is_overnight",
    "paired_pnl",
    "position_diagnostics",
    "safe_corr",
]
