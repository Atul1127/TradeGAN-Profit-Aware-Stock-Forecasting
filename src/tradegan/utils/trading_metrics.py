"""Trading metrics extracted from the original TradeGAN implementation."""

from __future__ import annotations

import torch


def getPnL(predicted, real, nsamp):
    sgn_fake = torch.sign(predicted)
    PnL = torch.sum(sgn_fake * real)
    PnL = 10000 * PnL / nsamp
    return PnL


def getSR(predicted, real):
    sgn_fake = torch.sign(predicted)
    SR = torch.mean(sgn_fake * real) / torch.std(sgn_fake * real)
    return SR


__all__ = ["getPnL", "getSR"]
