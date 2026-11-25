"""Objective formulas used by TradeGAN training."""
from __future__ import annotations

import torch


def trading_terms(prediction: torch.Tensor, real: torch.Tensor, tanh_coefficient: float = 100.0):
    """Return differentiable PnL, MSE, Sharpe and volatility terms.

    ``prediction`` and ``real`` may be 1-D or carry the ``(1, batch, 1)`` shape
    produced by the models; they are flattened before the terms are computed.
    This is the single implementation used by calibration, GAN training and
    LSTM training, so the formula tests cover the code that actually trains.
    """
    prediction = prediction.reshape(-1)
    real = real.reshape(-1)
    if prediction.numel() == 0:
        raise ValueError("prediction must not be empty")
    if prediction.shape != real.shape:
        raise ValueError("prediction and real must contain the same number of values")
    pnl_samples = torch.tanh(tanh_coefficient * prediction) * real
    pnl = pnl_samples.mean()
    mse = torch.norm(prediction - real).square() / prediction.shape[0]
    std = pnl_samples.std()
    eps = torch.finfo(pnl.dtype).eps
    # A batch with (near-)constant PnL has no defined Sharpe ratio. Return zero
    # rather than dividing by eps, which would produce an exploding gradient.
    sr = torch.where(std > eps, pnl / std.clamp_min(eps), torch.zeros_like(pnl))
    return pnl, mse, sr, std


def gan_loss_bce(bce):
    return bce


def gan_loss_pnl(bce, alpha, pnl):
    return bce - alpha * pnl


def gan_loss_pnl_mse(bce, alpha, pnl, beta, mse):
    return bce - alpha * pnl + beta * mse


def gan_loss_pnl_mse_sr(bce, alpha, pnl, beta, mse, gamma, sr):
    return bce - alpha * pnl + beta * mse - gamma * sr


def gan_loss_pnl_mse_std(bce, alpha, pnl, beta, mse, delta, std):
    return bce - alpha * pnl + beta * mse + delta * std


def gan_loss_pnl_sr(bce, alpha, pnl, gamma, sr):
    return bce - alpha * pnl - gamma * sr


def gan_loss_mse(bce, beta, mse):
    return bce + beta * mse


def gan_loss_sr(bce, gamma, sr):
    return bce - gamma * sr


def gan_loss_sr_mse(bce, beta, mse, gamma, sr):
    return bce + beta * mse - gamma * sr


def gan_loss_pnl_std(bce, alpha, pnl, delta, std):
    return bce - alpha * pnl + delta * std


def lstm_loss_mse(mse):
    return mse


def lstm_loss_pnl(mse, alpha, pnl):
    return mse - alpha * pnl


def lstm_loss_pnl_std(mse, alpha, pnl, delta, std):
    return mse - alpha * pnl + delta * std


def lstm_loss_pnl_sr(mse, alpha, pnl, gamma, sr):
    return mse - alpha * pnl - gamma * sr


def lstm_loss_sr(mse, gamma, sr):
    return mse - gamma * sr


def lstm_loss_std(mse, delta, std):
    return mse + delta * std


__all__ = [
    "trading_terms",
    "gan_loss_bce",
    "gan_loss_pnl",
    "gan_loss_pnl_mse",
    "gan_loss_pnl_mse_sr",
    "gan_loss_pnl_mse_std",
    "gan_loss_pnl_sr",
    "gan_loss_mse",
    "gan_loss_sr",
    "gan_loss_sr_mse",
    "gan_loss_pnl_std",
    "lstm_loss_mse",
    "lstm_loss_pnl",
    "lstm_loss_pnl_std",
    "lstm_loss_pnl_sr",
    "lstm_loss_sr",
    "lstm_loss_std",
]
