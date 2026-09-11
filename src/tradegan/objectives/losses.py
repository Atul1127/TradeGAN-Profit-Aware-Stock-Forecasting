"""Explicit objective formulas used by the original TradeGAN experiments.

These helpers express the same scalar combinations used by the legacy
training loops. They are not wired into the legacy path yet; keeping them
separate lets us test objective equivalence before replacing any training loop.
"""

import torch


def trading_terms(prediction, real, tanh_coefficient=100.0):
    sign_approx = torch.tanh(tanh_coefficient * prediction)
    pnl_samples = sign_approx * real
    pnl = torch.mean(pnl_samples)
    mse = (torch.norm(prediction - real) ** 2) / prediction.shape[0]
    sr = torch.mean(pnl_samples) / torch.std(pnl_samples)
    std = torch.std(pnl_samples)
    return pnl, mse, sr, std


def gan_loss_bce(discriminator_loss):
    return discriminator_loss


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
