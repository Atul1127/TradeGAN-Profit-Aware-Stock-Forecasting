"""Gradient-scale calibration utilities for TradeGAN.

Calibration measures how hard each loss term pushes the generator, so the
terms can be weighted to push about as hard as the base term (BCE for the GAN,
MSE for the LSTM).

Calibration must not change the models that are later trained and compared.
Earlier versions called ``optimizer.step()`` after computing every term's
gradient, which trained the real generator on whichever gradient was computed
last (the PnL-volatility term) and gave every objective except ForGAN extra
training epochs. The functions below now work on deep copies. The copies keep
training on the base term only, so the measured scales follow a realistic
training path, and the caller's models and optimizers are returned unchanged.
"""
from __future__ import annotations

import copy

import torch

from .losses import trading_terms


def _gradient_norm(module: torch.nn.Module) -> torch.Tensor:
    total = torch.zeros((), device=next(module.parameters()).device)
    for parameter in module.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.detach().norm(2).square()
    return total.sqrt()


def _scale_weight(base: torch.Tensor, term: torch.Tensor) -> torch.Tensor:
    """Median of per-batch gradient-size ratios.

    A median is used rather than a mean: when a term's gradient is close to zero
    in a few batches (common for the Sharpe term early on), the ratio for those
    batches explodes and would dominate a mean.
    """
    eps = torch.finfo(base.dtype).eps
    ratios = base / term.clamp_min(eps)
    ratios = ratios[torch.isfinite(ratios)]
    if ratios.numel() == 0:
        return torch.ones((), device=base.device)
    return ratios.median()


def _copy_optimizer(optimizer: torch.optim.Optimizer, module: torch.nn.Module) -> torch.optim.Optimizer:
    """Fresh optimizer of the same class and settings, bound to ``module``."""
    return type(optimizer)(module.parameters(), **optimizer.defaults)


def GradientCheck(
    ticker, gen, disc, gen_opt, disc_opt, criterion, n_epochs, train_data,
    batch_size, hid_d, hid_g, z_dim, lr_d=0.0001, lr_g=0.0001, h=1,
    l=10, pred=1, diter=1, tanh_coeff=100, device="cpu", plot=False
):
    """Estimate relative gradient scales for GAN objective components."""
    del ticker, lr_d, lr_g, h, plot
    if n_epochs < 0 or batch_size <= 0 or diter <= 0:
        raise ValueError("n_epochs must be non-negative; batch_size and diter must be positive")
    if train_data.shape[0] == 0:
        raise ValueError("train_data must not be empty")

    work_gen = copy.deepcopy(gen)
    work_disc = copy.deepcopy(disc)
    work_gen_opt = _copy_optimizer(gen_opt, work_gen)
    work_disc_opt = _copy_optimizer(disc_opt, work_disc)
    work_gen.train()
    work_disc.train()
    scales = [[] for _ in range(5)]

    for _ in range(n_epochs):
        shuffled = train_data[torch.randperm(train_data.shape[0], device=train_data.device)]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size]
            current = batch.shape[0]
            condition = batch[:, :l].unsqueeze(0).to(device=device, dtype=torch.float)
            real = batch[:, l:l + pred].unsqueeze(0).to(device=device, dtype=torch.float)
            h0d = torch.zeros((1, current, hid_d), device=device)
            c0d = torch.zeros_like(h0d)
            h0g = torch.zeros((1, current, hid_g), device=device)
            c0g = torch.zeros_like(h0g)

            for _ in range(diter):
                work_disc_opt.zero_grad(set_to_none=True)
                fake = work_gen(torch.randn((1, current, z_dim), device=device), condition, h0g, c0g)
                fake_pred = work_disc(torch.cat((condition, fake.detach()), dim=-1), h0d, c0d)
                real_pred = work_disc(torch.cat((condition, real), dim=-1), h0d, c0d)
                dloss = (
                    criterion(fake_pred, torch.zeros_like(fake_pred))
                    + criterion(real_pred, torch.ones_like(real_pred))
                ) / 2
                dloss.backward()
                work_disc_opt.step()

            fake = work_gen(torch.randn((1, current, z_dim), device=device), condition, h0g, c0g)
            fake_pred = work_disc(torch.cat((condition, fake), dim=-1), h0d, c0d)
            pnl, mse, sr, std = trading_terms(fake, real, tanh_coeff)
            bce = criterion(fake_pred, torch.ones_like(fake_pred))
            components = [bce, pnl, mse, sr, std]

            for index, component in enumerate(components):
                work_gen_opt.zero_grad(set_to_none=True)
                component.backward(retain_graph=True)
                scales[index].append(_gradient_norm(work_gen).detach())

            # Advance the copy with the adversarial gradient only (ForGAN-style),
            # so later measurements come from a realistically trained generator.
            work_gen_opt.zero_grad(set_to_none=True)
            work_disc.zero_grad(set_to_none=True)
            bce.backward()
            work_gen_opt.step()

    if not scales[0]:
        one = torch.ones((), device=device)
        return gen, disc, gen_opt, disc_opt, one, one, one, one

    bce, pnl, mse, sr, std = (torch.stack(values) for values in scales)
    return (
        gen,
        disc,
        gen_opt,
        disc_opt,
        _scale_weight(bce, pnl),
        _scale_weight(bce, mse),
        _scale_weight(bce, sr),
        _scale_weight(bce, std),
    )


def GradientCheckLSTM(
    ticker, gen, gen_opt, n_epochs, train_data, batch_size, hid_d, hid_g,
    z_dim, lr_d=0.0001, lr_g=0.0001, h=1, l=10, pred=1, diter=1,
    tanh_coeff=100, device="cpu", plot=False
):
    """Estimate relative gradient scales for LSTM objective components."""
    del ticker, hid_d, z_dim, lr_d, lr_g, h, diter, plot
    if n_epochs < 0 or batch_size <= 0:
        raise ValueError("n_epochs must be non-negative and batch_size must be positive")
    if train_data.shape[0] == 0:
        raise ValueError("train_data must not be empty")

    work_gen = copy.deepcopy(gen)
    work_opt = _copy_optimizer(gen_opt, work_gen)
    work_gen.train()
    scales = [[] for _ in range(4)]

    for _ in range(n_epochs):
        shuffled = train_data[torch.randperm(train_data.shape[0], device=train_data.device)]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size]
            current = batch.shape[0]
            condition = batch[:, :l].unsqueeze(0).to(device=device, dtype=torch.float)
            real = batch[:, l:l + pred].unsqueeze(0).to(device=device, dtype=torch.float)
            h0 = torch.zeros((1, current, hid_g), device=device)
            fake = work_gen(condition, h0, torch.zeros_like(h0))
            pnl, mse, sr, std = trading_terms(fake, real, tanh_coeff)
            components = [pnl, mse, sr, std]

            for index, component in enumerate(components):
                work_opt.zero_grad(set_to_none=True)
                component.backward(retain_graph=True)
                scales[index].append(_gradient_norm(work_gen).detach())

            # Advance the copy with the base (MSE) gradient only.
            work_opt.zero_grad(set_to_none=True)
            mse.backward()
            work_opt.step()

    if not scales[0]:
        one = torch.ones((), device=device)
        return gen, gen_opt, one, torch.zeros((), device=device), one, one

    pnl, mse, sr, std = (torch.stack(values) for values in scales)
    return (
        gen,
        gen_opt,
        _scale_weight(mse, pnl),
        torch.zeros((), device=device),
        _scale_weight(mse, sr),
        _scale_weight(mse, std),
    )


__all__ = ["GradientCheck", "GradientCheckLSTM"]
