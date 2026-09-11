"""Gradient-scale calibration utilities for TradeGAN."""
from __future__ import annotations

import torch

from .losses import trading_terms


def _gradient_norm(module: torch.nn.Module) -> torch.Tensor:
    total = torch.zeros((), device=next(module.parameters()).device)
    for parameter in module.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.detach().norm(2).square()
    return total.sqrt()


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(numerator.dtype).eps
    return numerator / denominator.clamp_min(eps)


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

    gen.train()
    disc.train()
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
                disc_opt.zero_grad(set_to_none=True)
                fake = gen(torch.randn((1, current, z_dim), device=device), condition, h0g, c0g)
                fake_pred = disc(torch.cat((condition, fake.detach()), dim=-1), h0d, c0d)
                real_pred = disc(torch.cat((condition, real), dim=-1), h0d, c0d)
                dloss = (
                    criterion(fake_pred, torch.zeros_like(fake_pred))
                    + criterion(real_pred, torch.ones_like(real_pred))
                ) / 2
                dloss.backward()
                disc_opt.step()

            fake = gen(torch.randn((1, current, z_dim), device=device), condition, h0g, c0g)
            fake_pred = disc(torch.cat((condition, fake), dim=-1), h0d, c0d)
            prediction = fake.squeeze(0).squeeze(-1)
            target = real.squeeze(0).squeeze(-1)
            pnl, mse, sr, std = trading_terms(prediction, target, tanh_coeff)
            components = [
                criterion(fake_pred, torch.ones_like(fake_pred)),
                pnl,
                mse,
                sr,
                std,
            ]

            for index, component in enumerate(components):
                gen_opt.zero_grad(set_to_none=True)
                component.backward(retain_graph=index < len(components) - 1)
                scales[index].append(_gradient_norm(gen).detach())
            gen_opt.step()

    if not scales[0]:
        zero = torch.ones((), device=device)
        return gen, disc, gen_opt, disc_opt, zero, zero, zero, zero

    bce, pnl, mse, sr, std = (torch.stack(values) for values in scales)
    return (
        gen,
        disc,
        gen_opt,
        disc_opt,
        _safe_ratio(bce, pnl).mean(),
        _safe_ratio(bce, mse).mean(),
        _safe_ratio(bce, sr).mean(),
        _safe_ratio(bce, std).mean(),
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

    gen.train()
    scales = [[] for _ in range(4)]

    for _ in range(n_epochs):
        shuffled = train_data[torch.randperm(train_data.shape[0], device=train_data.device)]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size]
            current = batch.shape[0]
            condition = batch[:, :l].unsqueeze(0).to(device=device, dtype=torch.float)
            real = batch[:, l:l + pred].unsqueeze(0).to(device=device, dtype=torch.float)
            h0 = torch.zeros((1, current, hid_g), device=device)
            fake = gen(condition, h0, torch.zeros_like(h0))
            prediction = fake.squeeze(0).squeeze(-1)
            target = real.squeeze(0).squeeze(-1)
            pnl, mse, sr, std = trading_terms(prediction, target, tanh_coeff)
            components = [pnl, mse, sr, std]

            for index, component in enumerate(components):
                gen_opt.zero_grad(set_to_none=True)
                component.backward(retain_graph=index < len(components) - 1)
                scales[index].append(_gradient_norm(gen).detach())
            gen_opt.step()

    if not scales[0]:
        zero = torch.ones((), device=device)
        return gen, gen_opt, zero, torch.zeros((), device=device), zero, zero

    pnl, mse, sr, std = (torch.stack(values) for values in scales)
    return (
        gen,
        gen_opt,
        _safe_ratio(mse, pnl).mean(),
        torch.zeros((), device=device),
        _safe_ratio(mse, sr).mean(),
        _safe_ratio(mse, std).mean(),
    )


__all__ = ["GradientCheck", "GradientCheckLSTM"]
