"""LSTM objective training loops.

All six historical LSTM objectives share the same batching and optimization
logic.  The objective-specific differences live in small functions below.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import torch


def _terms(fake, real, tanh_coeff):
    prediction = fake.squeeze(0).squeeze(-1)
    target = real.squeeze(0).squeeze(-1)
    pnl_samples = torch.tanh(tanh_coeff * prediction) * target
    pnl = pnl_samples.mean()
    mse = torch.norm(prediction - target) ** 2 / prediction.shape[0]
    std = pnl_samples.std()
    sr = pnl / std if torch.isfinite(std) and std.item() > 0 else pnl * 0.0
    return pnl, mse, sr, std


def _train_lstm(
    gen,
    gen_opt,
    criterion,
    alpha,
    beta,
    gamma,
    delta,
    n_epochs,
    checkpoint_epoch,
    train_data,
    validation_data,
    batch_size,
    hid_d,
    hid_g,
    z_dim,
    lr_d=0.0001,
    lr_g=0.0001,
    h=1,
    l=10,
    pred=1,
    diter=1,
    tanh_coeff=100,
    device="cpu",
    plot=False,
    objective: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float, float], torch.Tensor] | None = None,
):
    del criterion, checkpoint_epoch, validation_data, hid_d, z_dim, lr_d, lr_g, h, diter
    if objective is None:
        objective = _lstm_mse

    if n_epochs < 0:
        raise ValueError("n_epochs must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if train_data.shape[0] == 0:
        raise ValueError("train_data must not be empty")

    ntrain = train_data.shape[0]
    nbatches = (ntrain + batch_size - 1) // batch_size
    losses: list[float] = []
    gen.train()

    for _epoch in range(n_epochs):
        permutation = torch.randperm(ntrain, device=train_data.device)
        shuffled = train_data[permutation]
        for batch_index in range(nbatches):
            start = batch_index * batch_size
            batch = shuffled[start : start + batch_size]
            current = batch.shape[0]
            if current == 0:
                continue

            condition = batch[:, :l].unsqueeze(0).to(device=device, dtype=torch.float)
            real = batch[:, l : l + pred].unsqueeze(0).to(device=device, dtype=torch.float)
            h0 = torch.zeros((1, current, hid_g), device=device, dtype=torch.float)
            c0 = torch.zeros((1, current, hid_g), device=device, dtype=torch.float)

            gen_opt.zero_grad(set_to_none=True)
            fake = gen(condition, h0, c0)
            loss = objective(fake, real, alpha, beta, gamma, delta, tanh_coeff)
            loss.backward()
            gen_opt.step()
            losses.append(float(loss.detach().item()))

    if plot:
        plt.figure("LSTM loss")
        plt.plot(losses)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return gen, gen_opt


def _lstm_mse(fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del alpha, gamma, delta, tanh_coeff
    _, mse, _, _ = _terms(fake, real, 100.0)
    return mse


def _lstm_pnl(fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del beta, gamma, delta
    pnl, mse, _, _ = _terms(fake, real, tanh_coeff)
    return mse - alpha * pnl


def _lstm_pnl_std(fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del beta, gamma
    pnl, mse, _, std = _terms(fake, real, tanh_coeff)
    return mse - alpha * pnl + delta * std


def _lstm_pnl_sr(fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del beta, delta
    pnl, mse, sr, _ = _terms(fake, real, tanh_coeff)
    return mse - alpha * pnl - gamma * sr


def _lstm_sr(fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del alpha, beta, delta
    pnl, mse, sr, _ = _terms(fake, real, tanh_coeff)
    return mse - gamma * sr


def _lstm_std(fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del alpha, beta, gamma
    _, mse, _, std = _terms(fake, real, tanh_coeff)
    return mse + delta * std


def _run(objective, *args, **kwargs):
    return _train_lstm(*args, objective=objective, **kwargs)


def TrainLoopnLSTMPnL(*args, **kwargs):
    return _run(_lstm_pnl, *args, **kwargs)


def TrainLoopnLSTMPnLSTD(*args, **kwargs):
    return _run(_lstm_pnl_std, *args, **kwargs)


def TrainLoopnLSTMPnLSR(*args, **kwargs):
    return _run(_lstm_pnl_sr, *args, **kwargs)


def TrainLoopnLSTMSR(*args, **kwargs):
    return _run(_lstm_sr, *args, **kwargs)


def TrainLoopnLSTMSTD(*args, **kwargs):
    return _run(_lstm_std, *args, **kwargs)


def TrainLoopnLSTM(*args, **kwargs):
    return _run(_lstm_mse, *args, **kwargs)


__all__ = [
    "TrainLoopnLSTMPnL",
    "TrainLoopnLSTMPnLSTD",
    "TrainLoopnLSTMPnLSR",
    "TrainLoopnLSTMSR",
    "TrainLoopnLSTMSTD",
    "TrainLoopnLSTM",
]
