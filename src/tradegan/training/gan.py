"""GAN training loops.

The original project contained nine near-duplicate training functions.  This
module keeps their public names/signatures while sharing one implementation so
objective formulas are explicit and optimizer updates are correct.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import torch

from tradegan.utils.tensors import combine_vectors


def _gradient_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().norm(2).item()
        total += norm * norm
    return total**0.5


def _train_gan(
    gen,
    disc,
    gen_opt,
    disc_opt,
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
    del checkpoint_epoch, validation_data, lr_d, lr_g, h
    if objective is None:
        objective = _objective_bce

    if n_epochs < 0:
        raise ValueError("n_epochs must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if diter <= 0:
        raise ValueError("diter must be positive")

    ntrain = train_data.shape[0]
    if ntrain == 0:
        raise ValueError("train_data must not be empty")

    nbatches = (ntrain + batch_size - 1) // batch_size
    discloss: list[float] = []
    genloss: list[float] = []
    dscpred_real: list[float] = []
    dscpred_fake: list[float] = []

    gen.train()
    disc.train()

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

            h0d = torch.zeros((1, current, hid_d), device=device, dtype=torch.float)
            c0d = torch.zeros((1, current, hid_d), device=device, dtype=torch.float)
            h0g = torch.zeros((1, current, hid_g), device=device, dtype=torch.float)
            c0g = torch.zeros((1, current, hid_g), device=device, dtype=torch.float)

            last_fake_pred = None
            last_real_pred = None
            last_disc_loss = None

            for _ in range(diter):
                disc_opt.zero_grad(set_to_none=True)
                noise = torch.randn((1, current, z_dim), device=device, dtype=torch.float)
                fake = gen(noise, condition, h0g, c0g)
                fake_pair = combine_vectors(condition, fake.detach(), dim=-1)
                real_pair = combine_vectors(condition, real, dim=-1)
                fake_pred = disc(fake_pair, h0d, c0d)
                real_pred = disc(real_pair, h0d, c0d)

                fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
                real_loss = criterion(real_pred, torch.ones_like(real_pred))
                disc_loss = (fake_loss + real_loss) / 2
                disc_loss.backward()
                disc_opt.step()

                last_fake_pred = fake_pred.detach()
                last_real_pred = real_pred.detach()
                last_disc_loss = disc_loss.detach().item()

            dscpred_real.append(float(last_real_pred.mean().item()))
            dscpred_fake.append(float(last_fake_pred.mean().item()))
            discloss.append(float(last_disc_loss))

            gen_opt.zero_grad(set_to_none=True)
            noise = torch.randn((1, current, z_dim), device=device, dtype=torch.float)
            fake = gen(noise, condition, h0g, c0g)
            fake_pair = combine_vectors(condition, fake, dim=-1)

            # Freeze discriminator parameters during the generator update.
            for parameter in disc.parameters():
                parameter.requires_grad_(False)
            try:
                fake_pred = disc(fake_pair, h0d, c0d)
                bce = criterion(fake_pred, torch.ones_like(fake_pred))
                gen_loss = objective(
                    bce,
                    fake,
                    real,
                    float(alpha),
                    float(beta),
                    float(gamma),
                    float(delta),
                    tanh_coeff,
                )
                gen_loss.backward()
            finally:
                for parameter in disc.parameters():
                    parameter.requires_grad_(True)

            gen_opt.step()
            genloss.append(float(gen_loss.detach().item()))

    if plot:
        _plot_training(dscpred_fake, dscpred_real, genloss, discloss)

    return gen, disc, gen_opt, disc_opt


def _objective_terms(fake, real, tanh_coeff):
    prediction = fake.squeeze(0).squeeze(-1)
    target = real.squeeze(0).squeeze(-1)
    pnl_samples = torch.tanh(tanh_coeff * prediction) * target
    pnl = pnl_samples.mean()
    mse = torch.norm(prediction - target) ** 2 / prediction.shape[0]
    std = pnl_samples.std()
    sr = pnl / std if torch.isfinite(std) and std.item() > 0 else pnl * 0.0
    return pnl, mse, sr, std


def _objective_bce(bce, fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del fake, real, alpha, beta, gamma, delta, tanh_coeff
    return bce


def _objective_pnl(bce, fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del beta, gamma, delta
    pnl, *_ = _objective_terms(fake, real, tanh_coeff)
    return bce - alpha * pnl


def _objective_pnl_mse(bce, fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del gamma, delta
    pnl, mse, *_ = _objective_terms(fake, real, tanh_coeff)
    return bce - alpha * pnl + beta * mse


def _objective_pnl_mse_sr(bce, fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del delta
    pnl, mse, sr, _ = _objective_terms(fake, real, tanh_coeff)
    return bce - alpha * pnl + beta * mse - gamma * sr


def _objective_pnl_mse_std(bce, fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del gamma
    pnl, mse, _, std = _objective_terms(fake, real, tanh_coeff)
    return bce - alpha * pnl + beta * mse + delta * std


def _objective_pnl_sr(bce, fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del beta, delta
    pnl, _, sr, _ = _objective_terms(fake, real, tanh_coeff)
    return bce - alpha * pnl - gamma * sr


def _objective_mse(bce, fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del alpha, gamma, delta, tanh_coeff
    _, mse, _, _ = _objective_terms(fake, real, 100.0)
    return bce + beta * mse


def _objective_sr(bce, fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del alpha, beta, delta
    _, _, sr, _ = _objective_terms(fake, real, tanh_coeff)
    return bce - gamma * sr


def _objective_sr_mse(bce, fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del alpha, delta
    _, mse, sr, _ = _objective_terms(fake, real, tanh_coeff)
    return bce + beta * mse - gamma * sr


def _objective_pnl_std(bce, fake, real, alpha, beta, gamma, delta, tanh_coeff):
    del beta, gamma
    pnl, _, _, std = _objective_terms(fake, real, tanh_coeff)
    return bce - alpha * pnl + delta * std


def _plot_training(fake_pred, real_pred, genloss, discloss):
    plt.figure("Discriminator predictions")
    plt.plot(fake_pred, alpha=0.5, label="generated")
    plt.plot(real_pred, alpha=0.5, label="real")
    plt.legend(loc="best")
    plt.show()

    plt.figure("Generator loss")
    plt.plot(genloss)
    plt.show()

    plt.figure("Discriminator loss")
    plt.plot(discloss)
    plt.show()


def _run(objective, *args, **kwargs):
    return _train_gan(*args, objective=objective, **kwargs)


def TrainLoopForGAN(*args, **kwargs):
    return _run(_objective_bce, *args, **kwargs)


def TrainLoopMainPnLnv(*args, **kwargs):
    return _run(_objective_pnl, *args, **kwargs)


def TrainLoopMainPnLMSEnv(*args, **kwargs):
    return _run(_objective_pnl_mse, *args, **kwargs)


def TrainLoopMainPnLMSESTDnv(*args, **kwargs):
    return _run(_objective_pnl_mse_std, *args, **kwargs)


def TrainLoopMainPnLMSESRnv(*args, **kwargs):
    return _run(_objective_pnl_mse_sr, *args, **kwargs)


def TrainLoopMainPnLSRnv(*args, **kwargs):
    return _run(_objective_pnl_sr, *args, **kwargs)


def TrainLoopMainMSEnv(*args, **kwargs):
    return _run(_objective_mse, *args, **kwargs)


def TrainLoopMainSRnv(*args, **kwargs):
    return _run(_objective_sr, *args, **kwargs)


def TrainLoopMainSRMSEenv(*args, **kwargs):
    return _run(_objective_sr_mse, *args, **kwargs)


def TrainLoopMainPnLSTDnv(*args, **kwargs):
    return _run(_objective_pnl_std, *args, **kwargs)


__all__ = [
    "TrainLoopForGAN",
    "TrainLoopMainPnLnv",
    "TrainLoopMainPnLMSEnv",
    "TrainLoopMainPnLMSESTDnv",
    "TrainLoopMainPnLMSESRnv",
    "TrainLoopMainPnLSRnv",
    "TrainLoopMainPnLSTDnv",
    "TrainLoopMainMSEnv",
    "TrainLoopMainSRnv",
    "TrainLoopMainSRMSEnv",
]
