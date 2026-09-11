"""Runnable TradeGAN experiment orchestration.

Objective training is provided by the extracted training modules.  The legacy
module is retained only for the still-migrating gradient calibration and
historical evaluation routines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import legacy
from .lstm_model import LSTMForecaster
from .models.gan import Discriminator, Generator
from .training.gan import (
    TrainLoopForGAN,
    TrainLoopMainMSEnv,
    TrainLoopMainPnLMSESTDnv,
    TrainLoopMainPnLMSESRnv,
    TrainLoopMainPnLMSEnv,
    TrainLoopMainPnLSTDnv,
    TrainLoopMainPnLSRnv,
    TrainLoopMainPnLnv,
    TrainLoopMainSRMSEnv,
    TrainLoopMainSRnv,
)
from .training.lstm import (
    TrainLoopnLSTM,
    TrainLoopnLSTMPnL,
    TrainLoopnLSTMPnLSTD,
    TrainLoopnLSTMPnLSR,
    TrainLoopnLSTMSR,
    TrainLoopnLSTMSTD,
)


GAN_OBJECTIVES: dict[str, Callable] = {
    "PnL": TrainLoopMainPnLnv,
    "PnL MSE": TrainLoopMainPnLMSEnv,
    "PnL MSE STD": TrainLoopMainPnLMSESTDnv,
    "PnL MSE SR": TrainLoopMainPnLMSESRnv,
    "PnL SR": TrainLoopMainPnLSRnv,
    "PnL STD": TrainLoopMainPnLSTDnv,
    "SR": TrainLoopMainSRnv,
    "SR MSE": TrainLoopMainSRMSEnv,
    "MSE": TrainLoopMainMSEnv,
}

LSTM_OBJECTIVES: dict[str, Callable] = {
    "PnL": TrainLoopnLSTMPnL,
    "PnL STD": TrainLoopnLSTMPnLSTD,
    "PnL SR": TrainLoopnLSTMPnLSR,
    "STD": TrainLoopnLSTMSTD,
    "SR": TrainLoopnLSTMSR,
    "MSE": TrainLoopnLSTM,
}


def _device(use_gpu: bool = True) -> torch.device:
    return torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")


def _safe_stats(train_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_data.mean()
    std = train_data.std().clamp_min(1e-8)
    return mean, std


def _split_tcs(ticker: str, data_dir: Path, metadata: Path, tr: float = 0.8, vl: float = 0.1):
    if ticker.startswith("X"):
        return legacy.split_train_val_testraw(
            ticker, str(data_dir) + "/", tr=tr, vl=vl, h=1, l=10, pred=1, plotcheck=False
        )
    return legacy.split_train_val_test(
        ticker, str(data_dir) + "/", str(metadata), tr=tr, vl=vl, h=1, l=10, pred=1, plotcheck=False
    )


def _new_gan(train_data, device, z_dim, hid_g, hid_d, lookback, pred, lr_g, lr_d):
    mean, std = _safe_stats(train_data)
    gen = Generator(z_dim, lookback, hid_g, pred, mean, std).to(device)
    disc = Discriminator(lookback + pred, hid_d, mean, std).to(device)
    return gen, disc, torch.optim.RMSprop(gen.parameters(), lr=lr_g), torch.optim.RMSprop(disc.parameters(), lr=lr_d)


def _train_forgan_fixed(gen, disc, gen_opt, disc_opt, criterion, n_epochs, train_data, batch_size,
                        hid_d, hid_g, z_dim, l, pred, diter, device):
    if n_epochs < 0 or batch_size <= 0 or diter <= 0:
        raise ValueError("n_epochs, batch_size and diter must be positive (epochs may be zero)")
    ntrain = train_data.shape[0]
    if ntrain == 0:
        raise ValueError("train_data must not be empty")
    for _ in range(n_epochs):
        shuffled = train_data[torch.randperm(ntrain, device=train_data.device)]
        for start in range(0, ntrain, batch_size):
            batch = shuffled[start:start + batch_size]
            current = batch.shape[0]
            condition = batch[:, :l].unsqueeze(0).to(device=device, dtype=torch.float)
            real = batch[:, l:l + pred].unsqueeze(0).to(device=device, dtype=torch.float)
            h0d = torch.zeros((1, current, hid_d), device=device)
            c0d = torch.zeros((1, current, hid_d), device=device)
            h0g = torch.zeros((1, current, hid_g), device=device)
            c0g = torch.zeros((1, current, hid_g), device=device)
            for _ in range(diter):
                disc_opt.zero_grad(set_to_none=True)
                noise = torch.randn((1, current, z_dim), device=device)
                fake = gen(noise, condition, h0g, c0g)
                fp = disc(torch.cat((condition, fake.detach()), dim=-1), h0d, c0d)
                rp = disc(torch.cat((condition, real), dim=-1), h0d, c0d)
                loss = (criterion(fp, torch.zeros_like(fp)) + criterion(rp, torch.ones_like(rp))) / 2
                loss.backward()
                disc_opt.step()
            gen_opt.zero_grad(set_to_none=True)
            noise = torch.randn((1, current, z_dim), device=device)
            fake = gen(noise, condition, h0g, c0g)
            for p in disc.parameters():
                p.requires_grad_(False)
            try:
                fp = disc(torch.cat((condition, fake), dim=-1), h0d, c0d)
                loss = criterion(fp, torch.ones_like(fp))
                loss.backward()
            finally:
                for p in disc.parameters():
                    p.requires_grad_(True)
            gen_opt.step()
    return gen, disc, gen_opt, disc_opt


def _save_curve(path: Path, values: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(values))
    plt.title(title)
    plt.xlabel("test period")
    plt.ylabel("cumulative PnL (bps)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_gan_experiment(ticker: str, root: Path, gan_epochs: int = 100, gradient_epochs: int = 100,
                       batch_size: int = 100, lr_g: float = 1e-4, lr_d: float = 1e-4,
                       lookback: int = 10, pred: int = 1, z_dim: int = 8, hid_g: int = 8,
                       hid_d: int = 8, tanh_coeff: float = 100, tr: float = 0.8,
                       vl: float = 0.1, diter: int = 1, use_gpu: bool = True) -> pd.DataFrame:
    data_dir = root / "data"
    metadata = root / "stocks-etfs-list.csv"
    metrics_dir = root / "results" / "metrics"
    checkpoints_dir = root / "results" / "checkpoints"
    figures_dir = root / "results" / "figures"
    for path in (metrics_dir, checkpoints_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_np, val_np, test_np, _ = _split_tcs(ticker, data_dir, metadata, tr, vl)
    device = _device(use_gpu)
    train = torch.from_numpy(train_np).float().to(device)
    val = torch.from_numpy(val_np).float().to(device)
    test = torch.from_numpy(test_np).float().to(device)
    criterion = nn.BCELoss().to(device)
    rows: list[dict] = []
    objectives = dict(GAN_OBJECTIVES)
    objectives["ForGAN"] = _train_forgan_fixed

    for name, trainer in objectives.items():
        print(f"[{ticker}] GAN objective: {name}")
        gen, disc, gen_opt, disc_opt = _new_gan(train, device, z_dim, hid_g, hid_d, lookback, pred, lr_g, lr_d)
        if name == "ForGAN":
            gen, disc, gen_opt, disc_opt = trainer(
                gen, disc, gen_opt, disc_opt, criterion, gan_epochs, train, batch_size,
                hid_d, hid_g, z_dim, lookback, pred, diter, device
            )
        else:
            _, _, gen_opt, disc_opt, alpha, beta, gamma, delta = legacy.GradientCheck(
                ticker, gen, disc, gen_opt, disc_opt, criterion, gradient_epochs, train,
                batch_size, hid_d, hid_g, z_dim, lr_d, lr_g, 1, lookback, pred, diter,
                tanh_coeff, device, False
            )
            gen, disc, gen_opt, disc_opt = trainer(
                gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta,
                gan_epochs, 20, train, val, batch_size, hid_d, hid_g, z_dim,
                lr_d, lr_g, 1, lookback, pred, diter, tanh_coeff, device, False
            )

        checkpoint = checkpoints_dir / f"{ticker}-GAN-{name.replace(' ', '_')}-generator.pt"
        torch.save({"model_state_dict": gen.state_dict(), "optimizer_state_dict": gen_opt.state_dict(),
                    "objective": name, "ticker": ticker}, checkpoint)

        result_df, pnl, *_ = legacy.Evaluation2(
            ticker, 2, gen, test, val, 1, lookback, pred, hid_d, hid_g, z_dim,
            lr_g, lr_d, gan_epochs, name, 0, device, str(figures_dir) + "/",
            f"{ticker}-GAN-{name.replace(' ', '-')}", False
        )
        row = result_df.iloc[0].to_dict()
        row["objective"] = name
        rows.append(row)
        pnl = np.asarray(pnl)
        _save_curve(figures_dir / f"{ticker}-GAN-{name.replace(' ', '_')}-cumulative-pnl.png", pnl,
                    f"Cumulative PnL — {ticker} — {name}")
        pd.DataFrame(pnl).to_csv(metrics_dir / f"{ticker}-GAN-{name.replace(' ', '_')}-pnl.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(metrics_dir / f"{ticker}-GAN-results.csv", index=False)
    return results


def run_lstm_experiment(ticker: str, root: Path, lstm_epochs: int = 500, gradient_epochs: int = 100,
                        batch_size: int = 100, lr: float = 1e-4, lookback: int = 10,
                        pred: int = 1, hid_g: int = 8, tanh_coeff: float = 100,
                        tr: float = 0.8, vl: float = 0.1, use_gpu: bool = True) -> pd.DataFrame:
    data_dir = root / "data"
    metadata = root / "stocks-etfs-list.csv"
    metrics_dir = root / "results" / "metrics"
    checkpoints_dir = root / "results" / "checkpoints"
    figures_dir = root / "results" / "figures"
    for path in (metrics_dir, checkpoints_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_np, val_np, test_np, _ = _split_tcs(ticker, data_dir, metadata, tr, vl)
    device = _device(use_gpu)
    train = torch.from_numpy(train_np).float().to(device)
    val = torch.from_numpy(val_np).float().to(device)
    test = torch.from_numpy(test_np).float().to(device)
    mean, std = _safe_stats(train)
    rows: list[dict] = []

    for name, trainer in LSTM_OBJECTIVES.items():
        print(f"[{ticker}] LSTM objective: {name}")
        gen = LSTMForecaster(0, lookback, hid_g, pred, mean, std).to(device)
        opt = torch.optim.RMSprop(gen.parameters(), lr=lr)
        _, opt, alpha, beta, gamma, delta = legacy.GradientCheckLSTM(
            ticker, gen, opt, gradient_epochs, train, batch_size, hid_g, hid_g, 0,
            0, lr, 1, lookback, pred, 1, tanh_coeff, device, False
        )
        gen, opt = trainer(
            gen, opt, False, alpha, beta, gamma, delta, lstm_epochs, 20, train, val,
            batch_size, hid_g, hid_g, 0, lr, lr, 1, lookback, pred, 1, tanh_coeff, device, False
        )

        checkpoint = checkpoints_dir / f"{ticker}-LSTM-{name.replace(' ', '_')}-generator.pt"
        torch.save({"model_state_dict": gen.state_dict(), "optimizer_state_dict": opt.state_dict(),
                    "objective": name, "ticker": ticker}, checkpoint)
        result_df, pnl, *_ = legacy.Evaluation2LSTM(
            ticker, 2, gen, test, val, 1, lookback, pred, hid_g, hid_g, 0,
            lr, lr, lstm_epochs, name, 0, device, str(figures_dir) + "/",
            f"{ticker}-LSTM-{name.replace(' ', '-')}", False
        )
        row = result_df.iloc[0].to_dict()
        row["objective"] = name
        rows.append(row)
        pnl = np.asarray(pnl)
        _save_curve(figures_dir / f"{ticker}-LSTM-{name.replace(' ', '_')}-cumulative-pnl.png", pnl,
                    f"Cumulative PnL — {ticker} — LSTM {name}")
        pd.DataFrame(pnl).to_csv(metrics_dir / f"{ticker}-LSTM-{name.replace(' ', '_')}-pnl.csv", index=False)

    results = pd.DataFrame(rows)
    results.to_csv(metrics_dir / f"{ticker}-LSTM-results.csv", index=False)
    return results
