"""Corrected orchestration around the archived TradeGAN implementation.

The historical implementation in :mod:`tradegan.legacy` is retained unchanged
for provenance.  This module fixes experiment-level issues without rewriting
archived research code in place:

* every objective starts from a fresh model and optimizer;
* ForGAN performs a real generator backward/update step;
* each checkpoint is saved from the generator actually trained for that objective;
* output directories are created and written consistently;
* training statistics use the full training tensor for normalization.
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


GAN_OBJECTIVES: dict[str, Callable] = {
    "PnL": legacy.TrainLoopMainPnLnv,
    "PnL MSE": legacy.TrainLoopMainPnLMSEnv,
    "PnL MSE STD": legacy.TrainLoopMainPnLMSESTDnv,
    "PnL MSE SR": legacy.TrainLoopMainPnLMSESRnv,
    "PnL SR": legacy.TrainLoopMainPnLSRnv,
    "PnL STD": legacy.TrainLoopMainPnLSTDnv,
    "SR": legacy.TrainLoopMainSRnv,
    "SR MSE": legacy.TrainLoopMainSRMSEnv,
    "MSE": legacy.TrainLoopMainMSEnv,
}

LSTM_OBJECTIVES: dict[str, Callable] = {
    "PnL": legacy.TrainLoopnLSTMPnL,
    "PnL STD": legacy.TrainLoopnLSTMPnLSTD,
    "PnL SR": legacy.TrainLoopnLSTMPnLSR,
    "STD": legacy.TrainLoopnLSTMSTD,
    "SR": legacy.TrainLoopnLSTMSR,
    "MSE": legacy.TrainLoopnLSTM,
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
        ticker,
        str(data_dir) + "/",
        str(metadata),
        tr=tr,
        vl=vl,
        h=1,
        l=10,
        pred=1,
        plotcheck=False,
    )



def _new_gan(
    train_data: torch.Tensor,
    device: torch.device,
    z_dim: int,
    hid_g: int,
    hid_d: int,
    lookback: int,
    prediction_horizon: int,
    lr_g: float,
    lr_d: float,
):
    mean, std = _safe_stats(train_data)
    gen = legacy.Generator(
        noise_dim=z_dim,
        cond_dim=lookback,
        hidden_dim=hid_g,
        output_dim=prediction_horizon,
        mean=mean,
        std=std,
    ).to(device)
    disc = legacy.Discriminator(
        in_dim=lookback + prediction_horizon,
        hidden_dim=hid_d,
        mean=mean,
        std=std,
    ).to(device)
    return gen, disc, torch.optim.RMSprop(gen.parameters(), lr=lr_g), torch.optim.RMSprop(
        disc.parameters(), lr=lr_d
    )



def _train_forgan_fixed(
    gen,
    disc,
    gen_opt,
    disc_opt,
    criterion,
    n_epochs: int,
    train_data: torch.Tensor,
    batch_size: int,
    hid_d: int,
    hid_g: int,
    z_dim: int,
    l: int,
    pred: int,
    diter: int,
    device: torch.device,
):
    """Train the adversarial-only baseline with an actual generator update."""
    gen.train()
    ntrain = train_data.shape[0]

    for _ in range(n_epochs):
        permutation = torch.randperm(ntrain, device=train_data.device)
        train_data = train_data[permutation]
        for start in range(0, ntrain, batch_size):
            batch = train_data[start : start + batch_size]
            if batch.numel() == 0:
                continue
            current = batch.shape[0]
            condition = batch[:, :l].unsqueeze(0)
            real = batch[:, l : l + pred].unsqueeze(0)
            h0d = torch.zeros((1, current, hid_d), device=device)
            c0d = torch.zeros((1, current, hid_d), device=device)
            h0g = torch.zeros((1, current, hid_g), device=device)
            c0g = torch.zeros((1, current, hid_g), device=device)

            for _ in range(max(1, diter)):
                disc_opt.zero_grad(set_to_none=True)
                noise = torch.randn(1, current, z_dim, device=device)
                fake = gen(noise, condition, h0g, c0g)
                fake_pair = legacy.combine_vectors(condition, fake.detach(), dim=-1)
                real_pair = legacy.combine_vectors(condition, real, dim=-1)
                fake_pred = disc(fake_pair, h0d, c0d)
                real_pred = disc(real_pair, h0d, c0d)
                dloss = (
                    criterion(fake_pred, torch.zeros_like(fake_pred))
                    + criterion(real_pred, torch.ones_like(real_pred))
                ) / 2
                dloss.backward()
                disc_opt.step()

            gen_opt.zero_grad(set_to_none=True)
            noise = torch.randn(1, current, z_dim, device=device)
            fake = gen(noise, condition, h0g, c0g)
            fake_pair = legacy.combine_vectors(condition, fake, dim=-1)
            fake_pred = disc(fake_pair, h0d, c0d)
            gen_loss = criterion(fake_pred, torch.ones_like(fake_pred))
            gen_loss.backward()
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



def run_gan_experiment(
    ticker: str,
    root: Path,
    gan_epochs: int = 100,
    gradient_epochs: int = 100,
    batch_size: int = 100,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    lookback: int = 10,
    pred: int = 1,
    z_dim: int = 8,
    hid_g: int = 8,
    hid_d: int = 8,
    tanh_coeff: float = 100,
    tr: float = 0.8,
    vl: float = 0.1,
    diter: int = 1,
    use_gpu: bool = True,
) -> pd.DataFrame:
    data_dir = root / "data"
    metadata = root / "stocks-etfs-list.csv"
    metrics_dir = root / "results" / "metrics"
    checkpoints_dir = root / "results" / "checkpoints"
    figures_dir = root / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_np, val_np, test_np, _dates = _split_tcs(ticker, data_dir, metadata, tr=tr, vl=vl)
    device = _device(use_gpu)
    train_data = torch.from_numpy(train_np).float().to(device)
    val_data = torch.from_numpy(val_np).float().to(device)
    test_data = torch.from_numpy(test_np).float().to(device)
    criterion = nn.BCELoss().to(device)
    rows: list[dict] = []

    objective_functions = dict(GAN_OBJECTIVES)
    objective_functions["ForGAN"] = _train_forgan_fixed

    for objective, trainer in objective_functions.items():
        print(f"[{ticker}] GAN objective: {objective}")
        gen, disc, gen_opt, disc_opt = _new_gan(
            train_data, device, z_dim, hid_g, hid_d, lookback, pred, lr_g, lr_d
        )

        if objective == "ForGAN":
            gen, disc, gen_opt, disc_opt = trainer(
                gen,
                disc,
                gen_opt,
                disc_opt,
                criterion,
                gan_epochs,
                train_data,
                batch_size,
                hid_d,
                hid_g,
                z_dim,
                lookback,
                pred,
                diter,
                device,
            )
        else:
            _, _, gen_opt, disc_opt, alpha, beta, gamma, delta = legacy.GradientCheck(
                ticker,
                gen,
                disc,
                gen_opt,
                disc_opt,
                criterion,
                gradient_epochs,
                train_data,
                batch_size,
                hid_d,
                hid_g,
                z_dim,
                lr_d,
                lr_g,
                1,
                lookback,
                pred,
                diter,
                tanh_coeff,
                device,
                False,
            )
            gen, disc, gen_opt, disc_opt = trainer(
                gen,
                disc,
                gen_opt,
                disc_opt,
                criterion,
                alpha,
                beta,
                gamma,
                delta,
                gan_epochs,
                20,
                train_data,
                val_data,
                batch_size,
                hid_d,
                hid_g,
                z_dim,
                lr_d,
                lr_g,
                1,
                lookback,
                pred,
                diter,
                tanh_coeff,
                device,
                False,
            )

        checkpoint = checkpoints_dir / f"{ticker}-GAN-{objective.replace(' ', '_')}-generator.pt"
        torch.save({"g_state_dict": gen.state_dict()}, checkpoint)

        result_df, pnl, _, _, _, _, _, _ = legacy.Evaluation2(
            ticker,
            2,
            gen,
            test_data,
            val_data,
            1,
            lookback,
            pred,
            hid_d,
            hid_g,
            z_dim,
            lr_g,
            lr_d,
            gan_epochs,
            objective,
            0,
            device,
            str(figures_dir) + "/",
            f"{ticker}-GAN-{objective.replace(' ', '-')}",
            False,
        )
        row = result_df.iloc[0].to_dict()
        row["objective"] = objective
        rows.append(row)
        _save_curve(
            figures_dir / f"{ticker}-GAN-{objective.replace(' ', '_')}-cumulative-pnl.png",
            np.asarray(pnl),
            f"Cumulative PnL — {ticker} — {objective}",
        )
        pd.DataFrame(np.asarray(pnl)).to_csv(
            metrics_dir / f"{ticker}-GAN-{objective.replace(' ', '_')}-pnl.csv", index=False
        )

    results = pd.DataFrame(rows)
    results.to_csv(metrics_dir / f"{ticker}-GAN-results.csv", index=False)
    return results



def run_lstm_experiment(
    ticker: str,
    root: Path,
    lstm_epochs: int = 500,
    gradient_epochs: int = 100,
    batch_size: int = 100,
    lr: float = 1e-4,
    lookback: int = 10,
    pred: int = 1,
    hid_g: int = 8,
    tanh_coeff: float = 100,
    tr: float = 0.8,
    vl: float = 0.1,
    use_gpu: bool = True,
) -> pd.DataFrame:
    data_dir = root / "data"
    metadata = root / "stocks-etfs-list.csv"
    metrics_dir = root / "results" / "metrics"
    checkpoints_dir = root / "results" / "checkpoints"
    figures_dir = root / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_np, val_np, test_np, _dates = _split_tcs(ticker, data_dir, metadata, tr=tr, vl=vl)
    device = _device(use_gpu)
    train_data = torch.from_numpy(train_np).float().to(device)
    val_data = torch.from_numpy(val_np).float().to(device)
    test_data = torch.from_numpy(test_np).float().to(device)
    mean, std = _safe_stats(train_data)
    rows: list[dict] = []

    for objective, trainer in LSTM_OBJECTIVES.items():
        print(f"[{ticker}] LSTM objective: {objective}")
        gen = legacy.LSTM(
            noise_dim=0,
            cond_dim=lookback,
            hidden_dim=hid_g,
            output_dim=pred,
            mean=mean,
            std=std,
        ).to(device)
        gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lr)
        _, gen_opt, alpha, beta, gamma, delta = legacy.GradientCheckLSTM(
            ticker,
            gen,
            gen_opt,
            gradient_epochs,
            train_data,
            batch_size,
            hid_g,
            hid_g,
            0,
            0,
            lr,
            1,
            lookback,
            pred,
            1,
            tanh_coeff,
            device,
            False,
        )
        gen, gen_opt = trainer(
            gen,
            gen_opt,
            False,
            alpha,
            beta,
            gamma,
            delta,
            lstm_epochs,
            20,
            train_data,
            val_data,
            batch_size,
            hid_g,
            hid_g,
            0,
            lr,
            lr,
            1,
            lookback,
            pred,
            1,
            tanh_coeff,
            device,
            False,
        )
        checkpoint = checkpoints_dir / f"{ticker}-LSTM-{objective.replace(' ', '_')}-generator.pt"
        torch.save({"g_state_dict": gen.state_dict()}, checkpoint)
        result_df, pnl, _, _ = legacy.Evaluation2LSTM(
            ticker,
            2,
            gen,
            test_data,
            val_data,
            1,
            lookback,
            pred,
            hid_g,
            hid_g,
            0,
            lr,
            lr,
            lstm_epochs,
            objective,
            0,
            device,
            str(figures_dir) + "/",
            f"{ticker}-LSTM-{objective.replace(' ', '-')}",
            False,
        )
        row = result_df.iloc[0].to_dict()
        row["objective"] = objective
        rows.append(row)
        _save_curve(
            figures_dir / f"{ticker}-LSTM-{objective.replace(' ', '_')}-cumulative-pnl.png",
            np.asarray(pnl),
            f"Cumulative PnL — {ticker} — LSTM {objective}",
        )
        pd.DataFrame(np.asarray(pnl)).to_csv(
            metrics_dir / f"{ticker}-LSTM-{objective.replace(' ', '_')}-pnl.csv", index=False
        )

    results = pd.DataFrame(rows)
    results.to_csv(metrics_dir / f"{ticker}-LSTM-results.csv", index=False)
    return results
