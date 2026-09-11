"""Runnable TradeGAN experiment orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data.splits import split_train_val_test, split_train_val_testraw
from .evaluation.gan import Evaluation2
from .evaluation.lstm import Evaluation2LSTM
from .lstm_model import LSTMForecaster
from .models.gan import Discriminator, Generator
from .objectives.gradient_analysis import GradientCheck, GradientCheckLSTM
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
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def _safe_stats(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = data.mean()
    std = data.std().clamp_min(1e-8)
    return mean, std


def _split_tcs(ticker: str, data_dir: Path, metadata: Path, tr: float = 0.8, vl: float = 0.1):
    if ticker.startswith("X"):
        return split_train_val_testraw(
            ticker, str(data_dir) + "/", tr=tr, vl=vl, h=1, l=10, pred=1, plotcheck=False
        )
    return split_train_val_test(
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


def _new_gan(train: torch.Tensor, device: torch.device, z_dim: int, hid_g: int, hid_d: int,
             lookback: int, pred: int, lr_g: float, lr_d: float):
    mean, std = _safe_stats(train)
    generator = Generator(z_dim, lookback, hid_g, pred, mean, std).to(device)
    discriminator = Discriminator(lookback + pred, hid_d, mean, std).to(device)
    return (
        generator,
        discriminator,
        torch.optim.RMSprop(generator.parameters(), lr=lr_g),
        torch.optim.RMSprop(discriminator.parameters(), lr=lr_d),
    )


def _save_curve(path: Path, values: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(values))
    plt.title(title)
    plt.xlabel("test period")
    plt.ylabel("cumulative PnL (bps)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     objective: str, ticker: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "objective": objective,
            "ticker": ticker,
        },
        path,
    )


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
    mc_samples: int = 1000,
) -> pd.DataFrame:
    root = Path(root)
    metrics = root / "results" / "metrics"
    checkpoints = root / "results" / "checkpoints"
    figures = root / "results" / "figures"
    for directory in (metrics, checkpoints, figures):
        directory.mkdir(parents=True, exist_ok=True)

    train_np, val_np, test_np, _dates = _split_tcs(
        ticker, root / "data", root / "stocks-etfs-list.csv", tr, vl
    )
    device = _device(use_gpu)
    train = torch.from_numpy(train_np).float().to(device)
    val = torch.from_numpy(val_np).float().to(device)
    test = torch.from_numpy(test_np).float().to(device)
    criterion = nn.BCELoss()
    rows: list[dict] = []

    for objective, trainer in {**GAN_OBJECTIVES, "ForGAN": TrainLoopForGAN}.items():
        print(f"[{ticker}] GAN objective: {objective}")
        generator, discriminator, gen_opt, disc_opt = _new_gan(
            train, device, z_dim, hid_g, hid_d, lookback, pred, lr_g, lr_d
        )

        if objective == "ForGAN":
            generator, discriminator, gen_opt, disc_opt = trainer(
                generator,
                discriminator,
                gen_opt,
                disc_opt,
                criterion,
                0.0,
                0.0,
                0.0,
                0.0,
                gan_epochs,
                0,
                train,
                val,
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
        else:
            generator, discriminator, gen_opt, disc_opt, alpha, beta, gamma, delta = GradientCheck(
                ticker,
                generator,
                discriminator,
                gen_opt,
                disc_opt,
                criterion,
                gradient_epochs,
                train,
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
            generator, discriminator, gen_opt, disc_opt = trainer(
                generator,
                discriminator,
                gen_opt,
                disc_opt,
                criterion,
                alpha,
                beta,
                gamma,
                delta,
                gan_epochs,
                20,
                train,
                val,
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

        _save_checkpoint(
            checkpoints / f"{ticker}-GAN-{objective.replace(' ', '_')}-generator.pt",
            generator,
            gen_opt,
            objective,
            ticker,
        )
        result_df, pnl, *_ = Evaluation2(
            ticker,
            2,
            generator,
            test,
            val,
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
            str(figures) + "/",
            objective,
            False,
            mc_samples,
        )
        row = result_df.iloc[0].to_dict()
        row["objective"] = objective
        rows.append(row)
        pnl = np.asarray(pnl)
        _save_curve(
            figures / f"{ticker}-GAN-{objective.replace(' ', '_')}-cumulative-pnl.png",
            pnl,
            f"Cumulative PnL — {ticker} — {objective}",
        )
        pd.DataFrame(pnl).to_csv(
            metrics / f"{ticker}-GAN-{objective.replace(' ', '_')}-pnl.csv", index=False
        )

    results = pd.DataFrame(rows)
    results.to_csv(metrics / f"{ticker}-GAN-results.csv", index=False)
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
    root = Path(root)
    metrics = root / "results" / "metrics"
    checkpoints = root / "results" / "checkpoints"
    figures = root / "results" / "figures"
    for directory in (metrics, checkpoints, figures):
        directory.mkdir(parents=True, exist_ok=True)

    train_np, val_np, test_np, _dates = _split_tcs(
        ticker, root / "data", root / "stocks-etfs-list.csv", tr, vl
    )
    device = _device(use_gpu)
    train = torch.from_numpy(train_np).float().to(device)
    val = torch.from_numpy(val_np).float().to(device)
    test = torch.from_numpy(test_np).float().to(device)
    mean, std = _safe_stats(train)
    rows: list[dict] = []

    for objective, trainer in LSTM_OBJECTIVES.items():
        print(f"[{ticker}] LSTM objective: {objective}")
        model = LSTMForecaster(0, lookback, hid_g, pred, mean, std).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        model, optimizer, alpha, beta, gamma, delta = GradientCheckLSTM(
            ticker,
            model,
            optimizer,
            gradient_epochs,
            train,
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
        model, optimizer = trainer(
            model,
            optimizer,
            False,
            alpha,
            beta,
            gamma,
            delta,
            lstm_epochs,
            20,
            train,
            val,
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
        _save_checkpoint(
            checkpoints / f"{ticker}-LSTM-{objective.replace(' ', '_')}-generator.pt",
            model,
            optimizer,
            objective,
            ticker,
        )
        result_df, pnl, *_ = Evaluation2LSTM(
            ticker,
            2,
            model,
            test,
            val,
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
            str(figures) + "/",
            objective,
            False,
        )
        row = result_df.iloc[0].to_dict()
        row["objective"] = objective
        rows.append(row)
        pnl = np.asarray(pnl)
        _save_curve(
            figures / f"{ticker}-LSTM-{objective.replace(' ', '_')}-cumulative-pnl.png",
            pnl,
            f"Cumulative PnL — {ticker} — LSTM {objective}",
        )
        pd.DataFrame(pnl).to_csv(
            metrics / f"{ticker}-LSTM-{objective.replace(' ', '_')}-pnl.csv", index=False
        )

    results = pd.DataFrame(rows)
    results.to_csv(metrics / f"{ticker}-LSTM-results.csv", index=False)
    return results
