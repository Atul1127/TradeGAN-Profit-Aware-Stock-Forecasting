"""Monte-Carlo evaluation for the conditional GAN."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..utils.trading_metrics import directional_accuracy, pnl_std
from .common import (
    DEFAULT_COST_BPS,
    annualised_sharpe,
    half_day_split,
    paired_pnl,
    position_diagnostics,
    safe_corr,
)


def _evaluate(model, data, lookback, hidden_dim, latent_dim, device, mc_samples, start=None):
    n = data.shape[0]
    if n == 0:
        raise ValueError("evaluation data must not be empty")

    condition = data[:, :lookback].unsqueeze(0).to(device=device, dtype=torch.float)
    real = data[:, -1].to(device=device, dtype=torch.float)
    h0 = torch.zeros((1, n, hidden_dim), device=device)
    c0 = torch.zeros_like(h0)

    draws = []
    model.eval()
    with torch.no_grad():
        for _ in range(mc_samples):
            noise = torch.randn((1, n, latent_dim), device=device)
            prediction = model(noise, condition, h0, c0).squeeze(0).squeeze(-1)
            draws.append(prediction)

    samples = torch.stack(draws)
    mean_prediction = samples.mean(dim=0)
    position = torch.sign(mean_prediction)
    signed_pnl = 10000 * position * real
    paired = paired_pnl(signed_pnl, start)
    mse = torch.mean((mean_prediction - real) ** 2)
    mae = torch.mean(torch.abs(mean_prediction - real))
    return {
        "mean": mean_prediction,
        "samples": samples,
        "real": real,
        "position": position,
        "pnl": signed_pnl,
        "paired_pnl": paired,
        "mse": mse,
        "rmse": torch.sqrt(mse),
        "mae": mae,
        "sharpe_scaled": annualised_sharpe(paired),
        "prob_positive": (samples > 0).float().mean(dim=0),
    }


def Evaluation2(
    ticker,
    freq,
    gen,
    test_data,
    val_data,
    h,
    l,
    pred,
    hid_d,
    hid_g,
    z_dim,
    lrg,
    lrd,
    n_epochs,
    losstype,
    sr_val,
    device,
    plotsloc,
    f_name,
    plot=False,
    mc_samples=1000,
    test_start=None,
    val_start=None,
    cost_bps=DEFAULT_COST_BPS,
):
    """Evaluate one GAN on test and validation data.

    ``test_start`` and ``val_start`` are the global half-day indices of the
    first target in each split. When given, daily PnL is paired on real day
    boundaries and overnight/intraday results are labelled correctly.
    """
    del freq, h, pred, sr_val, plotsloc, f_name, plot
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    if l <= 0 or hid_g <= 0 or z_dim <= 0:
        raise ValueError("lookback, hidden_dim and latent_dim must be positive")

    test = _evaluate(gen, test_data, l, hid_g, z_dim, device, mc_samples, test_start)
    val = _evaluate(gen, val_data, l, hid_g, z_dim, device, mc_samples, val_start)

    test_real = test["real"]
    val_real = val["real"]
    test_mean = test["mean"]
    val_mean = val["mean"]

    # (overnight, intraday) when test_start is known, otherwise (even, odd).
    first_half, second_half = (x.cpu().numpy() for x in half_day_split(test["pnl"], test_start))
    distribution = test["samples"][1 if mc_samples > 1 else 0].cpu().numpy()
    metrics = {
        "lrd": lrd,
        "lrg": lrg,
        "type": losstype,
        "epochs": n_epochs,
        "ticker": ticker,
        "hid_g": hid_g,
        "hid_d": hid_d,
        "MSE": float(test["mse"].item()),
        "RMSE": float(test["rmse"].item()),
        "MAE": float(test["mae"].item()),
        "PnL_w": float(test["paired_pnl"].mean().item()),
        "SR_w scaled": test["sharpe_scaled"],
        "PnL STD": float(pnl_std(test["paired_pnl"]).item()),
        "Directional Accuracy": float(directional_accuracy(test_mean, test_real).item()),
        "MSE val": float(val["mse"].item()),
        "RMSE val": float(val["rmse"].item()),
        "MAE val": float(val["mae"].item()),
        "PnL_w val": float(val["paired_pnl"].mean().item()),
        "SR_w scaled val": val["sharpe_scaled"],
        "PnL STD val": float(pnl_std(val["paired_pnl"]).item()),
        "Directional Accuracy val": float(directional_accuracy(val_mean, val_real).item()),
        "Corr": safe_corr(test_mean, test_real),
        "Corr val": safe_corr(val_mean, val_real),
        "Pos mn": float((test_mean > 0).float().mean()),
        "Neg mn": float((test_mean < 0).float().mean()),
        "Pos mn val": float((val_mean > 0).float().mean()),
        "Neg mn val": float((val_mean < 0).float().mean()),
        "narrow dist": bool(test["samples"][0].std() < 0.0002),
        "narrow means dist": bool(test_mean.std() < 0.0002),
        "Mean P(up)": float(test["prob_positive"].mean()),
    }
    metrics.update(position_diagnostics(test["position"], test_real, test_start, cost_bps=cost_bps))
    metrics.update(position_diagnostics(val["position"], val_real, val_start, suffix=" val", cost_bps=cost_bps))

    return (
        pd.DataFrame([metrics]),
        test["paired_pnl"].detach().cpu().numpy(),
        first_half,
        second_half,
        test_mean.detach().cpu().numpy(),
        test_real.detach().cpu().numpy(),
        distribution,
        test_real.detach().cpu().numpy(),
    )


def Evaluation3(
    tickers,
    freq,
    gen,
    test,
    val,
    h,
    l,
    pred,
    hid_d,
    hid_g,
    z_dim,
    lrg,
    lrd,
    n_epochs,
    losstype,
    sr_val,
    device,
    plotsloc,
    f_name,
    plot=False,
):
    rows = []
    test_pnls = []
    val_pnls = []
    means_test = []
    means_val = []

    for index, ticker in enumerate(tickers):
        result, test_pnl, _, _, test_mean, _, _, _ = Evaluation2(
            ticker, freq, gen, test[index], val[index], h, l, pred, hid_d, hid_g,
            z_dim, lrg, lrd, n_epochs, losstype, sr_val, device, plotsloc, f_name,
            plot,
        )
        _val_result, val_pnl, _, _, val_mean, *_ = Evaluation2(
            ticker, freq, gen, val[index], test[index], h, l, pred, hid_d, hid_g,
            z_dim, lrg, lrd, n_epochs, losstype, sr_val, device, plotsloc, f_name,
            plot,
        )
        rows.append(result.iloc[0])
        test_pnls.append(test_pnl)
        val_pnls.append(val_pnl)
        means_test.append(test_mean)
        means_val.append(val_mean)

    return (
        pd.DataFrame(rows).reset_index(drop=True),
        np.sum(test_pnls, axis=0),
        np.sum(val_pnls, axis=0),
        np.asarray(means_test),
        np.asarray(means_val),
    )


__all__ = ["Evaluation2", "Evaluation3"]
