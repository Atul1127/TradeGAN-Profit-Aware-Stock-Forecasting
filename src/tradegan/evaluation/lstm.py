"""Evaluation utilities for deterministic LSTM forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..utils.trading_metrics import directional_accuracy, pnl_std
from .common import annualised_sharpe, half_day_split, paired_pnl, position_diagnostics, safe_corr


def _evaluate(model, data, lookback, hidden_dim, device, start=None):
    n = data.shape[0]
    if n == 0:
        raise ValueError("evaluation data must not be empty")
    condition = data[:, :lookback].unsqueeze(0).to(device=device, dtype=torch.float)
    real = data[:, -1].to(device=device, dtype=torch.float)
    h0 = torch.zeros((1, n, hidden_dim), device=device)
    c0 = torch.zeros_like(h0)
    model.eval()
    with torch.no_grad():
        prediction = model(condition, h0, c0).squeeze(0).squeeze(-1)
    position = torch.sign(prediction)
    pnl = 10000 * position * real
    paired = paired_pnl(pnl, start)
    mse = torch.mean((prediction - real) ** 2)
    return prediction, real, position, pnl, paired, annualised_sharpe(paired), mse


def Evaluation2LSTM(
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
    test_start=None,
    val_start=None,
):
    """Evaluate deterministic LSTM forecasts on test and validation data.

    See :func:`tradegan.evaluation.gan.Evaluation2` for ``test_start``/``val_start``.
    """
    del freq, h, pred, hid_d, z_dim, sr_val, plotsloc, f_name, plot

    test_pred, test_real, test_pos, test_pnl, test_pair, test_sr, test_mse = _evaluate(
        gen, test_data, l, hid_g, device, test_start
    )
    val_pred, val_real, val_pos, val_pnl, val_pair, val_sr, val_mse = _evaluate(
        gen, val_data, l, hid_g, device, val_start
    )
    metrics = {
        "lrd": lrd,
        "lrg": lrg,
        "type": losstype,
        "epochs": n_epochs,
        "ticker": ticker,
        "MSE": float(test_mse.item()),
        "RMSE": float(torch.sqrt(test_mse).item()),
        "MAE": float(torch.mean(torch.abs(test_pred - test_real)).item()),
        "PnL_m test": float(test_pair.mean().item()),
        "SR_m scaled test": test_sr,
        "PnL STD": float(pnl_std(test_pair).item()),
        "Directional Accuracy": float(directional_accuracy(test_pred, test_real).item()),
        "MSE val": float(val_mse.item()),
        "RMSE val": float(torch.sqrt(val_mse).item()),
        "MAE val": float(torch.mean(torch.abs(val_pred - val_real)).item()),
        "PnL_m val": float(val_pair.mean().item()),
        "SR_m scaled val": val_sr,
        "PnL STD val": float(pnl_std(val_pair).item()),
        "Directional Accuracy val": float(directional_accuracy(val_pred, val_real).item()),
        "Corr": safe_corr(test_pred, test_real),
        "Corr val": safe_corr(val_pred, val_real),
        "Pos mn": float((test_pred > 0).float().mean()),
        "Neg mn": float((test_pred < 0).float().mean()),
        "Pos mn val": float((val_pred > 0).float().mean()),
        "Neg mn val": float((val_pred < 0).float().mean()),
    }
    metrics.update(position_diagnostics(test_pos, test_real, test_start))
    metrics.update(position_diagnostics(val_pos, val_real, val_start, suffix=" val"))

    # (overnight, intraday) when test_start is known, otherwise (even, odd).
    first_half, second_half = (x.cpu().numpy() for x in half_day_split(test_pnl, test_start))
    return (
        pd.DataFrame([metrics]),
        test_pair.detach().cpu().numpy(),
        first_half,
        second_half,
    )


__all__ = ["Evaluation2LSTM"]
