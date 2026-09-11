"""Evaluation utilities for deterministic LSTM forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _paired_pnl(pnl: torch.Tensor) -> torch.Tensor:
    usable = 2 * (pnl.numel() // 2)
    if usable == 0:
        raise ValueError("At least two observations are required for paired PnL evaluation")
    return pnl[:usable].reshape(-1, 2).sum(dim=1)


def _safe_corr(prediction: torch.Tensor, real: torch.Tensor) -> float:
    x = prediction.detach().cpu().numpy().reshape(-1)
    y = real.detach().cpu().numpy().reshape(-1)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _evaluate(model, data, lookback, hidden_dim, device):
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
    pnl = 10000 * torch.sign(prediction) * real
    paired = _paired_pnl(pnl)
    sharpe = paired.mean() / paired.std().clamp_min(torch.finfo(paired.dtype).eps)
    return prediction, real, pnl, paired, sharpe


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
):
    """Evaluate deterministic LSTM forecasts on test and validation data."""
    del freq, h, pred, hid_d, z_dim, sr_val, plotsloc, f_name, plot

    test_pred, test_real, test_pnl, test_pair, test_sr = _evaluate(
        gen, test_data, l, hid_g, device
    )
    val_pred, val_real, val_pnl, val_pair, val_sr = _evaluate(
        gen, val_data, l, hid_g, device
    )

    scale = np.sqrt(252.0)
    metrics = {
        "lrd": lrd,
        "lrg": lrg,
        "type": losstype,
        "epochs": n_epochs,
        "ticker": ticker,
        "RMSE": float(torch.sqrt(torch.mean((test_pred - test_real) ** 2)).item()),
        "MAE": float(torch.mean(torch.abs(test_pred - test_real)).item()),
        "PnL_m test": float(test_pair.mean().item()),
        "SR_m scaled test": float((test_sr * scale).item()),
        "RMSE val": float(torch.sqrt(torch.mean((val_pred - val_real) ** 2)).item()),
        "MAE val": float(torch.mean(torch.abs(val_pred - val_real)).item()),
        "PnL_m val": float(val_pair.mean().item()),
        "SR_m scaled val": float((val_sr * scale).item()),
        "Corr": _safe_corr(test_pred, test_real),
        "Corr val": _safe_corr(val_pred, val_real),
        "Pos mn": float((test_pred > 0).float().mean()),
        "Neg mn": float((test_pred < 0).float().mean()),
        "Pos mn val": float((val_pred > 0).float().mean()),
        "Neg mn val": float((val_pred < 0).float().mean()),
    }

    even = test_pnl[: 2 * (len(test_pnl) // 2) : 2].cpu().numpy()
    odd = test_pnl[1 : 2 * (len(test_pnl) // 2) : 2].cpu().numpy()
    return (
        pd.DataFrame([metrics]),
        test_pair.detach().cpu().numpy(),
        even,
        odd,
    )


__all__ = ["Evaluation2LSTM"]
