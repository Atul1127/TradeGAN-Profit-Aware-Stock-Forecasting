"""Monte-Carlo evaluation for the conditional GAN."""
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


def _evaluate(model, data, lookback, hidden_dim, latent_dim, device, mc_samples):
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
    signed_pnl = 10000 * torch.sign(mean_prediction) * real
    paired = _paired_pnl(signed_pnl)
    sharpe = paired.mean() / paired.std().clamp_min(torch.finfo(paired.dtype).eps)
    rmse = torch.sqrt(torch.mean((mean_prediction - real) ** 2))
    mae = torch.mean(torch.abs(mean_prediction - real))
    return {
        "mean": mean_prediction,
        "samples": samples,
        "real": real,
        "pnl": signed_pnl,
        "paired_pnl": paired,
        "rmse": rmse,
        "mae": mae,
        "sharpe": sharpe,
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
):
    """Evaluate one GAN on test and validation data.

    The legacy arguments are retained for API compatibility; evaluation uses
    the model, data tensors and explicit Monte-Carlo sample count only.
    """
    del freq, hid_d, h, pred, lrg, lrd, sr_val, plotsloc, f_name, plot
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    if l <= 0 or hid_g <= 0 or z_dim <= 0:
        raise ValueError("lookback, hidden_dim and latent_dim must be positive")

    test = _evaluate(gen, test_data, l, hid_g, z_dim, device, mc_samples)
    val = _evaluate(gen, val_data, l, hid_g, z_dim, device, mc_samples)

    test_real = test["real"]
    val_real = val["real"]
    test_mean = test["mean"]
    val_mean = val["mean"]

    even = test["pnl"][: 2 * (len(test["pnl"]) // 2) : 2].cpu().numpy()
    odd = test["pnl"][1 : 2 * (len(test["pnl"]) // 2) : 2].cpu().numpy()
    distribution = test["samples"][1 if mc_samples > 1 else 0].cpu().numpy()

    scale = np.sqrt(252.0)
    metrics = {
        "lrd": lrd,
        "lrg": lrg,
        "type": losstype,
        "epochs": n_epochs,
        "ticker": ticker,
        "hid_g": hid_g,
        "hid_d": hid_d,
        "RMSE": float(test["rmse"].item()),
        "MAE": float(test["mae"].item()),
        "PnL_w": float(test["paired_pnl"].mean().item()),
        "SR_w scaled": float((test["sharpe"] * scale).item()),
        "RMSE val": float(val["rmse"].item()),
        "MAE val": float(val["mae"].item()),
        "PnL_w val": float(val["paired_pnl"].mean().item()),
        "SR_w scaled val": float((val["sharpe"] * scale).item()),
        "Corr": _safe_corr(test_mean, test_real),
        "Corr val": _safe_corr(val_mean, val_real),
        "Pos mn": float((test_mean > 0).float().mean()),
        "Neg mn": float((test_mean < 0).float().mean()),
        "Pos mn val": float((val_mean > 0).float().mean()),
        "Neg mn val": float((val_mean < 0).float().mean()),
        "narrow dist": bool(test["samples"][0].std() < 0.0002),
        "narrow means dist": bool(test_mean.std() < 0.0002),
    }

    return (
        pd.DataFrame([metrics]),
        test["paired_pnl"].detach().cpu().numpy(),
        even,
        odd,
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
        val_result, val_pnl, _, _, val_mean, *_ = Evaluation2(
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
