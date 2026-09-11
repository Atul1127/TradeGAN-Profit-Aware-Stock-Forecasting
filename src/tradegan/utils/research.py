"""Research comparison helpers for held-out experiment results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .trading_metrics import fold_change, percent_reduction


def research_comparison(gan: pd.DataFrame, lstm: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Select a GAN objective using validation Sharpe and compare test results.

    Model selection is performed only from validation metrics. Test metrics are
    copied from the selected rows without recomputation or row mixing.
    """
    gan = gan.copy()
    lstm = lstm.copy()

    for frame_name, frame in (("GAN", gan), ("LSTM", lstm)):
        if frame.empty:
            raise ValueError(f"{frame_name} results must not be empty")
        if "ticker" not in frame:
            raise ValueError(f"{frame_name} results must contain ticker")

    gan_tickers = set(gan["ticker"].astype(str))
    lstm_tickers = set(lstm["ticker"].astype(str))
    if len(gan_tickers) != 1 or gan_tickers != lstm_tickers:
        raise ValueError("GAN and LSTM results must refer to the same single ticker")

    required_gan = {
        "type", "SR_w scaled val", "SR_w scaled", "PnL_w", "MSE", "PnL STD",
        "Directional Accuracy"
    }
    required_lstm = {
        "type", "SR_m scaled val", "SR_m scaled test", "PnL_m test", "MSE",
        "PnL STD", "Directional Accuracy"
    }
    missing_gan = required_gan.difference(gan.columns)
    missing_lstm = required_lstm.difference(lstm.columns)
    if missing_gan:
        raise ValueError(f"GAN results missing columns: {sorted(missing_gan)}")
    if missing_lstm:
        raise ValueError(f"LSTM results missing columns: {sorted(missing_lstm)}")

    fin_candidates = gan[gan["type"] != "ForGAN"]
    if fin_candidates.empty:
        raise ValueError("No FinGAN objective rows were returned")

    selected_fin = fin_candidates.loc[fin_candidates["SR_w scaled val"].idxmax()]
    forg = gan[gan["type"] == "ForGAN"]
    best_lstm = lstm.loc[lstm["SR_m scaled val"].idxmax()]

    baselines: list[tuple[str, pd.Series]] = []
    if not forg.empty:
        baselines.append(("ForGAN", forg.iloc[0]))
    baselines.append((f"LSTM:{best_lstm['type']}", best_lstm))

    rows = []
    for baseline_name, baseline in baselines:
        candidate_pnl = float(selected_fin["PnL_w"])
        baseline_pnl = float(
            baseline["PnL_w"] if baseline_name == "ForGAN" else baseline["PnL_m test"]
        )
        candidate_mse = float(selected_fin["MSE"])
        baseline_mse = float(baseline["MSE"])
        candidate_std = float(selected_fin["PnL STD"])
        baseline_std = float(baseline["PnL STD"])
        candidate_da = float(selected_fin["Directional Accuracy"])
        baseline_da = float(baseline["Directional Accuracy"])

        values = {
            "selected_model": "GAN",
            "selected_objective": selected_fin["type"],
            "baseline": baseline_name,
            "test_pnl": candidate_pnl,
            "baseline_test_pnl": baseline_pnl,
            "pnl_fold_change": fold_change(candidate_pnl, baseline_pnl),
            "test_scaled_sharpe": float(selected_fin["SR_w scaled"]),
            "baseline_scaled_sharpe": float(
                baseline["SR_w scaled"] if baseline_name == "ForGAN" else baseline["SR_m scaled test"]
            ),
            "directional_accuracy": candidate_da,
            "baseline_directional_accuracy": baseline_da,
            "test_mse": candidate_mse,
            "baseline_mse": baseline_mse,
            "mse_reduction_pct": percent_reduction(candidate_mse, baseline_mse),
            "test_pnl_std": candidate_std,
            "baseline_pnl_std": baseline_std,
            "pnl_std_reduction_pct": percent_reduction(candidate_std, baseline_std),
        }
        if not all(pd.notna(value) for value in values.values() if not isinstance(value, str)):
            raise ValueError("Research comparison contains non-finite numeric values")
        if not 0.0 <= values["directional_accuracy"] <= 1.0:
            raise ValueError("Directional accuracy must be between 0 and 1")
        if not 0.0 <= values["baseline_directional_accuracy"] <= 1.0:
            raise ValueError("Baseline directional accuracy must be between 0 and 1")
        rows.append(values)

    result = pd.DataFrame(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


__all__ = ["research_comparison"]
