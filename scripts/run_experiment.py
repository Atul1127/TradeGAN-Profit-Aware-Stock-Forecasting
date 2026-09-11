"""Run TradeGAN experiments and summarize research metrics."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradegan.fixed_experiment import run_gan_experiment, run_lstm_experiment
from tradegan.utils.trading_metrics import fold_change, percent_reduction


def _set_seed(seed: int) -> None:
    """Set the random seeds used by the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _finite_float(row: pd.Series, column: str) -> float:
    """Read a numeric metric and fail loudly when the value is missing or non-finite."""
    value = float(row[column])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite metric in comparison column: {column}")
    return value


def _research_comparison(gan: pd.DataFrame, lstm: pd.DataFrame) -> pd.DataFrame:
    """Select models using validation Sharpe, then compare on the untouched test rows."""
    gan = gan.copy()
    lstm = lstm.copy()

    if gan.empty or lstm.empty:
        raise ValueError("GAN and LSTM result tables must both be non-empty")
    if "ticker" not in gan or "ticker" not in lstm:
        raise ValueError("Result tables must contain ticker metadata")

    tickers = set(gan["ticker"].astype(str)) | set(lstm["ticker"].astype(str))
    if len(tickers) != 1:
        raise ValueError(f"Comparison mixes tickers: {sorted(tickers)}")

    required_gan = {
        "type",
        "SR_w scaled val",
        "SR_w scaled",
        "PnL_w",
        "MSE",
        "PnL STD",
        "Directional Accuracy",
        "MSE val",
        "PnL_w val",
        "PnL STD val",
        "Directional Accuracy val",
    }
    required_lstm = {
        "type",
        "SR_m scaled val",
        "SR_m scaled test",
        "PnL_m test",
        "MSE",
        "PnL STD",
        "Directional Accuracy",
        "MSE val",
        "PnL_m val",
        "PnL STD val",
        "Directional Accuracy val",
    }
    missing_gan = sorted(required_gan - set(gan.columns))
    missing_lstm = sorted(required_lstm - set(lstm.columns))
    if missing_gan:
        raise ValueError(f"GAN result table is missing columns: {missing_gan}")
    if missing_lstm:
        raise ValueError(f"LSTM result table is missing columns: {missing_lstm}")

    fin_candidates = gan[gan["type"] != "ForGAN"]
    forg = gan[gan["type"] == "ForGAN"]
    if fin_candidates.empty:
        raise ValueError("No FinGAN objective rows were returned")
    if forg.empty:
        raise ValueError("ForGAN baseline row was not returned")

    # Model selection happens on validation Sharpe only.
    selected_fin = fin_candidates.loc[fin_candidates["SR_w scaled val"].idxmax()]
    selected_lstm = lstm.loc[lstm["SR_m scaled val"].idxmax()]
    baseline_forg = forg.iloc[0]

    # The selected rows themselves are the source of every test metric below.
    candidate_pnl = _finite_float(selected_fin, "PnL_w")
    candidate_sharpe = _finite_float(selected_fin, "SR_w scaled")
    candidate_mse = _finite_float(selected_fin, "MSE")
    candidate_std = _finite_float(selected_fin, "PnL STD")
    candidate_acc = _finite_float(selected_fin, "Directional Accuracy")

    if not 0.0 <= candidate_acc <= 1.0:
        raise ValueError("Directional accuracy must be between 0 and 1")

    comparisons = [
        ("ForGAN", baseline_forg, "PnL_w", "SR_w scaled"),
        (f"LSTM:{selected_lstm['type']}", selected_lstm, "PnL_m test", "SR_m scaled test"),
    ]

    rows: list[dict] = []
    for baseline_name, baseline, pnl_column, sharpe_column in comparisons:
        baseline_pnl = _finite_float(baseline, pnl_column)
        baseline_sharpe = _finite_float(baseline, sharpe_column)
        baseline_mse = _finite_float(baseline, "MSE")
        baseline_std = _finite_float(baseline, "PnL STD")
        baseline_acc = _finite_float(baseline, "Directional Accuracy")
        if not 0.0 <= baseline_acc <= 1.0:
            raise ValueError(f"Directional accuracy must be between 0 and 1 for {baseline_name}")

        rows.append(
            {
                "ticker": str(selected_fin["ticker"]),
                "selection_metric": "validation_scaled_sharpe",
                "selected_model": "GAN",
                "selected_objective": str(selected_fin["type"]),
                "baseline": baseline_name,
                "test_pnl": candidate_pnl,
                "baseline_test_pnl": baseline_pnl,
                "pnl_fold_change": fold_change(candidate_pnl, baseline_pnl),
                "test_scaled_sharpe": candidate_sharpe,
                "baseline_test_scaled_sharpe": baseline_sharpe,
                "directional_accuracy": candidate_acc,
                "baseline_directional_accuracy": baseline_acc,
                "test_mse": candidate_mse,
                "baseline_test_mse": baseline_mse,
                "mse_reduction_pct": percent_reduction(candidate_mse, baseline_mse),
                "test_pnl_std": candidate_std,
                "baseline_test_pnl_std": baseline_std,
                "pnl_std_reduction_pct": percent_reduction(candidate_std, baseline_std),
            }
        )

    result = pd.DataFrame(rows)
    if result["selected_objective"].nunique() != 1:
        raise AssertionError("Comparison rows disagree on the selected GAN objective")
    if result["test_pnl"].nunique() != 1 or result["test_scaled_sharpe"].nunique() != 1:
        raise AssertionError("Comparison rows disagree on the selected GAN test metrics")

    output = ROOT / "results" / "metrics" / "research_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print("\n=== Research comparison (selected by validation Sharpe) ===")
    print(result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TradeGAN experiments for a ticker.")
    parser.add_argument("--ticker", default="TCS")
    parser.add_argument("--gan-epochs", type=int, default=100)
    parser.add_argument("--lstm-epochs", type=int, default=500)
    parser.add_argument(
        "--gradient-epochs",
        type=int,
        default=25,
        help="Number of gradient-balance iterations used before each objective.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    args = parser.parse_args()

    _set_seed(args.seed)

    gan_results = run_gan_experiment(
        ticker=args.ticker,
        root=ROOT,
        gan_epochs=args.gan_epochs,
        gradient_epochs=args.gradient_epochs,
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
    )

    lstm_results = run_lstm_experiment(
        ticker=args.ticker,
        root=ROOT,
        lstm_epochs=args.lstm_epochs,
        gradient_epochs=args.gradient_epochs,
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
    )

    _research_comparison(gan_results, lstm_results)


if __name__ == "__main__":
    main()
