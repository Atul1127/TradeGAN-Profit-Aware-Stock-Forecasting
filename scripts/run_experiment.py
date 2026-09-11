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


def _research_comparison(gan: pd.DataFrame, lstm: pd.DataFrame) -> pd.DataFrame:
    """Select the GAN objective by validation Sharpe and compare baselines."""
    gan = gan.copy()
    lstm = lstm.copy()

    fin_candidates = gan[gan["type"] != "ForGAN"]
    if fin_candidates.empty:
        raise ValueError("No FinGAN objective rows were returned")

    selected_fin = fin_candidates.loc[fin_candidates["SR_w scaled val"].idxmax()]
    forg = gan[gan["type"] == "ForGAN"]
    best_lstm = lstm.loc[lstm["SR_m scaled val"].idxmax()]

    baselines = []
    if not forg.empty:
        baselines.append(("ForGAN", forg.iloc[0]))
    baselines.append((f"LSTM:{best_lstm['type']}", best_lstm))

    rows = []
    for baseline_name, baseline in baselines:
        candidate_pnl = float(selected_fin["PnL_w"])
        baseline_pnl = float(baseline.get("PnL_w", baseline.get("PnL_m test")))
        candidate_mse = float(selected_fin["MSE"])
        baseline_mse = float(baseline["MSE"])
        candidate_std = float(selected_fin["PnL STD"])
        baseline_std = float(baseline["PnL STD"])

        rows.append(
            {
                "selected_model": "GAN",
                "selected_objective": selected_fin["type"],
                "baseline": baseline_name,
                "test_pnl": candidate_pnl,
                "baseline_pnl": baseline_pnl,
                "pnl_fold_change": fold_change(candidate_pnl, baseline_pnl),
                "test_scaled_sharpe": float(selected_fin["SR_w scaled"]),
                "baseline_scaled_sharpe": float(
                    baseline.get("SR_w scaled", baseline.get("SR_m scaled test"))
                ),
                "directional_accuracy": float(selected_fin["Directional Accuracy"]),
                "baseline_directional_accuracy": float(baseline["Directional Accuracy"]),
                "test_mse": candidate_mse,
                "baseline_mse": baseline_mse,
                "mse_reduction_pct": percent_reduction(candidate_mse, baseline_mse),
                "test_pnl_std": candidate_std,
                "baseline_pnl_std": baseline_std,
                "pnl_std_reduction_pct": percent_reduction(candidate_std, baseline_std),
            }
        )

    result = pd.DataFrame(rows)
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
