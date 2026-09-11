"""Run TradeGAN experiments with isolated, corrected objective training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradegan.fixed_experiment import run_gan_experiment, run_lstm_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TradeGAN experiments for a ticker.")
    parser.add_argument("--ticker", default="TCS")
    parser.add_argument("--gan-epochs", type=int, default=100)
    parser.add_argument("--lstm-epochs", type=int, default=500)
    parser.add_argument(
        "--gradient-epochs",
        type=int,
        default=100,
        help="Number of gradient-balance iterations used before each objective.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    args = parser.parse_args()

    run_gan_experiment(
        ticker=args.ticker,
        root=ROOT,
        gan_epochs=args.gan_epochs,
        gradient_epochs=args.gradient_epochs,
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
    )
    run_lstm_experiment(
        ticker=args.ticker,
        root=ROOT,
        lstm_epochs=args.lstm_epochs,
        gradient_epochs=args.gradient_epochs,
        batch_size=args.batch_size,
        use_gpu=not args.cpu,
    )


if __name__ == "__main__":
    main()
