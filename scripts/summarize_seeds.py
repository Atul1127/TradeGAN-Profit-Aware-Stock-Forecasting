"""Summarise TradeGAN runs over several seeds.

Run the experiment once per seed, each into its own folder:

    for s in 1 2 3 4 5; do
        python scripts/run_experiment.py --ticker TCS --gan-epochs 100 --lstm-epochs 500 \
            --gradient-epochs 25 --seed $s --cpu --results-dir results/seeds/seed_$s
    done
    python scripts/summarize_seeds.py

The script reports two views:

1. The validation-selected GAN per seed. This is the honest headline: the
   procedure (pick on validation, test once) repeated over seeds.
2. Every objective's test metrics averaged over seeds. This shows whether the
   ranking of objectives is stable or within seed-to-seed noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

GAN_METRICS = {
    "SR_w scaled": "test_sharpe",
    "PnL_w": "test_pnl",
    "Timing SR scaled": "timing_sharpe",
    "Net SR scaled": "net_sharpe",
    "PnL_w over best constant": "pnl_over_best_constant",
    "Long share": "long_share",
    "Turnover per day": "turnover_per_day",
    "Directional Accuracy": "directional_accuracy",
}


def _load_runs(runs: Path, ticker: str) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(runs.glob("seed_*")):
        path = seed_dir / "metrics" / f"{ticker}-GAN-results.csv"
        if not path.exists():
            print(f"skipping {seed_dir.name}: {path.name} not found")
            continue
        frame = pd.read_csv(path)
        frame["seed"] = seed_dir.name.removeprefix("seed_")
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No runs found under {runs}. Expected {runs}/seed_*/metrics/{ticker}-GAN-results.csv")
    return pd.concat(frames, ignore_index=True)


def _available(frame: pd.DataFrame) -> dict[str, str]:
    return {column: name for column, name in GAN_METRICS.items() if column in frame.columns}


def selected_per_seed(results: pd.DataFrame) -> pd.DataFrame:
    """Validation-selected non-ForGAN objective for each seed, with its test metrics."""
    metrics = _available(results)
    rows = []
    for seed, group in results.groupby("seed", sort=True):
        candidates = group[group["type"] != "ForGAN"]
        chosen = candidates.loc[candidates["SR_w scaled val"].idxmax()]
        row = {"seed": seed, "selected_objective": chosen["type"], "val_sharpe": chosen["SR_w scaled val"]}
        row.update({name: chosen[column] for column, name in metrics.items()})
        forgan = group[group["type"] == "ForGAN"]
        if not forgan.empty:
            row["forgan_test_sharpe"] = forgan.iloc[0]["SR_w scaled"]
        if "Always-short SR scaled" in chosen:
            row["always_short_sharpe"] = chosen["Always-short SR scaled"]
        rows.append(row)
    return pd.DataFrame(rows)


def per_objective(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each objective's test metrics over seeds."""
    metrics = _available(results)
    renamed = results.rename(columns=metrics)
    summary = renamed.groupby("type")[list(metrics.values())].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    if "pnl_over_best_constant" in metrics.values():
        beats = renamed.assign(beats=renamed["pnl_over_best_constant"] > 0).groupby("type")["beats"].mean()
        summary["share_of_seeds_beating_best_constant"] = beats
    summary["seeds"] = renamed.groupby("type")["seed"].nunique()
    return summary.sort_values("test_sharpe_mean", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=ROOT / "results" / "seeds")
    parser.add_argument("--ticker", default="TCS")
    args = parser.parse_args()
    runs = args.runs if args.runs.is_absolute() else ROOT / args.runs

    results = _load_runs(runs, args.ticker)
    selected = selected_per_seed(results)
    objectives = per_objective(results)

    selected.to_csv(runs / "selected_per_seed.csv", index=False)
    objectives.to_csv(runs / "objectives_over_seeds.csv")

    pd.set_option("display.width", 200)
    print("\n=== Validation-selected GAN, one row per seed ===")
    print(selected.round(3).to_string(index=False))
    numeric = selected.select_dtypes("number")
    print("\nmean over seeds:")
    print(numeric.mean().round(3).to_string())
    print("std over seeds:")
    print(numeric.std().round(3).to_string())
    print("\n=== Every objective, test metrics over seeds ===")
    print(objectives.round(3).to_string())
    print(f"\nWrote {runs / 'selected_per_seed.csv'} and {runs / 'objectives_over_seeds.csv'}")


if __name__ == "__main__":
    main()
