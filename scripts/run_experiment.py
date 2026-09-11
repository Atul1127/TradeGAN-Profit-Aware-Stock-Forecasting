"""Run the archived TradeGAN experiment with project-relative paths."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradegan import legacy


def run(ticker: str, gan_epochs: int = 100, lstm_epochs: int = 500) -> None:
    data_dir = ROOT / "data"
    metadata = ROOT / "stocks-etfs-list.csv"
    results_dir = ROOT / "results"
    checkpoints_dir = results_dir / "checkpoints"
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    legacy.FinGAN_combos(
        ticker,
        str(results_dir) + "/",
        str(checkpoints_dir) + "/",
        str(figures_dir) + "/",
        str(data_dir) + "/",
        str(metadata),
        vl_later=True,
        lrg=0.0001,
        lrd=0.0001,
        n_epochs=gan_epochs,
        ngrad=100,
        h=1,
        l=10,
        pred=1,
        ngpu=1,
        tanh_coeff=100,
        tr=0.8,
        vl=0.1,
        z_dim=8,
        hid_d=8,
        hid_g=8,
        checkpoint_epoch=20,
        batch_size=100,
        diter=1,
        plot=False,
    )

    legacy.LSTM_combos(
        ticker,
        str(results_dir) + "/",
        str(checkpoints_dir) + "/",
        str(figures_dir) + "/",
        str(data_dir) + "/",
        str(metadata),
        vl_later=True,
        lrg=0.0001,
        lrd=0.0001,
        n_epochs=lstm_epochs,
        ngrad=100,
        h=1,
        l=10,
        pred=1,
        ngpu=1,
        tanh_coeff=100,
        tr=0.8,
        vl=0.1,
        z_dim=32,
        hid_d=64,
        hid_g=1,
        checkpoint_epoch=20,
        batch_size=100,
        diter=1,
        plot=False,
        freq=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the archived TradeGAN experiment.")
    parser.add_argument("--ticker", default="TCS")
    parser.add_argument("--gan-epochs", type=int, default=100)
    parser.add_argument("--lstm-epochs", type=int, default=500)
    args = parser.parse_args()
    run(args.ticker, args.gan_epochs, args.lstm_epochs)
