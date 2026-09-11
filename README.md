# TradeGAN — Profit-Aware Stock Forecasting

A reproduction of the Fin-GAN methodology of Vuletić & Cont (2023), applied to half-day TCS excess returns against the Nifty IT sector index.

## Repository layout

```text
├── configs/               reference experiment configuration
├── data/                  local/downloaded market data
├── docs/                  methodology and reproduction notes
├── experiments/           experiment documentation
├── results/               archived metrics, figures, checkpoints
├── scripts/               runnable entry points
├── src/tradegan/          Python package
│   ├── data/              data preparation
│   ├── models/            model components
│   ├── objectives/        trading-aware objectives
│   ├── training/          training loops
│   ├── evaluation/        metrics and backtests
│   ├── experiments/       experiment orchestration
│   ├── utils/             shared utilities
│   └── legacy.py          compatibility re-exports for old imports
├── tests/                 automated tests
└── stocks-etfs-list.csv   ticker/benchmark metadata
```

## Method

TradeGAN uses a conditional GAN with LSTM generator and discriminator components following the ForGAN architecture. The model forecasts half-day excess log-returns for TCS relative to the Nifty IT benchmark. The generator objective can combine adversarial BCE with MSE, PnL, Sharpe-ratio, and volatility terms.

The historical implementation uses a differentiable `tanh` surrogate for trading-sign objectives so the economics-aware terms remain trainable.

## Attribution

The Fin-GAN methodology, model name, ForGAN-based architecture, and economics-driven generator objectives originate with Milena Vuletić and Rama Cont, *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks* (2023). This repository is a reproduction/application rather than the original method.

## Data

`stocks-etfs-list.csv` contains ticker-to-benchmark metadata. Historical prices can be downloaded into `data/` with `scripts/download_data.py`. The downloader writes the adjusted `AdjOpen`/`AdjClose` schema expected by the return-construction code. Downloaded market data is local runtime data and should not be committed.

## Results

Historical experiment outputs are preserved under `results/` and separated from source code. The original PDF write-up is preserved under `docs/TradeGAN.pdf`.

## Running

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download data from the repository root:

```bash
python scripts/download_data.py
```

Run a smoke test before a full experiment:

```bash
pytest -q
python scripts/run_experiment.py --ticker TCS --gan-epochs 1 --lstm-epochs 1 --gradient-epochs 1 --cpu
```

For longer runs, increase the epoch arguments in `scripts/run_experiment.py` or use an equivalent command-line invocation. `configs/default.yaml` documents the intended default hyperparameters; the current runner is CLI-driven and does not automatically load that YAML file.

## Reproducibility notes

The repository now has a separated training/evaluation implementation plus a compatibility shim for historical imports. Historical checkpoints and result files are retained so structural cleanup does not overwrite archived artifacts. The one-epoch command above is a pipeline smoke test, not a performance benchmark.
