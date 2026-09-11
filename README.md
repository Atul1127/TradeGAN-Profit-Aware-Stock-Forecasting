# TradeGAN — Profit-Aware Stock Forecasting

A reproduction of the Fin-GAN methodology of Vuletić & Cont (2023), applied to half-day TCS excess returns against the Nifty IT sector index.

## Repository layout

```text
├── configs/               experiment configuration
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
│   └── legacy.py          original monolithic implementation
├── tests/                 automated tests
└── stocks-etfs-list.csv   ticker/benchmark metadata
```

## Method

TradeGAN uses a conditional GAN with LSTM generator and discriminator components following the ForGAN architecture. The model forecasts half-day excess log-returns for TCS relative to the Nifty IT benchmark. The generator objective can combine adversarial BCE with MSE, PnL, Sharpe-ratio, and volatility terms.

The historical implementation uses a differentiable `tanh` surrogate for trading-sign objectives so the economics-aware terms remain trainable.

## Attribution

The Fin-GAN methodology, model name, ForGAN-based architecture, and economics-driven generator objectives originate with Milena Vuletić and Rama Cont, *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks* (2023). This repository is a reproduction/application rather than the original method.

## Data

`stocks-etfs-list.csv` contains ticker-to-benchmark metadata. Historical prices can be downloaded into `data/`. Downloaded market data is local runtime data and should not be committed.

## Results

Historical experiment outputs are preserved under `results/` and separated from source code. The original PDF write-up is preserved under `docs/TradeGAN.pdf`.

## Running

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then download data and run the experiment script from the repository root. The scripts use project-relative paths rather than machine-specific absolute paths.

## Reproducibility notes

The archived research implementation has known limitations, including the adaptive gradient-norm weighting calculation and the absence of a realized directional-accuracy metric. These should be corrected in a separate correctness pass so historical results are not silently changed during structural cleanup.
