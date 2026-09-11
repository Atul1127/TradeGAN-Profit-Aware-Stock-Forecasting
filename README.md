# TradeGAN — Profit-Aware Stock Forecasting

A reproduction of the Fin-GAN methodology of Vuletić & Cont (2023), applied to half-day TCS excess returns against the Nifty IT sector index.

## Project structure

```text
TradeGAN-Profit-Aware-Stock-Forecasting/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
├── data/
├── src/
│   ├── tradegan/
│   │   ├── data/
│   │   ├── models/
│   │   ├── objectives/
│   │   ├── training/
│   │   ├── evaluation/
│   │   ├── experiments/
│   │   └── utils/
│   ├── TradeGAN.py        # legacy monolithic implementation
│   ├── app.py              # legacy experiment driver
│   └── data_maker.py       # legacy data downloader
├── scripts/
├── tests/
├── docs/
├── notebooks/
└── results/
```

## Method

TradeGAN uses a conditional GAN with LSTM generator and discriminator components following the ForGAN architecture. The model forecasts half-day excess log-returns for TCS relative to the Nifty IT benchmark. The generator objective can combine adversarial BCE with MSE, PnL, Sharpe-ratio, and volatility terms.

Trading uses a differentiable `tanh` surrogate for the forecast sign so the economics-aware objectives remain trainable.

## Attribution

The Fin-GAN methodology, model name, ForGAN-based architecture, and economics-driven generator objectives originate with Milena Vuletić and Rama Cont, *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks* (2023). This repository is a reproduction/application rather than the original method.

## Data

`stocks-etfs-list.csv` contains ticker-to-benchmark metadata. Historical prices can be downloaded into `data/` using the data downloader. Downloaded market data is kept locally and should not be committed.

## Results

The repository contains prior TCS experiment outputs under `results/`. These are retained as research artifacts and are separate from the source code.

## Reproducibility

The original implementation currently has several research limitations that should be considered when interpreting the archived results:

- the adaptive gradient-norm weighting implementation needs correctness review;
- directional accuracy is not reported as a realized-sign hit rate;
- the original experiment driver used machine-specific absolute paths;
- dependency versions were not originally pinned;
- tests and continuous integration were not part of the original implementation.

These issues are tracked separately from the structural cleanup so that refactoring does not silently change the historical experiment.

## Reference

Vuletić, M. and Cont, R. (2023). *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks*.
