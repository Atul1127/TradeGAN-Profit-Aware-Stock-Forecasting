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
│   ├── TradeGAN.py
│   ├── app.py
│   └── data_maker.py
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

`stocks-etfs-list.csv` contains ticker-to-benchmark metadata. Historical prices can be downloaded into `data/`. Keep downloaded market data local rather than committing it.

## Results

Historical TCS experiment outputs are retained under `results/` as research artifacts and are kept separate from source code.

## Reproducibility notes

The archived experiment should be interpreted with the limitations documented in the research code: adaptive gradient-norm weighting needs correctness review, directional accuracy was not originally reported as a realized-sign hit rate, the original driver used machine-specific absolute paths, and dependencies/tests were not originally formalized.

## Reference

Vuletić, M. and Cont, R. (2023). *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks*.
