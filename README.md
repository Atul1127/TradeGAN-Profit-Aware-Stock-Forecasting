# TradeGAN

**A reproduction/application of the Fin-GAN methodology of Vuletić & Cont (2023), applied to half-day TCS excess log-returns against the Nifty IT benchmark (`^CNXIT`).**

> ### Attribution
> The Fin-GAN methodology, model name, ForGAN-based architecture, and economics-driven generator objectives originate with Milena Vuletić and Rama Cont, *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks* (2023). This repository is a reproduction/application and is not the original implementation.

## What it does

TradeGAN uses an LSTM-based conditional GAN to forecast **half-day excess log-returns** for TCS relative to the Nifty IT index.

The central idea is to optimize the generator for both forecast quality and trading performance rather than relying only on conventional prediction error.

| Objective term | What it rewards |
| --- | --- |
| `MSE` | Forecast accuracy |
| `PnL` | Profit from trading the predicted sign |
| `SR` | Sharpe ratio of the trading strategy |
| `STD` | Lower PnL volatility |
| `BCE` | Adversarial realism |

Trading-sign objectives use a differentiable `tanh` surrogate so PnL- and Sharpe-based terms can contribute gradients during training.

## Results

One completed TCS run evaluated multiple GAN objectives on the same held-out test region. The consolidated experiment metrics are preserved under `results/metrics/`.

### GAN objective comparison

| Objective | Test PnL | Test Sharpe | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: | ---: |
| `MSE` | −4.25 | −0.96 | 0.00966 | 0.00767 |
| `BCE` | −4.19 | −0.94 | 0.00965 | 0.00767 |
| `PnL MSE` | 0.93 | 0.82 | **0.00648** | 0.00467 |
| `PnL MSE STD` | 3.22 | 1.31 | **0.00648** | 0.00467 |
| `PnL MSE SR` | 3.35 | 0.85 | 0.00685 | 0.00513 |
| `PnL` | 1.35 | 1.15 | 0.00653 | 0.00465 |
| `PnL SR` | 3.40 | 0.76 | 0.00743 | 0.00576 |
| `PnL STD` | 8.93 | 1.67 | 0.12479 | 0.12292 |
| `SR` | 8.93 | 1.67 | 0.47092 | 0.47001 |
| **`SR MSE`** | **9.86** | **2.00** | 0.10053 | 0.08667 |

### LSTM baseline

The strongest observed LSTM configuration in the consolidated results was:

| Objective | Test PnL | Test Sharpe | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: | ---: |
| **`STD`** | **12.31** | **2.36** | **0.00633** | **0.00452** |

### What the results show

The experiment demonstrates the intended trade-off between **forecasting accuracy and trading performance**.

The error-focused `MSE` and `BCE` objectives produced negative test PnL and negative Sharpe in this run. Adding trading-oriented terms produced positive trading results, with `SR MSE` reaching the strongest observed GAN Sharpe (**2.00**) and PnL (**9.86**) among the consolidated GAN rows.

At the same time, the `SR MSE` model has substantially worse forecasting error than the best error-oriented configurations. The `PnL MSE` / `PnL MSE STD` family provides a stronger compromise between forecast error and trading performance, while the LSTM `STD` baseline achieved the lowest RMSE/MAE and the highest Sharpe among the specific configurations reported here.

**The key point:** a model that forecasts best is not necessarily the model that trades best. The objective function materially changes that trade-off.

### Experimental caveat

These are **run-specific observations**, not universal benchmarks. They depend on the data snapshot, train/validation/test split, random seed, model configuration, training duration, and evaluation convention.

A later research-comparison run also produced exploratory metrics such as **15.1996 test PnL**, **6.4154 scaled Sharpe**, and reported PnL fold changes versus baseline models. Because the generated comparison output contained inconsistent metric alignment across rows, those numbers are **not presented as validated benchmark claims** here.

The repository also does **not** claim a `3.18` Sharpe ratio, `85%` directional accuracy, or `2.88×` PnL improvement as verified results.

## Method

TradeGAN forecasts excess returns rather than raw prices:

1. Load adjusted TCS and benchmark prices.
2. Construct interleaved half-day returns and excess returns.
3. Split the observations into training, validation, and test regions.
4. Train conditional GAN and LSTM models.
5. Optimize adversarial and economics-aware objectives.
6. Evaluate forecasting and trading metrics on held-out data.

The historical implementation uses a differentiable `tanh` surrogate for sign-based trading objectives so PnL and Sharpe terms can participate in gradient-based optimization.

## Known limitations

1. Results come from a limited experimental setting and should not be interpreted as statistical evidence of generalization.
2. Different objectives optimize different goals, so “best” depends on the metric being evaluated.
3. Trading metrics and forecasting metrics must be compared under the same test window and conventions.
4. Directional accuracy is not used as a headline result in the consolidated experiment table.
5. The exploratory research-comparison output requires further validation before its metrics should be used as benchmark claims.

## Repository layout

```text
.
├── docs/
│   ├── methodology.md
│   ├── reproduction.md
│   └── TradeGAN.pdf
├── results/
│   ├── checkpoints/
│   ├── figures/
│   └── metrics/
├── scripts/
│   ├── download_data.py
│   └── run_experiment.py
├── src/
│   └── tradegan/
│       ├── data/
│       ├── evaluation/
│       ├── experiments/
│       ├── models/
│       ├── objectives/
│       ├── training/
│       ├── utils/
│       ├── fixed_experiment.py
│       ├── lstm_model.py
│       └── legacy.py
├── tests/
├── stocks-etfs-list.csv
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Python **3.10+** is supported.

```bash
python -m pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
python -m pip install -e .
```

## Data

`stocks-etfs-list.csv` stores ticker-to-benchmark metadata. For the documented experiment, TCS is evaluated against the Nifty IT index.

Download market data with:

```bash
python scripts/download_data.py
```

Market data is treated as local runtime data rather than a committed repository artifact.

## Run

### Smoke test

```bash
python scripts/run_experiment.py --ticker TCS --gan-epochs 1 --lstm-epochs 1 --gradient-epochs 1 --cpu
```

### Full experiment

```bash
python scripts/run_experiment.py --ticker TCS --gan-epochs 100 --lstm-epochs 500 --gradient-epochs 100 --cpu
```

The documented consolidated results were obtained from a completed TCS run using **100 GAN epochs, 500 LSTM epochs, and 10 gradient-calibration epochs**.

## Tests

Run:

```bash
pytest -q
```

The expanded local test suite contains **14 tests** covering objectives, model behavior, training updates, evaluation, validation, metrics, and checkpoint round-trips.

## Outputs

| Path | Contents |
| --- | --- |
| `results/metrics/` | CSV metrics, PnL series, and experiment summaries |
| `results/figures/` | PnL, return, and distribution plots |
| `results/checkpoints/` | Trained model checkpoints |
| `docs/TradeGAN.pdf` | Preserved research write-up |

## Reproducibility

For meaningful comparisons, record the ticker and benchmark, data window, train/validation/test split, random seed, objective, model dimensions, learning rates, epoch counts, gradient-calibration iterations, device, and evaluation convention.

See [`docs/reproduction.md`](docs/reproduction.md) for the reproduction workflow.

## Reference

Vuletić, M. and Cont, R. (2023). *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks.*

Please cite the original research for the methodology and distinguish reproduced results from newly generated experiments.
