# TradeGAN

**A reproduction/application of the Fin-GAN methodology of Vuletić & Cont (2023), applied to half-day TCS excess log-returns against the Nifty IT benchmark (`^CNXIT`).**

> ### Attribution
> The Fin-GAN methodology, model name, ForGAN-based architecture, and economics-driven generator objectives originate with Milena Vuletić and Rama Cont, *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks* (2023). This repository is a reproduction/application and is not the original implementation.

## What it does

TradeGAN uses an LSTM-based conditional GAN to forecast **half-day excess log-returns** for TCS relative to the Nifty IT index.

The central idea is to optimize the generator for both forecast quality and trading performance rather than relying only on conventional prediction error.

## Results

Two result views are preserved: the consolidated objective sweep and a controlled research-comparison run. All reported metrics below are **run-specific observations**, not universal benchmarks.

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

### Controlled research comparison

A controlled TCS run used **100 GAN epochs, 500 LSTM epochs, 25 gradient-calibration iterations, seed 42, and CPU execution**. The GAN objective was selected using **validation scaled Sharpe only**, then evaluated on the held-out test set.

The selected objective was **`SR MSE`**.

| Comparison | TradeGAN / selected GAN | Baseline | Relative result |
| --- | ---: | ---: | ---: |
| Test PnL vs ForGAN | **15.1996** | 2.3692 | **6.415× PnL** |
| Test scaled Sharpe vs ForGAN | **2.7278** | 0.4128 | 6.61× baseline Sharpe |
| Directional accuracy vs ForGAN | **57.90%** | 50.17% | +7.73 pp |
| Test MSE vs ForGAN | **0.000040** | 0.000044 | **8.97% lower** |
| PnL STD vs ForGAN | **88.4562** | 91.1123 | **2.92% lower** |
| Test PnL vs LSTM:SR | **15.1996** | 5.9086 | **2.572× PnL** |
| Test scaled Sharpe vs LSTM:SR | **2.7278** | 1.0155 | 2.69× baseline Sharpe |
| Directional accuracy vs LSTM:SR | **57.90%** | 51.37% | +6.54 pp |
| Test MSE vs LSTM:SR | **0.000040** | 0.000040 | **0.55% higher** |
| PnL STD vs LSTM:SR | **88.4562** | 92.3649 | **4.23% lower** |

### Interpretation

The experiments support the main motivation of the project: **forecasting accuracy and trading performance are not the same objective**.

In the consolidated sweep, error-focused `MSE` and `BCE` produced negative test PnL and negative Sharpe, while several economics-aware objectives produced positive trading results. `SR MSE` gave the strongest consolidated GAN Sharpe (**2.00**) and PnL (**9.86**) in that run.

In the controlled comparison, validation-based selection chose `SR MSE`. On the held-out test region it produced **15.1996 PnL**, **2.7278 scaled Sharpe**, **57.90% directional accuracy**, and a **6.415× PnL multiple versus ForGAN**. Against the selected LSTM `SR` baseline, the same GAN produced **2.572× the PnL** and a higher scaled Sharpe.

The selected GAN's MSE was **8.97% lower than ForGAN** but **0.55% higher than LSTM:SR**, while its PnL standard deviation was lower than both baselines. These are single-run observations, not evidence that the strategy generalizes to other assets, periods, or markets.

### Important interpretation rule

The **6.415× figure is PnL fold change, not Sharpe**. The corresponding selected GAN test scaled Sharpe is **2.7278**. Directional accuracy is **57.90%**; the value `88.4562` is PnL standard deviation, not directional accuracy.

## Method

TradeGAN forecasts excess returns rather than raw prices:

1. Load adjusted TCS and benchmark prices.
2. Construct interleaved half-day returns and excess returns.
3. Split the observations into training, validation, and test regions.
4. Train conditional GAN and LSTM models.
5. Optimize adversarial and economics-aware objectives.
6. Select the controlled-run GAN objective using validation Sharpe and evaluate on the held-out test region.
7. Report forecasting and trading metrics under the same evaluation convention.

The historical implementation uses a differentiable `tanh` surrogate for sign-based trading objectives so PnL and Sharpe terms can participate in gradient-based optimization.

## Known limitations

1. Results come from a limited experimental setting and should not be interpreted as statistical evidence of generalization.
2. Different objectives optimize different goals, so “best” depends on the metric being evaluated.
3. Comparisons must use the same test window, trading convention, and scaling convention.
4. The controlled comparison uses one seed and one asset/benchmark pair; multi-seed and multi-asset validation is still needed.
5. Sharpe is reported using this repository's paired-PnL and scaling convention and should not be compared directly with results using different conventions.

## Repository layout

```text
.
├── docs/
│   ├── methodology.md
│   ├── reproduction.md
│   └── TradeGAN.pdf
├── results/
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
│       └── lstm_model.py
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

Or:

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
python scripts/run_experiment.py --ticker TCS --gan-epochs 1 --lstm-epochs 1 --gradient-epochs 1 --seed 42 --cpu
```

### Controlled research experiment

```bash
python scripts/run_experiment.py --ticker TCS --gan-epochs 100 --lstm-epochs 500 --gradient-epochs 25 --seed 42 --cpu
```

This run selects the GAN objective from validation scaled Sharpe and writes `results/metrics/research_comparison.csv`.

## Tests

Run:

```bash
pytest -q
```

The expanded local test suite currently contains **16 tests** covering objective formulas, model behavior, training updates, evaluation, data validation, research metrics, comparison alignment, and checkpoint round-trips.

## Outputs

| Path | Contents |
| --- | --- |
| `results/metrics/` | CSV metrics, PnL series, and experiment summaries |
| `results/figures/` | PnL, return, and distribution plots |
| `docs/TradeGAN.pdf` | Preserved research write-up |

## Reproducibility

For meaningful comparisons, record the ticker and benchmark, data window, train/validation/test split, random seed, objective, model dimensions, learning rates, epoch counts, gradient-calibration iterations, device, and evaluation convention.

See [`docs/reproduction.md`](docs/reproduction.md) for the reproduction workflow.

## Reference

Vuletić, M. and Cont, R. (2023). *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks.*

Please cite the original research for the methodology and distinguish reproduced results from newly generated experiments.
