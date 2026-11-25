# TradeGAN

**A reproduction/application of the Fin-GAN methodology of Vuletić & Cont (2023), applied to half-day TCS excess log-returns against the Nifty IT benchmark (`^CNXIT`).**

> ### Attribution
> The Fin-GAN methodology, model name, ForGAN-based architecture, and economics-driven generator objectives originate with Milena Vuletić and Rama Cont, *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks* (2023). This repository is a reproduction/application and is not the original implementation.

## What it does

TradeGAN uses a conditional GAN with LSTM layers to forecast **half-day excess log-returns** for TCS relative to the Nifty IT index.

The central idea is to optimize the generator for both forecast quality and trading performance rather than relying only on conventional prediction error.

## Results

> **Status (Nov 2026): the numbers below predate the fixes listed in [Changes since the reported results](#changes-since-the-reported-results) and need a re-run.** In particular, gradient calibration used to train every non-ForGAN model for extra epochs, so the GAN-vs-ForGAN comparison was not on equal terms. Treat every figure here as a single-run observation, not a benchmark.

Two result views are preserved: the consolidated objective sweep and a controlled research-comparison run.

### All GAN objectives — test results (consolidated sweep)

This table shows every objective's **test** metrics side by side. It is not a model-selection table: the objective to report is the one chosen on validation (`PnL SR`, validation Sharpe 2.59), which scored **0.76** on test. Picking the best row by test Sharpe would be selecting on the test set.

| Objective | Val Sharpe | Test PnL (bps/day) | Test Sharpe | Long share (test) | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MSE` | 0.01 | −4.25 | −0.96 | 84% | 0.00966 | 0.00767 |
| `BCE` (ForGAN) | −0.05 | −4.19 | −0.94 | 84% | 0.00965 | 0.00767 |
| `PnL MSE` | 1.74 | 0.93 | 0.82 | 57% | 0.00648 | 0.00467 |
| `PnL MSE STD` | 1.44 | 3.22 | 1.31 | 45% | 0.00648 | 0.00467 |
| `PnL MSE SR` | 2.31 | 3.35 | 0.85 | 65% | 0.00685 | 0.00513 |
| `PnL` | 0.28 | 1.35 | 1.15 | 64% | 0.00653 | 0.00465 |
| **`PnL SR`** (validation choice) | **2.59** | 3.40 | 0.76 | 66% | 0.00743 | 0.00576 |
| `PnL STD` | 0.68 | 8.93 | 1.67 | 0% | 0.12479 | 0.12292 |
| `SR` | 0.68 | 8.93 | 1.67 | 0% | 0.47092 | 0.47001 |
| `SR MSE` | 1.79 | 9.86 | 2.00 | 6% | 0.10053 | 0.08667 |
| *Always short (no model)* | — | *8.93* | *1.67* | *0%* | — | — |

**How to read this table.**

- `PnL STD` and `SR` predicted a negative excess return on every test and validation half-day, and their saved PnL series are byte-for-byte identical. They are the always-short strategy (short TCS, long Nifty IT), not forecasts.
- TCS lagged the index over the test year, so always-short alone made 8.93 bps a day (Sharpe 1.67). `SR MSE` was short 94% of the time and beat that by about 1 bp a day. Most of its result is the same directional bet.
- `MSE` and `BCE` were long about 84% of the time in the same year, which explains much of their loss. Their result is also mostly direction bias rather than bad timing.
- The objectives with very large RMSE (`PnL STD`, `SR`, `SR MSE`) pushed predictions far from zero. This is consistent with the `tanh(100·x)` surrogate rewarding prediction size, not just sign, at typical return sizes.

### LSTM baseline (consolidated sweep)

| Objective | Test PnL | Test Sharpe | Test RMSE | Test MAE |
| --- | ---: | ---: | ---: | ---: |
| `STD` | 12.31 | 2.36 | 0.00633 | 0.00452 |

In this sweep an LSTM trained with an economic loss beat every GAN objective on test. Again, this is a test-set comparison, not a validation choice.

### Controlled research comparison

A controlled TCS run used 100 GAN epochs, 500 LSTM epochs, 25 gradient-calibration epochs, seed 42 and CPU execution. The GAN objective was selected on **validation** scaled Sharpe only, then evaluated once on the held-out test set. It selected `SR MSE`.

| Comparison | Selected GAN (`SR MSE`) | Baseline |
| --- | ---: | ---: |
| Test PnL (bps/day) vs ForGAN | 15.20 | 2.37 |
| Test scaled Sharpe vs ForGAN | 2.73 | 0.41 |
| Directional accuracy vs ForGAN | 57.90% | 50.17% |
| Test MSE vs ForGAN | 0.000040 | 0.000044 |
| Test PnL vs LSTM:SR | 15.20 | 5.91 |
| Test scaled Sharpe vs LSTM:SR | 2.73 | 1.02 |

Caveats for this run:

- It was produced with the calibration bug, so every objective except ForGAN had 25 extra training epochs.
- Its long/short split was not recorded. The same objective was short 94% of the time in the sweep, so the always-short benchmark (Sharpe 1.67 on the same test year) is the bar to clear, not ForGAN.
- With about 261 test days, the standard error of an annualised Sharpe ratio is roughly 1, so 2.73 has a 95% interval of roughly 0.8 to 4.7. The winner was also picked from 9 objectives with one seed.

### Interpretation

Forecast accuracy and trading profit are different objectives: the lowest-error models (`MSE`, `BCE`) lost money. But the saved results cannot yet separate forecasting skill from a directional bet that happened to pay off in a single test year. The re-run reports constant-position baselines and timing-only PnL so that question can be answered.

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

1. **One stock, one test year, one seed.** A Sharpe of about 2 over about 261 test days is roughly ±1. Walk-forward testing and 5–10 seeds per objective are needed before ranking objectives.
2. **No transaction costs.** PnL is `sign(forecast) × return`, with no brokerage, taxes, spread or slippage. The strategy can change position at every open and close.
3. **Not directly tradeable.** The short leg is the Nifty IT index itself, which would need index futures or an ETF. A cash-market short of TCS cannot be held overnight in India. The backtest also trades exactly at the official close, which is also an input to the overnight forecast.
4. **1:1 hedge.** The target is TCS minus the index, not a beta-weighted residual, and TCS is a large constituent of the index.
5. **The LSTMs see one time step.** Inputs are shaped as 1 step × 10 features, so the recurrence is never used and the LSTM acts like a gated feed-forward layer. Reshaping to 10 steps × 1 feature is a planned change.
6. **No overnight/intraday feature.** The model is not told which half-day it is predicting, although the two behave differently.
7. **Only the sign of the mean forecast is traded.** The GAN's predicted distribution (for example P(up), now reported as `Mean P(up)`) is not yet used for sizing.

## Changes since the reported results

- **Calibration no longer trains the models.** `GradientCheck` and `GradientCheckLSTM` used to call `optimizer.step()` after computing every term's gradient. That trained the real model on the last gradient computed (the volatility term) and gave every objective except ForGAN extra epochs. Calibration now runs on deep copies, and the caller's models and optimizers come back unchanged. Weights use the median of per-batch gradient ratios instead of the mean, which a single near-zero gradient could blow up.
- **Training uses the tested loss formulas.** The training loops had their own copies of the PnL/MSE/Sharpe/STD formulas, separate from the ones the tests checked. All training and calibration now goes through `objectives.losses`.
- **Validation and test are scored on raw returns.** Clipping to ±15% still applies to model inputs and training targets, but no longer to the returns a strategy is scored on.
- **Dates match returns.** `excessreturns` returned one date per day for a series with two returns per day. It now returns one date per half-day return.
- **Daily PnL follows real day boundaries.** Half-day PnLs used to be paired from the first test observation, so a "day" could be an intraday return plus the next overnight return, and the overnight/intraday labels depended on where the split happened to fall. Pairs now run overnight → intraday, and both halves are labelled (`Overnight SR scaled`, `Intraday SR scaled`).
- **New baselines and diagnostics in every result row (test and validation):** `Long share`, always-long and always-short PnL and Sharpe, `PnL_w over best constant`, and `Timing PnL_w` / `Timing SR scaled` (PnL after removing the model's average position). `research_comparison.csv` carries these for the selected GAN.
- **Test suite repaired.** The stale `legacy` equivalence tests, which compared each function with itself, were removed, and the layout test no longer requires the git-ignored `results/checkpoints/` directory. Regression tests were added for each fix above. Dead modules (`utils/research.py`, `models/lstm.py`) were deleted.

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

This run selects the GAN objective from validation scaled Sharpe and writes `results/metrics/research_comparison.csv`, including the selected model's long share and the always-long/always-short baselines for the same test window.

## Tests

Run:

```bash
pytest -q
```

The suite contains **22 tests** covering:

- objective formulas, and that training uses them
- calibration leaving models unchanged
- model behaviour and training updates
- date alignment and unclipped evaluation targets
- day-aligned PnL pairing and constant-position baselines
- evaluation, data validation and research-comparison alignment
- checkpoint round-trips

## Outputs

| Path | Contents |
| --- | --- |
| `results/metrics/` | CSV metrics, PnL series, and experiment summaries |
| `results/figures/` | Cumulative PnL plots, written by the runner (none are committed) |
| `docs/TradeGAN.pdf` | Preserved research write-up |

## Reproducibility

For meaningful comparisons, record the ticker and benchmark, data window, train/validation/test split, random seed, objective, model dimensions, learning rates, epoch counts, gradient-calibration iterations, device, and evaluation convention.

See [`docs/reproduction.md`](docs/reproduction.md) for the reproduction workflow.

## Reference

Vuletić, M. and Cont, R. (2023). *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks.*

Please cite the original research for the methodology and distinguish reproduced results from newly generated experiments.
