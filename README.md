# TradeGAN

**A reproduction/application of the Fin-GAN methodology of Vuletić & Cont (2023), applied to half-day TCS excess log-returns against the Nifty IT benchmark (`^CNXIT`).**

> ### Attribution
> The Fin-GAN methodology, model name, ForGAN-based architecture, and economics-driven generator objectives originate with Milena Vuletić and Rama Cont, *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks* (2023). This repository is a reproduction/application and is not the original implementation.

## What it does

TradeGAN uses a conditional GAN with LSTM layers to forecast **half-day excess log-returns** for TCS relative to the Nifty IT index.

The central idea is to optimize the generator for both forecast quality and trading performance rather than relying only on conventional prediction error.

## Results

These results come from the controlled run after the November 2026 fixes: TCS against Nifty IT, 2010–2021, an 80/10/10 time split, a test window of 291 trading days, seed 42, 100 GAN epochs, 500 LSTM epochs and 25 calibration epochs. PnL is gross, in bps of pair notional per day. Sharpe is annualised. All figures come from **one seed**; see [Known limitations](#known-limitations).

### Headline (objective chosen on validation only)

Validation Sharpe picked `SR MSE` (2.67, the highest of the 9 objectives). It was then evaluated once on test.

| Test metric | `SR MSE` | ForGAN | Best LSTM on validation (`SR`) | Always short |
| --- | ---: | ---: | ---: | ---: |
| PnL (bps/day) | **14.38** | 2.56 | 3.40 | 8.52 |
| Sharpe | **2.61** | 0.44 | 0.59 | 1.62 |
| Directional accuracy | 55.8% | 50.0% | 51.0% | — |
| Long share | 50.2% | 73.9% | 66.3% | 0% |
| PnL over best constant position | **+5.86** | −5.96 | −5.12 | 0 |
| Timing-only Sharpe | **2.62** | 1.29 | 1.12 | — |

TCS lagged the index over the test year, so always-short is the bar to beat. `SR MSE` was long half the time and beat it by 5.9 bps a day. With its average position removed, its Sharpe stays at 2.6, so its profit comes from timing rather than a directional bet. Its hit rate was 57% of days and its maximum drawdown about 830 bps. Intraday half-days contributed more (Sharpe 2.36) than overnight ones (1.05).

### All GAN objectives (test)

This table shows every objective for comparison. It is not a selection table: picking the best row by test Sharpe would be selecting on the test set.

| Objective | Val Sharpe | Test PnL | Test Sharpe | Long share | Over best constant | Timing Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`SR MSE`** (selected) | **2.67** | 14.38 | 2.61 | 50% | +5.86 | 2.62 |
| `PnL MSE SR` | 0.46 | 14.43 | 2.57 | 51% | +5.91 | 2.61 |
| `PnL MSE` | −0.23 | 6.57 | 1.16 | 69% | −1.95 | 1.84 |
| `PnL SR` | 1.88 | 5.72 | 1.02 | 62% | −2.80 | 1.40 |
| `PnL MSE STD` | 1.04 | 1.67 | 0.29 | 50% | −6.85 | 0.27 |
| `PnL STD` | 1.78 | 0.07 | 0.01 | 54% | −8.45 | 0.12 |
| `PnL` | 1.08 | −0.84 | −0.15 | 55% | −9.36 | 0.00 |
| `SR` | 0.39 | −8.69 | −1.47 | 63% | −17.21 | −1.14 |
| `MSE` | −0.41 | −9.28 | −1.67 | 61% | −17.80 | −1.36 |
| ForGAN (adversarial only) | 0.25 | 2.56 | 0.44 | 74% | −5.96 | 1.29 |

Only the two objectives that combine MSE with the Sharpe term beat always-short. They made almost the same PnL, but their daily PnLs correlate only 0.54, so they are not the same model. No objective collapsed to always one side after the fixes. Before the fixes, `SR` and `PnL STD` had been short 100% of the time.

### LSTM baselines (test)

| Objective | Val Sharpe | Test PnL | Test Sharpe | Long share | Over best constant |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SR` (selected on validation) | 2.23 | 3.40 | 0.59 | 66% | −5.12 |
| `PnL` | 1.42 | 6.25 | 1.08 | 66% | −2.27 |
| `MSE` | 0.93 | 6.16 | 1.10 | 60% | −2.35 |
| `STD` | 0.72 | 6.08 | 1.04 | 56% | −2.44 |
| `PnL STD` | 1.68 | 6.06 | 1.05 | 67% | −2.46 |
| `PnL SR` | 1.57 | 1.23 | 0.22 | 66% | −7.29 |

No LSTM beat always-short on test.

### How much to trust this

- **Validation is a weak guide to test performance.** Across the 9 GAN objectives, the rank correlation between validation and test Sharpe is 0.47. For the 6 LSTMs it is −0.43. `SR MSE` topped both validation and test here, but `PnL MSE SR` made the same on test after ranking 6th on validation. Seed-to-seed variation could easily reorder objectives.
- **Sampling error is large.** Over 291 days, the standard error of an annualised Sharpe is about 0.93, so 2.61 has a 95% interval of roughly 0.8 to 4.4.
- **No costs in these numbers.** A model that is long half the time changes position often. The runner now reports turnover, break-even cost and net Sharpe; those columns appear from the next run.

Pre-fix results are archived in `results/metrics/pre_fix_sweep/`. They are not comparable with the tables above.

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

1. **One seed, one stock, one test year (291 days).** Use the multi-seed workflow under [Run](#run) before ranking objectives.
2. **Costs are an assumption.** Net PnL subtracts a flat cost per unit of pair notional traded (default 5 bps, one-way, both legs together; `--cost-bps`). Real costs depend on brokerage, taxes, spreads and the instruments used. Report the break-even cost alongside it.
3. **Not directly tradeable.** The short leg is the Nifty IT index, which would need index futures or an ETF. A cash-market short of TCS cannot be held overnight in India. The backtest trades exactly at the official close, which is also an input to the overnight forecast.
4. **1:1 hedge.** The target is TCS minus the index, not a beta-weighted residual, and TCS is a large constituent of the index.
5. **The LSTMs see one time step.** Inputs are 1 step × 10 features, so the recurrence is never used.
6. **No overnight/intraday feature.** The model is not told which half-day it predicts.
7. **Only the sign of the mean forecast is traded.** The GAN's predicted distribution (`Mean P(up)`) is not yet used for sizing.

## Changes in the November 2026 fixes

- **Calibration no longer trains the models.** `GradientCheck` and `GradientCheckLSTM` used to call `optimizer.step()` after computing every term's gradient. That trained the real model on the last gradient computed (the volatility term) and gave every objective except ForGAN extra epochs. Calibration now runs on deep copies, and the caller's models and optimizers come back unchanged. Weights use the median of per-batch gradient ratios instead of the mean, which a single near-zero gradient could blow up.
- **Training uses the tested loss formulas.** The training loops had their own copies of the PnL/MSE/Sharpe/STD formulas, separate from the ones the tests checked. All training and calibration now goes through `objectives.losses`.
- **Validation and test are scored on raw returns.** Clipping to ±15% still applies to model inputs and training targets, but no longer to the returns a strategy is scored on.
- **Dates match returns.** `excessreturns` returned one date per day for a series with two returns per day. It now returns one date per half-day return.
- **Daily PnL follows real day boundaries.** Half-day PnLs used to be paired from the first test observation, so a "day" could be an intraday return plus the next overnight return, and the overnight/intraday labels depended on where the split happened to fall. Pairs now run overnight → intraday, and both halves are labelled (`Overnight SR scaled`, `Intraday SR scaled`).
- **New baselines and diagnostics in every result row (test and validation):** `Long share`, always-long and always-short PnL and Sharpe, `PnL_w over best constant`, and `Timing PnL_w` / `Timing SR scaled` (PnL after removing the model's average position). `research_comparison.csv` carries these for the selected GAN.
- **Costs and turnover.** Every result row reports `Turnover per day`, `Break-even cost bps`, `Net PnL_w` and `Net SR scaled` at an assumed cost (`--cost-bps`, default 5).
- **Multi-seed runs.** `--results-dir` writes each run to its own folder, and `scripts/summarize_seeds.py` summarises them.
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

This run selects the GAN objective on validation scaled Sharpe and writes `results/metrics/research_comparison.csv`. That file includes the selected model's long share, always-long and always-short baselines, turnover, break-even cost and net Sharpe. Add `--cost-bps 3` (or another value) to change the cost assumption.

### Multi-seed check

```bash
for s in 1 2 3 4 5; do
  python scripts/run_experiment.py --ticker TCS --gan-epochs 100 --lstm-epochs 500 \
    --gradient-epochs 25 --seed $s --cpu --results-dir results/seeds/seed_$s
done
python scripts/summarize_seeds.py
```

The summary reports the validation-selected GAN for each seed, which is the honest headline, and every objective's mean and spread over seeds. It writes `results/seeds/selected_per_seed.csv` and `results/seeds/objectives_over_seeds.csv`.

## Tests

Run:

```bash
pytest -q
```

The suite contains **27 tests** covering:

- objective formulas, and that training uses them
- calibration leaving models unchanged
- model behaviour and training updates
- date alignment and unclipped evaluation targets
- day-aligned PnL pairing and constant-position baselines
- turnover and cost accounting, and the multi-seed summary
- evaluation, data validation and research-comparison alignment
- checkpoint round-trips

## Outputs

| Path | Contents |
| --- | --- |
| `results/metrics/` | CSV metrics, PnL series, and experiment summaries; `pre_fix_sweep/` holds archived pre-fix results |
| `results/seeds/` | Per-seed runs and the multi-seed summary, when run |
| `results/figures/` | Cumulative PnL plots, written by the runner (none are committed) |
| `docs/TradeGAN.pdf` | Preserved research write-up |

## Reproducibility

For meaningful comparisons, record the ticker and benchmark, data window, train/validation/test split, random seed, objective, model dimensions, learning rates, epoch counts, gradient-calibration iterations, device, and evaluation convention.

See [`docs/reproduction.md`](docs/reproduction.md) for the reproduction workflow.

## Reference

Vuletić, M. and Cont, R. (2023). *Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks.*

Please cite the original research for the methodology and distinguish reproduced results from newly generated experiments.
