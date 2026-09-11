# Legacy Refactor Map

`legacy.py` remains the frozen behavioral reference during migration. Extracted modules now contain the first set of real implementations, with regression tests comparing them directly to `legacy.py`.

| Legacy responsibility | Extracted module | Status |
| --- | --- | --- |
| `ETF_find` | `data/market.py` | extracted + tested |
| `excessreturns`, `excessreturns_closeonly`, `rawreturns` | `data/returns.py` | extracted + tested |
| `split_train_val_test`, `split_train_testraw`, `split_train_val_testraw` | `data/splits.py` | extracted + tested |
| `Generator`, `Discriminator` | `models/gan.py` | extracted + tested |
| `LSTM` (legacy architecture) | `models/lstm.py` | extracted + tested |
| `combine_vectors` | `utils/tensors.py` | extracted + tested |
| `getPnL`, `getSR` | `utils/trading_metrics.py` | extracted + tested |
| `Evaluation2`, `Evaluation3` | `evaluation/gan.py` | boundary prepared; legacy implementation retained |
| `Evaluation2LSTM` | `evaluation/lstm.py` | boundary prepared; legacy implementation retained |
| `GradientCheck`, `GradientCheckLSTM` | `objectives/gradient_analysis.py` | boundary prepared; legacy implementation retained |
| GAN training loops | `training/gan.py` | boundary prepared; legacy implementation retained |
| LSTM training loops | `training/lstm.py` | boundary prepared; legacy implementation retained |
| `FinGAN_combos` | `experiments/gan.py` | boundary prepared; legacy implementation retained |
| `LSTM_combos` | `experiments/lstm.py` | boundary prepared; legacy implementation retained |
| Explicit objective formulas | `objectives/losses.py` | isolated; not yet wired into training |

## Migration rule

The first extraction stage copies the original implementations into focused modules and adds direct regression tests against the frozen monolith. The legacy file is deliberately left untouched until each subsystem has passed equivalence checks. This avoids turning a refactor into an algorithm change.

The corrected `fixed_experiment.py` remains the runnable path. The legacy implementations continue to serve as the reference for subsequent extraction of evaluation, gradient analysis, training loops, and orchestration.
