# Legacy Refactor Map

`legacy.py` is preserved as the behavioral reference while the project is migrated into smaller modules.

| Legacy responsibility | New module |
| --- | --- |
| `ETF_find` | `data/market.py` |
| `excessreturns`, `excessreturns_closeonly`, `rawreturns` | `data/returns.py` |
| `split_train_val_test`, `split_train_testraw`, `split_train_val_testraw` | `data/splits.py` |
| `Generator`, `Discriminator` | `models/gan.py` |
| `LSTM` (legacy implementation) | `models/lstm.py` |
| `combine_vectors`, `getPnL`, `getSR` | `utils/trading_metrics.py` |
| `Evaluation2`, `Evaluation3` | `evaluation/gan.py` |
| `Evaluation2LSTM` | `evaluation/lstm.py` |
| `GradientCheck`, `GradientCheckLSTM` | `objectives/gradient_analysis.py` |
| GAN training loops | `training/gan.py` |
| LSTM training loops | `training/lstm.py` |
| `FinGAN_combos` | `experiments/gan.py` |
| `LSTM_combos` | `experiments/lstm.py` |
| Explicit objective formulas | `objectives/losses.py` |

## Migration rule

The new modules initially delegate to the original functions/classes instead of copying or rewriting their bodies. This makes the first structural refactor behavior-preserving. The explicit loss helpers are isolated but are not wired into the legacy execution path until equivalence tests are added.

`fixed_experiment.py` remains the corrected runnable path. `legacy.py` remains the historical reference until each extracted component is verified against it.
