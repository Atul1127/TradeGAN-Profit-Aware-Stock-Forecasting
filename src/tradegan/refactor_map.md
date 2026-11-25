# TradeGAN Refactor Map

The former monolithic implementation has been split into focused modules. The `legacy.py` compatibility shim and the archived `models/lstm.py` have been removed.

| Responsibility | Module | Status |
| --- | --- | --- |
| Market/ETF lookup | `data/market.py` | complete |
| Return construction | `data/returns.py` | complete |
| Train/validation/test windows | `data/splits.py` | complete |
| GAN models | `models/gan.py` | complete |
| LSTM forecaster | `lstm_model.py` | complete |
| Trading metrics | `utils/trading_metrics.py` | complete |
| Tensor helpers | `utils/tensors.py` | complete |
| Objective formulas | `objectives/losses.py` | complete |
| Gradient calibration | `objectives/gradient_analysis.py` | complete |
| GAN training | `training/gan.py` | complete |
| LSTM training | `training/lstm.py` | complete |
| Shared evaluation helpers (day pairing, baselines) | `evaluation/common.py` | complete |
| GAN evaluation | `evaluation/gan.py` | complete |
| LSTM evaluation | `evaluation/lstm.py` | complete |
| GAN experiment entry point | `experiments/gan.py` | complete |
| LSTM experiment entry point | `experiments/lstm.py` | complete |
| Main runnable orchestration | `fixed_experiment.py` | complete |

## Imports

Import from the specific module you use, for example `from tradegan.objectives.losses import trading_terms`. All training and calibration code computes its loss terms through `objectives.losses`.

## Validation

Run these from the project environment before a long experiment:

```bash
pytest -q
python scripts/run_experiment.py --ticker TCS --gan-epochs 1 --lstm-epochs 1 --gradient-epochs 1 --cpu
```

For a full experiment, increase the epoch counts only after the smoke run completes cleanly.
