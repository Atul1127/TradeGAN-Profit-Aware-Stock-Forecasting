# Reproduction

1. Install dependencies from `requirements.txt`.
2. Download the required historical prices into `data/` using `scripts/download_data.py`.
3. Run the unit/integration tests:

```bash
pytest -q
```

4. Run a short end-to-end smoke test:

```bash
python scripts/run_experiment.py --ticker TCS --gan-epochs 1 --lstm-epochs 1 --gradient-epochs 1 --seed 42 --cpu
```

5. For the controlled TCS research run, use the reference settings below explicitly:

```bash
python scripts/run_experiment.py --ticker TCS --gan-epochs 100 --lstm-epochs 500 --gradient-epochs 25 --seed 42 --cpu
```

The reference experiment uses the 2010–2021 data window, an 80/10/10 train/validation/test split, seed 42, and 25 gradient-calibration iterations. The current runner is CLI-driven and does not load a YAML configuration file.

The controlled run evaluates each objective, selects the non-ForGAN GAN objective with the highest validation scaled Sharpe, evaluates that selected model on the held-out test set, and compares it with the ForGAN and best-validation-Sharpe LSTM baselines.

The resulting `results/metrics/research_comparison.csv` reports:

- test PnL and baseline PnL
- PnL fold change
- test and baseline scaled Sharpe
- directional accuracy
- MSE and MSE reduction percentage
- PnL standard deviation and its reduction percentage

This makes the reported performance claims directly testable rather than inferred from positive/negative prediction proportions or unrelated metrics.

6. Generated metrics, figures, and checkpoints are stored under `results/`.

A claimed performance figure is considered reproducible only when it is obtained from the controlled run and the baseline/evaluation definition is recorded alongside it.
