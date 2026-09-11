# Reproduction

1. Install dependencies from `requirements.txt`.
2. Download the required historical prices into `data/` using `scripts/download_data.py`.
3. Run the smoke tests:

```bash
pytest -q
python scripts/run_experiment.py --ticker TCS --gan-epochs 1 --lstm-epochs 1 --gradient-epochs 1 --cpu
```

4. For a longer experiment, increase the epoch arguments and remove `--cpu` when GPU execution is desired.
5. Generated metrics, figures, and checkpoints are stored under `results/`.

`configs/default.yaml` is a reference configuration for the intended experiment settings. The current runner is CLI-driven and does not automatically load YAML configuration files.

Historical outputs retained in this repository are separated from source code and are not overwritten by the smoke-test workflow.
