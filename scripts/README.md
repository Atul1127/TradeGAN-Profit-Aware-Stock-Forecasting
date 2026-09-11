# Scripts

Command-line entry points belong here. Scripts should be thin wrappers around the
`tradegan` package and should not contain model or data-processing logic.

Planned entry points:

- `download_data.py` — download raw market data.
- `preprocess.py` — build model-ready datasets.
- `train.py` — train a configured model/experiment.
- `evaluate.py` — evaluate a saved checkpoint.
- `run_experiment.py` — run a complete reproducible experiment.
