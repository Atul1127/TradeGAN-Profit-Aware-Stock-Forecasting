# Configuration

Experiment settings live here instead of being hardcoded in Python scripts.

- `default.yaml` contains the safe baseline configuration.
- Put experiment-specific overrides under `experiments/`.
- Keep source code independent of machine-specific paths.

The intended workflow is:

```text
config -> data pipeline -> model -> training -> evaluation -> artifacts
```
