# Tests

Run `pytest -q` from the repository root.

- `test_core_components.py`: loss formulas, model shapes, training updates, evaluation, data validation, research comparison and checkpoints.
- `test_fixes.py`: regression tests for the calibration, data-alignment and evaluation fixes. These cover calibration leaving models unchanged, training using the tested formulas, dates per half-day, raw evaluation targets, day-aligned pairing and constant-position baselines.
- `test_fixed_experiment.py`: runner helpers and repository layout.
