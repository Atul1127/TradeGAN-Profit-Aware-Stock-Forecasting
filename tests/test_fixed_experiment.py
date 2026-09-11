from pathlib import Path

from tradegan.fixed_experiment import _safe_stats
import torch


def test_safe_stats_has_finite_nonzero_std():
    mean, std = _safe_stats(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isfinite(mean)
    assert torch.isfinite(std)
    assert std.item() > 0


def test_project_layout_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "results" / "metrics").is_dir()
    assert (root / "results" / "checkpoints").is_dir()
