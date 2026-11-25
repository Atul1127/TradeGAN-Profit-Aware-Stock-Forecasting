"""Tests for turnover/cost metrics and the multi-seed summary."""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from scripts.summarize_seeds import per_objective, selected_per_seed
from tradegan.evaluation.common import cost_metrics


def test_constant_position_pays_only_to_open():
    real = torch.full((10,), 0.001)  # 10 bps per half-day
    out = cost_metrics(-torch.ones(10), real, start=1, cost_bps=5.0)
    # Opening the short trades 1 unit once over 5 days.
    assert out["Turnover per day"] == pytest.approx(1 / 5)
    assert out["Net PnL_w"] == pytest.approx(-20.0 - 5.0 / 5)
    assert out["Cost bps"] == 5.0


def test_flipping_every_half_day_is_charged_two_units_per_flip():
    position = torch.tensor([1.0, -1.0, 1.0, -1.0])
    real = torch.tensor([0.001, -0.001, 0.001, -0.001])  # always right: +10 bps each
    out = cost_metrics(position, real, start=1, cost_bps=3.0)
    # traded = 1 (open) + 2 + 2 + 2 = 7 units over 2 days; gross = 40 bps.
    assert out["Turnover per day"] == pytest.approx(3.5)
    assert out["Break-even cost bps"] == pytest.approx(40 / 7)
    assert out["Net PnL_w"] == pytest.approx((40 - 3.0 * 7) / 2)


def test_zero_cost_net_equals_gross():
    torch.manual_seed(0)
    position = torch.sign(torch.randn(50))
    real = torch.randn(50) * 0.01
    out = cost_metrics(position, real, start=0, cost_bps=0.0)
    from tradegan.evaluation.common import paired_pnl
    assert out["Net PnL_w"] == pytest.approx(float(paired_pnl(10000 * position * real, 0).mean()), rel=1e-5)
    with pytest.raises(ValueError):
        cost_metrics(position, real, start=0, cost_bps=-1.0)


def _fake_results():
    rows = []
    for seed, (val_a, val_b) in {"1": (2.0, 1.0), "2": (0.5, 1.5)}.items():
        for objective, val, test in (("A", val_a, 2.5), ("B", val_b, 0.5), ("ForGAN", 3.0, 0.1)):
            rows.append({
                "seed": seed, "type": objective, "SR_w scaled val": val, "SR_w scaled": test + float(seed),
                "PnL_w": 1.0, "Timing SR scaled": 1.0, "Net SR scaled": 0.5,
                "PnL_w over best constant": 1.0 if objective == "A" else -1.0,
                "Long share": 0.5, "Turnover per day": 1.0, "Directional Accuracy": 0.55,
                "Always-short SR scaled": 1.6,
            })
    return pd.DataFrame(rows)


def test_seed_summary_selects_on_validation_and_excludes_forgan():
    selected = selected_per_seed(_fake_results())
    assert selected["selected_objective"].tolist() == ["A", "B"]
    assert selected["test_sharpe"].tolist() == [3.5, 2.5]
    assert selected["forgan_test_sharpe"].tolist() == [1.1, 2.1]


def test_seed_summary_aggregates_each_objective():
    summary = per_objective(_fake_results())
    assert summary.loc["A", "test_sharpe_mean"] == pytest.approx(4.0)
    assert summary.loc["A", "share_of_seeds_beating_best_constant"] == 1.0
    assert summary.loc["B", "share_of_seeds_beating_best_constant"] == 0.0
    assert (summary["seeds"] == 2).all()
