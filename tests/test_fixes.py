"""Regression tests for calibration, data alignment and evaluation fixes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from tradegan.data.returns import excessreturns
from tradegan.data.splits import split_train_val_test, target_start_indices
from tradegan.evaluation.common import half_day_split, paired_pnl, position_diagnostics
from tradegan.evaluation.gan import Evaluation2
from tradegan.lstm_model import LSTMForecaster
from tradegan.models.gan import Discriminator, Generator
from tradegan.objectives.gradient_analysis import GradientCheck, GradientCheckLSTM
from tradegan.objectives.losses import trading_terms
from tradegan.training import gan as gan_training
from tradegan.training import lstm as lstm_training


def _snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _unchanged(module, snapshot):
    return all(torch.equal(snapshot[k], v) for k, v in module.state_dict().items())


def test_gan_calibration_does_not_change_models_or_optimizers():
    torch.manual_seed(0)
    gen = Generator(3, 4, 4, 1, torch.tensor(0.0), torch.tensor(1.0))
    disc = Discriminator(5, 4, torch.tensor(0.0), torch.tensor(1.0))
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=1e-2)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=1e-2)
    gen_before, disc_before = _snapshot(gen), _snapshot(disc)

    out = GradientCheck(
        "TCS", gen, disc, gen_opt, disc_opt, nn.BCELoss(), 3,
        torch.randn(32, 5), 8, 4, 4, 3, 1e-2, 1e-2, 1, 4, 1, 1, 100, "cpu", False,
    )
    assert out[0] is gen and out[1] is disc
    assert _unchanged(gen, gen_before)
    assert _unchanged(disc, disc_before)
    assert not gen_opt.state and not disc_opt.state
    for weight in out[4:]:
        assert torch.isfinite(weight) and weight > 0


def test_lstm_calibration_does_not_change_model():
    torch.manual_seed(1)
    model = LSTMForecaster(0, 4, 4, 1, torch.tensor(0.0), torch.tensor(1.0))
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-2)
    before = _snapshot(model)
    GradientCheckLSTM("TCS", model, opt, 3, torch.randn(32, 5), 8, 4, 4, 0,
                      1e-2, 1e-2, 1, 4, 1, 1, 100, "cpu", False)
    assert _unchanged(model, before)
    assert not opt.state


def test_training_objectives_use_the_tested_formulas():
    torch.manual_seed(2)
    fake = torch.randn(1, 6, 1)
    real = torch.randn(1, 6, 1)
    bce = torch.tensor(0.7)
    pnl, mse, sr, std = trading_terms(fake, real, 100.0)
    a, b, g, d = 1.5, 2.0, 0.3, 0.4

    expected = {
        gan_training._objective_pnl: bce - a * pnl,
        gan_training._objective_pnl_mse: bce - a * pnl + b * mse,
        gan_training._objective_pnl_mse_sr: bce - a * pnl + b * mse - g * sr,
        gan_training._objective_pnl_mse_std: bce - a * pnl + b * mse + d * std,
        gan_training._objective_pnl_sr: bce - a * pnl - g * sr,
        gan_training._objective_pnl_std: bce - a * pnl + d * std,
        gan_training._objective_mse: bce + b * mse,
        gan_training._objective_sr: bce - g * sr,
        gan_training._objective_sr_mse: bce + b * mse - g * sr,
    }
    for objective, value in expected.items():
        torch.testing.assert_close(objective(bce, fake, real, a, b, g, d, 100.0), value)

    torch.testing.assert_close(lstm_training._lstm_std(fake, real, a, b, g, d, 100.0), mse + d * std)
    torch.testing.assert_close(lstm_training._lstm_pnl_sr(fake, real, a, b, g, d, 100.0), mse - a * pnl - g * sr)


def test_sharpe_term_is_zero_for_constant_pnl():
    prediction = torch.ones(4, requires_grad=True)
    real = torch.full((4,), 0.01)
    _, _, sr, _ = trading_terms(prediction, real, 100.0)
    assert sr.item() == 0.0
    sr.backward()
    assert torch.isfinite(prediction.grad).all()


def _market_files(tmp_path, days=60, spike_day=None):
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=days)
    stock_log = np.cumsum(rng.normal(0, 0.01, 2 * days)) + np.log(100)
    index_log = np.cumsum(rng.normal(0, 0.01, 2 * days)) + np.log(1000)
    if spike_day is not None:
        stock_log[2 * spike_day + 1:] += 0.4  # +40% intraday jump on spike_day
    pd.DataFrame({"date": dates, "AdjOpen": np.exp(stock_log[0::2]), "AdjClose": np.exp(stock_log[1::2])}) \
        .to_csv(tmp_path / "TCS.csv", index=False)
    pd.DataFrame({"date": dates, "AdjOpen": np.exp(index_log[0::2]), "AdjClose": np.exp(index_log[1::2])}) \
        .to_csv(tmp_path / "^CNXIT.csv", index=False)
    pd.DataFrame({"ticker_x": ["TCS"], "ticker_y": ["^CNXIT"]}).to_csv(tmp_path / "meta.csv", index=False)
    return dates


def test_dates_have_one_entry_per_half_day_return(tmp_path):
    dates = _market_files(tmp_path, days=10)
    returns, dates_out = excessreturns(str(tmp_path) + "/", "TCS", "^CNXIT")
    assert len(dates_out) == len(returns) == 2 * len(dates) - 1
    # Return 0 is day-0 intraday, return 1 is overnight into day 1, return 2 is day-1 intraday.
    assert dates_out[0] == dates[0]
    assert dates_out[1] == dates[1] and dates_out[2] == dates[1]


def test_validation_and_test_targets_are_not_clipped(tmp_path):
    days = 200
    dates = _market_files(tmp_path, days=days, spike_day=days - 5)
    clipped, raw, _ = excessreturns(str(tmp_path) + "/", "TCS", "^CNXIT", return_raw=True)
    spike = 2 * (days - 5)  # intraday return index of spike_day
    assert raw[spike] > 0.3 and clipped[spike] < 0.3

    _train, _val, test, _ = split_train_val_test("TCS", str(tmp_path) + "/", str(tmp_path / "meta.csv"),
                                                tr=0.8, vl=0.1, l=10)
    assert test[:, -1].max() > 0.3          # targets use raw returns
    assert test[:, :-1].max() < 0.3          # inputs stay clipped


def test_target_start_indices_match_windows():
    n_obs, tr, vl, l = 1001, 0.8, 0.1, 10
    series = np.arange(n_obs, dtype=float)
    n_tr, n_vl = int(tr * n_obs), int(vl * n_obs)
    val_start, test_start = target_start_indices(n_obs, tr, vl, l)
    assert series[n_tr:n_tr + n_vl][l] == val_start
    assert series[n_tr + n_vl:][l] == test_start


def test_pairing_follows_day_boundaries():
    pnl = torch.arange(6, dtype=torch.float)  # global indices 4..9 when start=4
    # start=4 is intraday: drop it, then pair (5,6) (7,8); 9 has no intraday half.
    torch.testing.assert_close(paired_pnl(pnl, start=4), torch.tensor([1.0 + 2.0, 3.0 + 4.0]))
    # start=5 is overnight: pair (5,6) (7,8) (9,10).
    torch.testing.assert_close(paired_pnl(pnl, start=5), torch.tensor([1.0, 5.0, 9.0]))
    overnight, intraday = half_day_split(pnl, start=4)
    torch.testing.assert_close(overnight, torch.tensor([1.0, 3.0, 5.0]))
    torch.testing.assert_close(intraday, torch.tensor([0.0, 2.0, 4.0]))


def test_always_short_model_has_no_timing_skill():
    torch.manual_seed(4)
    real = torch.randn(200) * 0.01 - 0.001
    diag = position_diagnostics(-torch.ones(200), real, start=1)
    assert diag["Long share"] == 0.0
    assert diag["PnL_w over best constant"] == pytest.approx(0.0, abs=1e-6)
    assert diag["Timing PnL_w"] == pytest.approx(0.0, abs=1e-6)
    assert diag["Always-short PnL_w"] == pytest.approx(-diag["Always-long PnL_w"])


def test_evaluation_reports_baselines_and_half_days():
    torch.manual_seed(5)
    gen = Generator(3, 4, 4, 1, torch.tensor(0.0), torch.tensor(1.0))
    data = torch.randn(13, 5)
    result, *_ = Evaluation2("TCS", 2, gen, data, data, 1, 4, 1, 4, 4, 3, 1e-3, 1e-3, 1, "PnL",
                             0, "cpu", "", "", False, 4, test_start=10, val_start=11)
    for column in ("Long share", "Always-short PnL_w", "Timing SR scaled",
                   "Overnight SR scaled", "Intraday SR scaled", "Long share val"):
        assert column in result.columns
    assert np.isfinite(result.select_dtypes(include=[np.number]).to_numpy()).all()
