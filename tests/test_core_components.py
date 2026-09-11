from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytest

from tradegan.data.returns import _load_price_frame
from tradegan.data.splits import _validate_split_args
from tradegan.evaluation.gan import Evaluation2
from tradegan.evaluation.lstm import Evaluation2LSTM
from tradegan.fixed_experiment import _save_checkpoint
from tradegan.lstm_model import LSTMForecaster
from tradegan.models.gan import Discriminator, Generator
from tradegan.objectives.gradient_analysis import GradientCheck, GradientCheckLSTM
from tradegan.objectives.losses import (
    gan_loss_pnl,
    gan_loss_pnl_mse,
    gan_loss_pnl_mse_sr,
    gan_loss_pnl_mse_std,
    gan_loss_pnl_sr,
    gan_loss_pnl_std,
    gan_loss_mse,
    gan_loss_sr,
    gan_loss_sr_mse,
    lstm_loss_mse,
    lstm_loss_pnl,
    lstm_loss_pnl_sr,
    lstm_loss_pnl_std,
    lstm_loss_sr,
    lstm_loss_std,
    trading_terms,
)
from tradegan.training.gan import TrainLoopForGAN, TrainLoopMainPnLnv
from tradegan.training.lstm import TrainLoopnLSTMPnL


def _tiny_gan(batch: int = 8, lookback: int = 4, z_dim: int = 3, hidden: int = 4):
    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)
    generator = Generator(z_dim, lookback, hidden, 1, mean, std)
    discriminator = Discriminator(lookback + 1, hidden, mean, std)
    return generator, discriminator


def _tiny_lstm(lookback: int = 4, hidden: int = 4):
    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)
    return LSTMForecaster(0, lookback, hidden, 1, mean, std)


def test_trading_terms_match_direct_calculation():
    prediction = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
    real = torch.tensor([0.2, -0.4, 0.5])
    coefficient = 2.0
    pnl, mse, sr, std = trading_terms(prediction, real, coefficient)

    pnl_samples = torch.tanh(coefficient * prediction) * real
    expected_pnl = pnl_samples.mean()
    expected_mse = torch.norm(prediction - real).square() / prediction.shape[0]
    expected_std = pnl_samples.std()
    expected_sr = expected_pnl / expected_std

    torch.testing.assert_close(pnl, expected_pnl)
    torch.testing.assert_close(mse, expected_mse)
    torch.testing.assert_close(std, expected_std)
    torch.testing.assert_close(sr, expected_sr)

    (pnl + mse + sr + std).backward()
    assert torch.isfinite(prediction.grad).all()


def test_objective_formulas_are_correct():
    bce, alpha, beta, gamma, delta = (torch.tensor(v) for v in (1.0, 2.0, 3.0, 4.0, 5.0))
    pnl, mse, sr, std = (torch.tensor(v) for v in (0.1, 0.2, 0.3, 0.4))

    assert gan_loss_pnl(bce, alpha, pnl).item() == pytest.approx(0.8)
    assert gan_loss_pnl_mse(bce, alpha, pnl, beta, mse).item() == pytest.approx(1.4)
    assert gan_loss_pnl_mse_sr(bce, alpha, pnl, beta, mse, gamma, sr).item() == pytest.approx(0.2)
    assert gan_loss_pnl_mse_std(bce, alpha, pnl, beta, mse, delta, std).item() == pytest.approx(3.4)
    assert gan_loss_pnl_sr(bce, alpha, pnl, gamma, sr).item() == pytest.approx(-0.4)
    assert gan_loss_pnl_std(bce, alpha, pnl, delta, std).item() == pytest.approx(2.8)
    assert gan_loss_mse(bce, beta, mse).item() == pytest.approx(1.6)
    assert gan_loss_sr(bce, gamma, sr).item() == pytest.approx(-0.2)
    assert gan_loss_sr_mse(bce, beta, mse, gamma, sr).item() == pytest.approx(0.4)

    assert lstm_loss_mse(mse).item() == pytest.approx(0.2)
    assert lstm_loss_pnl(mse, alpha, pnl).item() == pytest.approx(0.0)
    assert lstm_loss_pnl_std(mse, alpha, pnl, delta, std).item() == pytest.approx(2.0)
    assert lstm_loss_pnl_sr(mse, alpha, pnl, gamma, sr).item() == pytest.approx(-1.0)
    assert lstm_loss_sr(mse, gamma, sr).item() == pytest.approx(-1.0)
    assert lstm_loss_std(mse, delta, std).item() == pytest.approx(2.2)


def test_models_support_non_default_dimensions_and_backward():
    torch.manual_seed(7)
    generator, discriminator = _tiny_gan(hidden=5, z_dim=3, lookback=4)
    batch = 6
    condition = torch.randn(1, batch, 4)
    noise = torch.randn(1, batch, 3)
    h0 = torch.zeros(1, batch, 5)
    c0 = torch.zeros_like(h0)
    fake = generator(noise, condition, h0, c0)
    pred = discriminator(torch.cat([condition, fake], dim=-1), h0, c0)

    assert fake.shape == (1, batch, 1)
    assert pred.shape == (1, batch, 1)
    assert torch.isfinite(fake).all()
    assert torch.isfinite(pred).all()

    loss = pred.mean() + fake.mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in generator.parameters())
    assert any(parameter.grad is not None for parameter in discriminator.parameters())

    lstm = _tiny_lstm(lookback=4, hidden=5)
    lstm_h0 = torch.zeros(1, batch, 5)
    lstm_out = lstm(condition, lstm_h0, torch.zeros_like(lstm_h0))
    assert lstm_out.shape == (1, batch, 1)
    assert torch.isfinite(lstm_out).all()


def test_training_loops_update_parameters():
    torch.manual_seed(11)
    train_data = torch.randn(16, 5)
    criterion = nn.BCELoss()

    generator, discriminator = _tiny_gan()
    gen_opt = torch.optim.RMSprop(generator.parameters(), lr=1e-3)
    disc_opt = torch.optim.RMSprop(discriminator.parameters(), lr=1e-3)
    before = {name: value.detach().clone() for name, value in generator.state_dict().items()}

    TrainLoopForGAN(
        generator, discriminator, gen_opt, disc_opt, criterion,
        0, 0, 0, 0, 1, 0, train_data, train_data,
        8, 4, 4, 3, 1e-3, 1e-3, 1, 4, 1, 1, 100, "cpu", False,
    )
    assert any(not torch.equal(before[name], value) for name, value in generator.state_dict().items())

    generator2, discriminator2 = _tiny_gan()
    gen_opt2 = torch.optim.RMSprop(generator2.parameters(), lr=1e-3)
    disc_opt2 = torch.optim.RMSprop(discriminator2.parameters(), lr=1e-3)
    before2 = {name: value.detach().clone() for name, value in generator2.state_dict().items()}
    TrainLoopMainPnLnv(
        generator2, discriminator2, gen_opt2, disc_opt2, criterion,
        1.0, 0.0, 0.0, 0.0, 1, 0, train_data, train_data,
        8, 4, 4, 3, 1e-3, 1e-3, 1, 4, 1, 1, 100, "cpu", False,
    )
    assert any(not torch.equal(before2[name], value) for name, value in generator2.state_dict().items())


def test_lstm_training_and_gradient_calibration_are_finite():
    torch.manual_seed(13)
    train_data = torch.randn(16, 5)
    model = _tiny_lstm()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    model, optimizer, alpha, beta, gamma, delta = GradientCheckLSTM(
        "TCS", model, optimizer, 1, train_data, 8, 4, 4, 0,
        1e-3, 1e-3, 1, 4, 1, 1, 100, "cpu", False,
    )
    for value in (alpha, beta, gamma, delta):
        assert torch.isfinite(value)

    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model, optimizer = TrainLoopnLSTMPnL(
        model, optimizer, False, alpha, beta, gamma, delta, 1, 0,
        train_data, train_data, 8, 4, 4, 0, 1e-3, 1e-3, 1, 4, 1, 1, 100, "cpu", False,
    )
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())

    generator, discriminator = _tiny_gan()
    gen_opt = torch.optim.RMSprop(generator.parameters(), lr=1e-3)
    disc_opt = torch.optim.RMSprop(discriminator.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    *_, gan_alpha, gan_beta, gan_gamma, gan_delta = GradientCheck(
        "TCS", generator, discriminator, gen_opt, disc_opt, criterion, 1,
        train_data, 8, 4, 4, 3, 1e-3, 1e-3, 1, 4, 1, 1, 100, "cpu", False,
    )
    for value in (gan_alpha, gan_beta, gan_gamma, gan_delta):
        assert torch.isfinite(value)


def test_evaluation_runs_with_small_mc_sample_count():
    torch.manual_seed(17)
    train_mean = torch.tensor(0.0)
    train_std = torch.tensor(1.0)
    generator = Generator(3, 4, 4, 1, train_mean, train_std)
    lstm = LSTMForecaster(0, 4, 4, 1, train_mean, train_std)
    test = torch.randn(12, 5)
    val = torch.randn(12, 5)

    gan_result, gan_pnl, *_ = Evaluation2(
        "TCS", 2, generator, test, val, 1, 4, 1, 4, 4, 3,
        1e-3, 1e-3, 1, "PnL", 0, "cpu", "", "", False, 2
    )
    assert len(gan_result) == 1
    assert len(gan_pnl) == 6
    assert np.isfinite(gan_result.select_dtypes(include=[np.number]).to_numpy()).all()

    lstm_result, lstm_pnl, *_ = Evaluation2LSTM(
        "TCS", 2, lstm, test, val, 1, 4, 1, 4, 4, 0,
        1e-3, 1e-3, 1, "PnL", 0, "cpu", "", "", False
    )
    assert len(lstm_result) == 1
    assert len(lstm_pnl) == 6
    assert np.isfinite(lstm_result.select_dtypes(include=[np.number]).to_numpy()).all()


def test_data_validation_rejects_invalid_inputs(tmp_path):
    with pytest.raises(ValueError, match="tr must be between"):
        _validate_split_args(0.0, 0.1, 1, 4, 1)
    with pytest.raises(ValueError, match=r"tr \+ vl"):
        _validate_split_args(0.9, 0.1, 1, 4, 1)
    with pytest.raises(ValueError, match="positive"):
        _validate_split_args(0.8, 0.1, 0, 4, 1)

    bad = pd.DataFrame({"date": ["2020-01-01"], "AdjOpen": [-1.0], "AdjClose": [1.0]})
    bad_path = tmp_path / "bad.csv"
    bad.to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match="non-positive prices"):
        _load_price_frame(tmp_path, "bad")


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(19)
    model, discriminator = _tiny_gan()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    path = tmp_path / "generator.pt"
    _save_checkpoint(path, model, optimizer, "PnL", "TCS")

    payload = torch.load(path, map_location="cpu")
    restored, _ = _tiny_gan()
    restored.load_state_dict(payload["model_state_dict"])

    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name])
    assert payload["objective"] == "PnL"
    assert payload["ticker"] == "TCS"
