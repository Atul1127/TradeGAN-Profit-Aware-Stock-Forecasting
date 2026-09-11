from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from tradegan import legacy
from tradegan.data.market import ETF_find as extracted_ETF_find
from tradegan.data.returns import excessreturns as extracted_excessreturns
from tradegan.data.returns import excessreturns_closeonly as extracted_excessreturns_closeonly
from tradegan.data.returns import rawreturns as extracted_rawreturns
from tradegan.data.splits import split_train_val_test as extracted_split_train_val_test
from tradegan.data.splits import split_train_testraw as extracted_split_train_testraw
from tradegan.data.splits import split_train_val_testraw as extracted_split_train_val_testraw
from tradegan.models.gan import Discriminator as ExtractedDiscriminator
from tradegan.models.gan import Generator as ExtractedGenerator
from tradegan.models.lstm import LSTM as ExtractedLegacyLSTM
from tradegan.utils.tensors import combine_vectors as extracted_combine_vectors
from tradegan.utils.trading_metrics import getPnL as extracted_getPnL
from tradegan.utils.trading_metrics import getSR as extracted_getSR


def _write_market_files(tmp_path):
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    base = np.linspace(100.0, 220.0, len(dates))
    stock = pd.DataFrame({"date": dates, "AdjOpen": base, "AdjClose": base + 1.0})
    benchmark = pd.DataFrame({"date": dates, "AdjOpen": base * 0.9, "AdjClose": base * 0.9 + 0.5})
    stock.to_csv(tmp_path / "TCS.csv", index=False)
    benchmark.to_csv(tmp_path / "^CNXIT.csv", index=False)
    pd.DataFrame({"ticker_x": ["TCS"], "ticker_y": ["^CNXIT"]}).to_csv(
        tmp_path / "stocks-etfs-list.csv", index=False
    )


def test_data_extraction_equivalent(tmp_path):
    _write_market_files(tmp_path)
    data_dir = str(tmp_path) + "/"
    metadata = str(tmp_path / "stocks-etfs-list.csv")

    assert legacy.ETF_find(metadata, "TCS") == extracted_ETF_find(metadata, "TCS")

    old_close, old_close_dates = legacy.excessreturns_closeonly(data_dir, "TCS", "^CNXIT")
    new_close, new_close_dates = extracted_excessreturns_closeonly(data_dir, "TCS", "^CNXIT")
    np.testing.assert_allclose(old_close, new_close)
    np.testing.assert_array_equal(old_close_dates.to_numpy(), new_close_dates.to_numpy())

    old_excess, old_dates = legacy.excessreturns(data_dir, "TCS", "^CNXIT")
    new_excess, new_dates = extracted_excessreturns(data_dir, "TCS", "^CNXIT")
    np.testing.assert_allclose(old_excess, new_excess)
    np.testing.assert_array_equal(old_dates.to_numpy(), new_dates.to_numpy())

    old_raw, old_raw_dates = legacy.rawreturns(data_dir, "TCS")
    new_raw, new_raw_dates = extracted_rawreturns(data_dir, "TCS")
    np.testing.assert_allclose(old_raw, new_raw)
    np.testing.assert_array_equal(old_raw_dates.to_numpy(), new_raw_dates.to_numpy())

    old_train, old_val, old_test, old_split_dates = legacy.split_train_val_test("TCS", data_dir, metadata)
    new_train, new_val, new_test, new_split_dates = extracted_split_train_val_test("TCS", data_dir, metadata)
    np.testing.assert_allclose(old_train, new_train)
    np.testing.assert_allclose(old_val, new_val)
    np.testing.assert_allclose(old_test, new_test)
    np.testing.assert_array_equal(old_split_dates.to_numpy(), new_split_dates.to_numpy())

    old_train_raw, old_test_raw = legacy.split_train_testraw("TCS", data_dir)
    new_train_raw, new_test_raw = extracted_split_train_testraw("TCS", data_dir)
    np.testing.assert_allclose(old_train_raw, new_train_raw)
    np.testing.assert_allclose(old_test_raw, new_test_raw)

    old_train_raw, old_val_raw, old_test_raw, _ = legacy.split_train_val_testraw("TCS", data_dir)
    new_train_raw, new_val_raw, new_test_raw, _ = extracted_split_train_val_testraw("TCS", data_dir)
    np.testing.assert_allclose(old_train_raw, new_train_raw)
    np.testing.assert_allclose(old_val_raw, new_val_raw)
    np.testing.assert_allclose(old_test_raw, new_test_raw)


def test_metric_and_tensor_extraction_equivalent():
    torch.manual_seed(123)
    predicted = torch.randn(12)
    real = torch.randn(12)
    torch.testing.assert_close(extracted_combine_vectors(predicted[:4], real[:4]), legacy.combine_vectors(predicted[:4], real[:4]))
    torch.testing.assert_close(extracted_getPnL(predicted, real, 12), legacy.getPnL(predicted, real, 12))
    torch.testing.assert_close(extracted_getSR(predicted, real), legacy.getSR(predicted, real))


def test_gan_model_extraction_equivalent():
    mean = torch.tensor(0.01)
    std = torch.tensor(0.2)
    batch = 5
    seq_len = 10
    z_dim = 8
    hidden = 8

    torch.manual_seed(123)
    old_gen = legacy.Generator(z_dim, seq_len, hidden, 1, mean, std)
    torch.manual_seed(123)
    new_gen = ExtractedGenerator(z_dim, seq_len, hidden, 1, mean, std)
    for old_param, new_param in zip(old_gen.parameters(), new_gen.parameters()):
        torch.testing.assert_close(old_param, new_param)

    condition = torch.randn(1, batch, seq_len)
    noise = torch.randn(1, batch, z_dim)
    h0 = torch.zeros(1, batch, hidden)
    c0 = torch.zeros(1, batch, hidden)
    torch.testing.assert_close(old_gen(noise, condition, h0, c0), new_gen(noise, condition, h0, c0))

    torch.manual_seed(123)
    old_disc = legacy.Discriminator(seq_len + 1, hidden, mean, std)
    torch.manual_seed(123)
    new_disc = ExtractedDiscriminator(seq_len + 1, hidden, mean, std)
    for old_param, new_param in zip(old_disc.parameters(), new_disc.parameters()):
        torch.testing.assert_close(old_param, new_param)
    pair = torch.cat([condition, torch.randn(1, batch, 1)], dim=-1)
    torch.testing.assert_close(old_disc(pair, h0, c0), new_disc(pair, h0, c0))


def test_legacy_lstm_extraction_equivalent():
    mean = torch.tensor(0.01)
    std = torch.tensor(0.2)
    batch = 5
    seq_len = 10

    torch.manual_seed(123)
    old_lstm = legacy.LSTM(0, seq_len, 8, 1, mean, std)
    torch.manual_seed(123)
    new_lstm = ExtractedLegacyLSTM(0, seq_len, 8, 1, mean, std)
    for old_param, new_param in zip(old_lstm.parameters(), new_lstm.parameters()):
        torch.testing.assert_close(old_param, new_param)

    condition = torch.randn(1, batch, seq_len)
    h0 = torch.zeros(1, batch, 1)
    c0 = torch.zeros(1, batch, 1)
    torch.testing.assert_close(old_lstm(condition, h0, c0), new_lstm(condition, h0, c0))
