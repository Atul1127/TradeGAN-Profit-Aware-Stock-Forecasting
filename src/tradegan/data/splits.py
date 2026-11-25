"""Train/validation/test window construction extracted from legacy.py."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tradegan.data.market import ETF_find
from tradegan.data.returns import excessreturns, rawreturns


def _validate_split_args(tr: float, vl: float, h: int, l: int, pred: int) -> None:
    if not 0 < tr < 1:
        raise ValueError("tr must be between 0 and 1")
    if not 0 <= vl < 1:
        raise ValueError("vl must be between 0 and 1")
    if tr + vl >= 1:
        raise ValueError("tr + vl must be less than 1")
    if h <= 0 or l <= 0 or pred <= 0:
        raise ValueError("h, l and pred must be positive")


def _window_data(series: np.ndarray, l: int, pred: int, h: int, split_name: str) -> np.ndarray:
    count = int((len(series) - l - pred) / h) + 1
    if count <= 0:
        raise ValueError(
            f"Not enough observations for {split_name}: need more than l + pred "
            f"({l + pred}), got {len(series)}"
        )
    out = np.zeros((count, l + pred))
    for i in tqdm(range(count)):
        start = i * h
        out[i, :] = series[start : start + l + pred]
    return out


def split_sizes(n_obs: int, tr: float = 0.8, vl: float = 0.1) -> tuple[int, int]:
    """Number of observations in the training and validation splits."""
    return int(tr * n_obs), int(vl * n_obs)


def target_start_indices(n_obs: int, tr: float = 0.8, vl: float = 0.1, l: int = 10) -> tuple[int, int]:
    """Global series index of the first validation and first test target.

    Windows never cross a split boundary, so the first target of a split sits
    ``l`` observations after the split starts. Even indices are intraday
    returns and odd indices are overnight returns (see ``data.returns``).
    """
    n_tr, n_vl = split_sizes(n_obs, tr, vl)
    return n_tr + l, n_tr + n_vl + l


def _windows_with_raw_targets(clipped: np.ndarray, raw: np.ndarray, l: int, pred: int, h: int,
                              split_name: str) -> np.ndarray:
    """Windows whose inputs are clipped but whose targets are the raw returns."""
    windows = _window_data(clipped, l, pred, h, split_name)
    windows[:, l:] = _window_data(raw, l, pred, h, split_name)[:, l:]
    return windows


def _split_series(clipped: np.ndarray, raw: np.ndarray, tr: float, vl: float, h: int, l: int, pred: int):
    n_tr, n_vl = split_sizes(len(clipped), tr, vl)
    # Training uses clipped targets so a bad tick cannot dominate the loss.
    train_data = _window_data(clipped[:n_tr], l, pred, h, "training")
    # Validation/test score the model on raw returns: that is the PnL it would earn.
    val_data = _windows_with_raw_targets(
        clipped[n_tr : n_tr + n_vl], raw[n_tr : n_tr + n_vl], l, pred, h, "validation"
    )
    test_data = _windows_with_raw_targets(
        clipped[n_tr + n_vl :], raw[n_tr + n_vl :], l, pred, h, "test"
    )
    return train_data, val_data, test_data, n_tr, n_vl


def split_train_val_test(stock, dataloc, etflistloc, tr=0.8, vl=0.1, h=1, l=10, pred=1, plotcheck=False):
    _validate_split_args(tr, vl, h, l, pred)
    etf = ETF_find(etflistloc, stock)
    excess_returns, raw_excess, dates_dt = excessreturns(dataloc, stock, etf, plotcheck, return_raw=True)
    train_data, val_data, test_data, N_tr, N_vl = _split_series(
        excess_returns, raw_excess, tr, vl, h, l, pred
    )

    if plotcheck:
        plt.figure("Excess returns")
        plt.plot(dates_dt, excess_returns)
        plt.title(stock + " excess returns")
        plt.axvline(x=dates_dt[N_tr], color="red")
        plt.axvline(x=dates_dt[N_tr + N_vl], color="red")
        plt.show()
    return train_data, val_data, test_data, dates_dt


def split_train_testraw(stock, dataloc, tr=0.8, vl=0.1, h=1, l=10, pred=1, plotcheck=False):
    _validate_split_args(tr, vl, h, l, pred)
    excess_returns, _dates_dt = rawreturns(dataloc, stock, plotcheck)
    N = len(excess_returns)
    N_tr = int(tr * N) + int(vl * N)
    N_tst = N - N_tr
    train_data = _window_data(excess_returns[:N_tr], l, pred, h, "training")
    test_data = _window_data(excess_returns[N_tr:], l, pred, h, "test")
    return train_data, test_data


def split_train_val_testraw(stock, dataloc, tr=0.8, vl=0.1, h=1, l=10, pred=1, plotcheck=False):
    _validate_split_args(tr, vl, h, l, pred)
    excess_returns, raw_returns, dates_dt = rawreturns(dataloc, stock, plotcheck, return_raw=True)
    train_data, val_data, test_data, N_tr, N_vl = _split_series(
        excess_returns, raw_returns, tr, vl, h, l, pred
    )

    if plotcheck:
        plt.figure("returns")
        plt.plot(dates_dt, excess_returns)
        plt.title(stock + " returns")
        plt.axvline(x=dates_dt[N_tr], color="red")
        plt.axvline(x=dates_dt[N_tr + N_vl], color="red")
        plt.show()
    return train_data, val_data, test_data, dates_dt


__all__ = [
    "split_sizes",
    "split_train_val_test",
    "split_train_testraw",
    "split_train_val_testraw",
    "target_start_indices",
]
