"""Train/validation/test window construction extracted from legacy.py."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from tradegan.data.market import ETF_find
from tradegan.data.returns import excessreturns, rawreturns


def split_train_val_test(stock, dataloc, etflistloc, tr=0.8, vl=0.1, h=1, l=10, pred=1, plotcheck=False):
    etf = ETF_find(etflistloc, stock)
    excess_returns, dates_dt = excessreturns(dataloc, stock, etf, plotcheck)
    N = len(excess_returns)
    N_tr = int(tr * N)
    N_vl = int(vl * N)
    N_tst = N - N_tr - N_vl
    train_sr = excess_returns[0:N_tr]
    val_sr = excess_returns[N_tr:N_tr + N_vl]
    test_sr = excess_returns[N_tr + N_vl:]

    n = int((N_tr - l - pred) / h) + 1
    train_data = np.zeros(shape=(n, l + pred))
    l_tot = 0
    for i in tqdm(range(n)):
        train_data[i, :] = train_sr[l_tot:l_tot + l + pred]
        l_tot += h

    n = int((N_vl - l - pred) / h) + 1
    val_data = np.zeros(shape=(n, l + pred))
    l_tot = 0
    for i in tqdm(range(n)):
        val_data[i, :] = val_sr[l_tot:l_tot + l + pred]
        l_tot += h

    n = int((N_tst - l - pred) / h) + 1
    test_data = np.zeros(shape=(n, l + pred))
    l_tot = 0
    for i in tqdm(range(n)):
        test_data[i, :] = test_sr[l_tot:l_tot + l + pred]
        l_tot += h

    return train_data, val_data, test_data, dates_dt


def split_train_testraw(stock, dataloc, tr=0.8, vl=0.1, h=1, l=10, pred=1, plotcheck=False):
    returns, dates_dt = rawreturns(dataloc, stock, plotcheck)
    N = len(returns)
    N_tr = int(tr * N) + int(vl * N)
    N_tst = N - N_tr
    train_sr = returns[0:N_tr]
    test_sr = returns[N_tr:]

    n = int((N_tr - l - pred) / h) + 1
    train_data = np.zeros(shape=(n, l + pred))
    l_tot = 0
    for i in tqdm(range(n)):
        train_data[i, :] = train_sr[l_tot:l_tot + l + pred]
        l_tot += h

    n = int((N_tst - l - pred) / h) + 1
    test_data = np.zeros(shape=(n, l + pred))
    l_tot = 0
    for i in tqdm(range(n)):
        test_data[i, :] = test_sr[l_tot:l_tot + l + pred]
        l_tot += h

    return train_data, test_data


def split_train_val_testraw(stock, dataloc, tr=0.8, vl=0.1, h=1, l=10, pred=1, plotcheck=False):
    returns, dates_dt = rawreturns(dataloc, stock, plotcheck)
    N = len(returns)
    N_tr = int(tr * N)
    N_vl = int(vl * N)
    N_tst = N - N_tr - N_vl
    train_sr = returns[0:N_tr]
    val_sr = returns[N_tr:N_tr + N_vl]
    test_sr = returns[N_tr + N_vl:]

    n = int((N_tr - l - pred) / h) + 1
    train_data = np.zeros(shape=(n, l + pred))
    l_tot = 0
    for i in tqdm(range(n)):
        train_data[i, :] = train_sr[l_tot:l_tot + l + pred]
        l_tot += h

    n = int((N_vl - l - pred) / h) + 1
    val_data = np.zeros(shape=(n, l + pred))
    l_tot = 0
    for i in tqdm(range(n)):
        val_data[i, :] = val_sr[l_tot:l_tot + l + pred]
        l_tot += h

    n = int((N_tst - l - pred) / h) + 1
    test_data = np.zeros(shape=(n, l + pred))
    l_tot = 0
    for i in tqdm(range(n)):
        test_data[i, :] = test_sr[l_tot:l_tot + l + pred]
        l_tot += h

    return train_data, val_data, test_data, dates_dt


__all__ = ["split_train_val_test", "split_train_testraw", "split_train_val_testraw"]
