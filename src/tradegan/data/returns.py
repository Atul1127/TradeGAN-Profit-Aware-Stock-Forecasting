"""Return construction helpers extracted from the original implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def excessreturns_closeonly(dataloc, stock, etf, plotcheck=False):
    s_df = pd.read_csv(dataloc + stock + ".csv")
    e_df = pd.read_csv(dataloc + etf + ".csv")
    dates_dt = pd.to_datetime(s_df['date'])
    d1 = pd.to_datetime("2022-01-01")
    smp = dates_dt < d1
    s_df = s_df[smp]
    e_df = e_df[smp]
    s_log = np.log(s_df['AdjClose'])
    e_log = np.log(e_df['AdjClose'])
    dates_dt = dates_dt[smp]
    s_ret = np.diff(s_log)
    e_ret = np.diff(e_log)
    excessret = s_ret - e_ret

    if plotcheck:
        plt.figure(stock + " price")
        plt.title(stock + " price")
        plt.plot(dates_dt, s_df['AdjClose'])
        plt.xlabel("date")
        plt.ylabel("price in USD")
        plt.show()
        plt.figure("Returns " + stock)
        plt.title("Returns " + stock)
        plt.plot(dates_dt[1:], s_ret, alpha=0.7, label='stock')
        plt.plot(dates_dt[1:], e_ret, alpha=0.7, label='etf')
        plt.plot(dates_dt[1:], excessret, alpha=0.7, label='excess return')
        plt.xlabel("date")
        plt.legend()
        plt.show()
    return excessret, dates_dt[1:]


def excessreturns(dataloc, stock, etf, plotcheck=False):
    cutoff_date = pd.Timestamp("2022-01-01")

    try:
        s_df = pd.read_csv(f"{dataloc}{stock}.csv", parse_dates=['date'])
    except FileNotFoundError:
        raise FileNotFoundError(f"Stock file '{dataloc}{stock}.csv' not found.")

    try:
        e_df = pd.read_csv(f"{dataloc}{etf}.csv", parse_dates=['date'])
    except FileNotFoundError:
        raise FileNotFoundError(f"ETF file '{dataloc}{etf}.csv' not found.")

    merged_df = pd.merge(
        s_df[s_df['date'] < cutoff_date],
        e_df[e_df['date'] < cutoff_date],
        on='date',
        suffixes=('_stock', '_etf')
    ).reset_index(drop=True)

    if merged_df.empty:
        raise ValueError(f"No overlapping dates found for stock '{stock}' and ETF '{etf}' before {cutoff_date}.")

    s_logclose = np.log(merged_df['AdjClose_stock'].values)
    e_logclose = np.log(merged_df['AdjClose_etf'].values)
    s_logopen = np.log(merged_df['AdjOpen_stock'].values)
    e_logopen = np.log(merged_df['AdjOpen_etf'].values)

    s_log = np.empty(2 * len(s_logclose))
    e_log = np.empty(2 * len(e_logclose))
    s_log[0::2] = s_logopen
    s_log[1::2] = s_logclose
    e_log[0::2] = e_logopen
    e_log[1::2] = e_logclose

    s_ret = np.diff(s_log)
    e_ret = np.diff(e_log)

    cap_value = 0.15
    s_ret = np.clip(s_ret, -cap_value, cap_value)
    e_ret = np.clip(e_ret, -cap_value, cap_value)
    excessret = s_ret - e_ret
    dates_dt = merged_df['date'].iloc[1:].reset_index(drop=True)

    if plotcheck:
        plt.figure(figsize=(14, 6))
        plt.plot(merged_df['date'], merged_df['AdjClose_stock'], label=f'{stock} AdjClose', color='blue')
        plt.title(f'{stock} Adjusted Close Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.figure(figsize=(14, 6))
        plt.plot(dates_dt, s_ret, alpha=0.7, label='Stock Returns', color='green')
        plt.plot(dates_dt, e_ret, alpha=0.7, label='ETF Returns', color='orange')
        plt.plot(dates_dt, excessret, alpha=0.7, label='Excess Returns', color='red')
        plt.title(f'Returns for {stock} vs {etf}')
        plt.xlabel('Date')
        plt.ylabel('Log Return')
        plt.legend()
        plt.grid(True)
        plt.show()

    return excessret, dates_dt


def rawreturns(dataloc, stock, plotcheck=False):
    s_df = pd.read_csv(dataloc + stock + ".csv")
    dates_dt = pd.to_datetime(s_df['date'])
    d1 = pd.to_datetime("2022-01-01")
    smp = dates_dt < d1
    s_df = s_df[smp]
    dates_dt = pd.to_datetime(s_df['date'])
    s_logclose = np.log(s_df['AdjClose'])
    s_logopen = np.log(s_df['AdjOpen'])
    s_log = np.zeros(2 * len(s_logclose))
    for i in range(len(s_logclose)):
        s_log[2 * i] = s_logopen[i]
        s_log[2 * i + 1] = s_logclose[i]
    s_ret = np.diff(s_log)
    s_ret[s_ret > 0.15] = 0.15
    s_ret[s_ret < -0.15] = -0.15
    dates_dt = pd.to_datetime(s_df['date'])

    if plotcheck:
        plt.figure(stock + " price")
        plt.title(stock + " price")
        plt.plot(dates_dt, s_df['AdjClose'])
        plt.xlabel("date")
        plt.ylabel("price in USD")
        plt.show()
        plt.figure("Returns " + stock)
        plt.title("Returns " + stock)
        plt.plot(range(len(s_ret)), s_ret)
        plt.legend()
        plt.show()
    return s_ret, dates_dt


__all__ = ["excessreturns_closeonly", "excessreturns", "rawreturns"]
