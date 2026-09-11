"""Return construction helpers extracted from the original implementation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_price_frame(dataloc, ticker: str) -> pd.DataFrame:
    path = Path(dataloc) / f"{ticker}.csv"
    try:
        frame = pd.read_csv(path, parse_dates=["date"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Price file '{path}' not found.") from exc

    required = {"date", "AdjOpen", "AdjClose"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Price file '{path}' is missing columns: {sorted(missing)}")

    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    frame = frame.dropna(subset=["date", "AdjOpen", "AdjClose"])
    if (frame[["AdjOpen", "AdjClose"]] <= 0).any().any():
        raise ValueError(f"Price file '{path}' contains non-positive prices.")
    return frame


def excessreturns_closeonly(dataloc, stock, etf, plotcheck=False):
    s_df = _load_price_frame(dataloc, stock)
    e_df = _load_price_frame(dataloc, etf)
    cutoff_date = pd.Timestamp("2022-01-01")

    merged_df = pd.merge(
        s_df[s_df["date"] < cutoff_date],
        e_df[e_df["date"] < cutoff_date],
        on="date",
        suffixes=("_stock", "_etf"),
    )
    if len(merged_df) < 2:
        raise ValueError(f"Not enough overlapping observations for '{stock}' and '{etf}'.")

    s_ret = np.diff(np.log(merged_df["AdjClose_stock"].to_numpy()))
    e_ret = np.diff(np.log(merged_df["AdjClose_etf"].to_numpy()))
    excessret = s_ret - e_ret
    dates_dt = merged_df["date"].iloc[1:].reset_index(drop=True)

    if plotcheck:
        plt.figure(f"{stock} price")
        plt.title(f"{stock} price")
        plt.plot(merged_df["date"], merged_df["AdjClose_stock"])
        plt.xlabel("date")
        plt.ylabel("price in USD")
        plt.show()
        plt.figure(f"Returns {stock}")
        plt.title(f"Returns {stock}")
        plt.plot(dates_dt, s_ret, alpha=0.7, label="stock")
        plt.plot(dates_dt, e_ret, alpha=0.7, label="etf")
        plt.plot(dates_dt, excessret, alpha=0.7, label="excess return")
        plt.xlabel("date")
        plt.legend()
        plt.show()
    return excessret, dates_dt


def excessreturns(dataloc, stock, etf, plotcheck=False):
    cutoff_date = pd.Timestamp("2022-01-01")
    s_df = _load_price_frame(dataloc, stock)
    e_df = _load_price_frame(dataloc, etf)

    merged_df = pd.merge(
        s_df[s_df["date"] < cutoff_date],
        e_df[e_df["date"] < cutoff_date],
        on="date",
        suffixes=("_stock", "_etf"),
    )

    if merged_df.empty:
        raise ValueError(f"No overlapping dates found for stock '{stock}' and ETF '{etf}' before {cutoff_date}.")

    s_logclose = np.log(merged_df["AdjClose_stock"].to_numpy())
    e_logclose = np.log(merged_df["AdjClose_etf"].to_numpy())
    s_logopen = np.log(merged_df["AdjOpen_stock"].to_numpy())
    e_logopen = np.log(merged_df["AdjOpen_etf"].to_numpy())

    s_log = np.empty(2 * len(s_logclose))
    e_log = np.empty(2 * len(e_logclose))
    s_log[0::2] = s_logopen
    s_log[1::2] = s_logclose
    e_log[0::2] = e_logopen
    e_log[1::2] = e_logclose

    s_ret = np.clip(np.diff(s_log), -0.15, 0.15)
    e_ret = np.clip(np.diff(e_log), -0.15, 0.15)
    excessret = s_ret - e_ret
    dates_dt = merged_df["date"].iloc[1:].reset_index(drop=True)

    if plotcheck:
        plt.figure(figsize=(14, 6))
        plt.plot(merged_df["date"], merged_df["AdjClose_stock"], label=f"{stock} AdjClose")
        plt.title(f"{stock} Adjusted Close Price")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.figure(figsize=(14, 6))
        plt.plot(dates_dt, s_ret, alpha=0.7, label="Stock Returns")
        plt.plot(dates_dt, e_ret, alpha=0.7, label="ETF Returns")
        plt.plot(dates_dt, excessret, alpha=0.7, label="Excess Returns")
        plt.title(f"Returns for {stock} vs {etf}")
        plt.xlabel("Date")
        plt.ylabel("Log Return")
        plt.legend()
        plt.grid(True)
        plt.show()

    return excessret, dates_dt


def rawreturns(dataloc, stock, plotcheck=False):
    s_df = _load_price_frame(dataloc, stock)
    s_df = s_df[s_df["date"] < pd.Timestamp("2022-01-01")].reset_index(drop=True)
    if len(s_df) < 2:
        raise ValueError(f"Not enough observations for '{stock}' to construct returns.")

    s_logclose = np.log(s_df["AdjClose"].to_numpy())
    s_logopen = np.log(s_df["AdjOpen"].to_numpy())
    s_log = np.empty(2 * len(s_df))
    s_log[0::2] = s_logopen
    s_log[1::2] = s_logclose
    s_ret = np.clip(np.diff(s_log), -0.15, 0.15)

    # Each return is an interval endpoint: close on day i, or open on day i+1.
    dates_dt = pd.Series(np.repeat(s_df["date"].to_numpy(), 2)[1:])

    if plotcheck:
        plt.figure(f"{stock} price")
        plt.title(f"{stock} price")
        plt.plot(s_df["date"], s_df["AdjClose"])
        plt.xlabel("date")
        plt.ylabel("price in USD")
        plt.show()
        plt.figure(f"Returns {stock}")
        plt.title(f"Returns {stock}")
        plt.plot(dates_dt, s_ret)
        plt.xlabel("date")
        plt.ylabel("log return")
        plt.show()
    return s_ret, dates_dt.reset_index(drop=True)


__all__ = ["excessreturns_closeonly", "excessreturns", "rawreturns"]
