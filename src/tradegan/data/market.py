"""Market and benchmark lookup helpers."""

from __future__ import annotations

import pandas as pd


def ETF_find(etflistloc, stock):
    data = pd.read_csv(etflistloc)
    matches = data.loc[data["ticker_x"] == stock, "ticker_y"]
    if matches.empty:
        raise KeyError(f"No ETF mapping found for ticker {stock!r}")
    return matches.iloc[0]


__all__ = ["ETF_find"]
