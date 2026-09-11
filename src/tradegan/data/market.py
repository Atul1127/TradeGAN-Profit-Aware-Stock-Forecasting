"""Market and benchmark lookup helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ETF_find(etflistloc, stock):
    data = pd.read_csv(etflistloc)
    out = np.array(data['ticker_y'][data['ticker_x'] == stock])[0]
    return out


__all__ = ["ETF_find"]
