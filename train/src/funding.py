"""Bybit perpetual funding aligned to the daily open-to-open holding periods.

At every settlement Bybit charges `position_qty * mark_price * funding_rate`
to positions open at that instant: a positive rate is paid by longs and
received by shorts. The backtest holds a position from open t (00:00 UTC,
executed just after the daily close) to open t+1, so it pays every settlement
in (00:00 t, 00:00 t+1]; the 00:00 settlement on the rebalance day belongs to
the position held before the rebalance.

`factor[t, j]` is the funding paid per unit of open-t notional while holding
asset j during day t:

    sum over settlements s in the period of rate_s * price(s) / open_t

where the unobserved mark price at s is interpolated linearly between open t
and open t+1. Funding P&L of a held notional is `-notional * factor`.

`rate_sum[t, j]` is the plain sum of those rates. It is fully known at the
close of day t (the last settlement is at 00:00 t+1, before the rebalance), so
it may be used by signals formed at that close.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DAY_SECONDS = 86_400.0


def load_funding_factors(
    directory: Path,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    open_prices: np.ndarray,
) -> dict[str, Any]:
    """Return per-day funding factors; NaN marks days outside a symbol's funding coverage."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Funding directory {directory} does not exist; run scraper_bybit.py --funding")
    t_count, n_assets = open_prices.shape
    factor = np.full((t_count, n_assets), np.nan, np.float64)
    rate_sum = np.full((t_count, n_assets), np.nan, np.float64)
    date0 = pd.Timestamp(dates[0])
    files = 0
    settlements = 0

    for j, symbol in enumerate(symbols):
        path = directory / f"{symbol}_funding.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        offset = (pd.to_datetime(frame["timestamp"]) - date0).dt.total_seconds().to_numpy(np.float64)
        day = np.ceil(offset / DAY_SECONDS).astype(np.int64) - 1
        keep = (day >= 0) & (day < t_count)
        if not keep.any():
            continue
        day = day[keep]
        fraction = offset[keep] / DAY_SECONDS - day
        rate = frame["funding_rate"].to_numpy(np.float64)[keep]

        start = open_prices[day, j].astype(np.float64)
        following = np.full(day.size, np.nan)
        inside = day + 1 < t_count
        following[inside] = open_prices[day[inside] + 1, j]
        priced = np.isfinite(start) & np.isfinite(following) & (start > 0)
        price_ratio = np.where(priced, 1.0 + fraction * (following / np.where(priced, start, 1.0) - 1.0), 1.0)

        column = factor[:, j]
        column[day.min() : day.max() + 1] = 0.0
        np.add.at(column, day, rate * price_ratio)
        rates = rate_sum[:, j]
        rates[day.min() : day.max() + 1] = 0.0
        np.add.at(rates, day, rate)
        files += 1
        settlements += int(day.size)

    return {"factor": factor, "rate_sum": rate_sum, "files": files, "settlements": settlements, "directory": str(directory)}
