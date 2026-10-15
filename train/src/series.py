"""Daily Bybit market series aligned to the panel so that row t holds what is known at close t.

Snapshot series (open interest, long/short account ratio) are stamped at 00:00
UTC: the value stamped 00:00 of day t+1 is the state at the close of day t.
Bar series (premium-index candles) are stamped with their own day.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAY_SECONDS = 86_400.0


def load_daily_series(
    directory: Path,
    kind: str,
    column: str,
    dates: pd.DatetimeIndex,
    symbols: list[str],
    snapshot: bool,
) -> np.ndarray:
    """Return a dates x symbols array of `column` from `<directory>/<SYMBOL>_<kind>.csv`."""
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} does not exist; run scraper_bybit.py with the matching series flag")
    t_count = len(dates)
    output = np.full((t_count, len(symbols)), np.nan, np.float64)
    date0 = pd.Timestamp(dates[0])
    for j, symbol in enumerate(symbols):
        path = directory / f"{symbol}_{kind}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["timestamp", column])
        offset = (pd.to_datetime(frame["timestamp"]) - date0).dt.total_seconds().to_numpy(np.float64)
        row = np.round(offset / DAY_SECONDS).astype(np.int64) - (1 if snapshot else 0)
        keep = (row >= 0) & (row < t_count)
        output[row[keep], j] = frame[column].to_numpy(np.float64)[keep]
    return output
