"""Append confirmed Bybit daily candles to the local training panel."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from data import REQUIRED_COLUMNS, list_data_files

KLINES_URL = "https://api.bybit.com/v5/market/kline"
DAY_MS = 86_400_000


@dataclass(frozen=True)
class UpdateResult:
    path: Path
    appended: int
    available: int
    requested: int
    error: str | None = None


def fetch_daily_klines(symbol: str, start: date, end: date) -> list[dict[str, object]]:
    """Fetch complete UTC daily linear-perpetual candles in ascending order."""
    query = {
        "category": "linear",
        "symbol": f"{symbol}USDT",
        "interval": "D",
        "start": str(int(datetime.combine(start, datetime.min.time(), UTC).timestamp() * 1000)),
        "end": str(int(datetime.combine(end + timedelta(days=1), datetime.min.time(), UTC).timestamp() * 1000 - 1)),
        "limit": "1000",
    }
    with urlopen(f"{KLINES_URL}?{urlencode(query)}", timeout=30) as response:
        payload: dict[str, Any] = json.load(response)
    if int(payload.get("retCode", -1)) != 0:
        raise RuntimeError(str(payload.get("retMsg", "unknown Bybit error")))
    rows = payload.get("result", {}).get("list", [])
    candles: list[dict[str, object]] = []
    for row in rows:
        opened = datetime.fromtimestamp(int(row[0]) / 1000, UTC).date()
        if start <= opened <= end:
            candles.append(
                {
                    "timestamp": opened.isoformat(),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
    return sorted(candles, key=lambda candle: str(candle["timestamp"]))


def update_file(path: Path, end: date) -> UpdateResult:
    frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    existing = pd.to_datetime(frame["timestamp"], errors="raise").dt.date
    start = existing.max() + timedelta(days=1)
    if start > end:
        return UpdateResult(path=path, appended=0, available=0, requested=0)
    requested = (end - start).days + 1
    symbol = path.name.removesuffix("_1440.csv")
    try:
        candles = fetch_daily_klines(symbol, start, end)
    except Exception as exc:
        return UpdateResult(path=path, appended=0, available=0, requested=requested, error=str(exc))
    if not candles:
        return UpdateResult(path=path, appended=0, available=0, requested=requested)
    additions = pd.DataFrame(candles, columns=REQUIRED_COLUMNS)
    merged = pd.concat((frame, additions), ignore_index=True).drop_duplicates("timestamp", keep="last")
    merged = merged.sort_values("timestamp", kind="stable")
    expected = pd.date_range(start, end, freq="D").date
    available = int(merged["timestamp"].isin([item.isoformat() for item in expected]).sum())
    merged.to_csv(path, index=False)
    return UpdateResult(path=path, appended=len(additions), available=available, requested=requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append confirmed Bybit daily candles to local CSV files")
    parser.add_argument("--data-dir", type=Path, default=Path("data/1d"))
    parser.add_argument("--through", type=date.fromisoformat, help="Last confirmed UTC date; default is yesterday UTC")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    end = args.through or (datetime.now(UTC).date() - timedelta(days=1))
    files = list_data_files(args.data_dir)
    results: list[UpdateResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(update_file, path, end) for path in files]
        for future in as_completed(futures):
            results.append(future.result())
    failures = sorted(result for result in results if result.error)
    appended = sum(result.appended for result in results)
    updated = sum(result.appended > 0 for result in results)
    unavailable = sorted(
        result.path.name
        for result in results
        if not result.error and result.requested > 0 and result.appended == 0
    )
    incomplete = sorted(
        f"{result.path.name} ({result.available}/{result.requested})"
        for result in results
        if not result.error and result.requested > 0 and 0 < result.available < result.requested
    )
    print(f"updated_files={updated} appended_candles={appended} through={end.isoformat()}")
    if unavailable:
        print(f"no_new_candles={len(unavailable)}: {', '.join(unavailable)}")
    if incomplete:
        print(f"incomplete_candles={len(incomplete)}: {', '.join(incomplete)}")
    if failures:
        raise SystemExit("failed downloads:\n" + "\n".join(f"{item.path.name}: {item.error}" for item in failures))


if __name__ == "__main__":
    main()
