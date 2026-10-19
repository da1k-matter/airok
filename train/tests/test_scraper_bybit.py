import asyncio

import pandas as pd

from scraper_bybit import (
    MINUTE_MS,
    Instrument,
    download_funding,
    download_klines,
    plan_windows,
)

HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS


class FakeClient:
    """Serves synthetic Bybit pages: newest first, capped at `limit`, inclusive time range."""

    def __init__(self, funding_stamps: list[int], kline_stamps: list[int]) -> None:
        self.funding_stamps = funding_stamps
        self.kline_stamps = kline_stamps
        self.requests = 0
        self.calls: list[tuple[str, dict]] = []

    async def get(self, endpoint: str, params: dict) -> dict:
        self.requests += 1
        self.calls.append((endpoint, params))
        if endpoint == "funding/history":
            low, high, limit = params["startTime"], params["endTime"], params["limit"]
            page = sorted((s for s in self.funding_stamps if low <= s <= high), reverse=True)[:limit]
            return {"list": [{"fundingRateTimestamp": str(s), "fundingRate": "0.0001"} for s in page]}
        low, high, limit = params["start"], params["end"], params["limit"]
        page = sorted((s for s in self.kline_stamps if low <= s <= high), reverse=True)[:limit]
        return {"list": [[str(s), "1", "2", "0.5", "1.5", "10", "15"] for s in page]}


def test_plan_windows_are_adjacent_and_cover_the_range():
    windows = plan_windows(0, 2_499, step_ms=1, per_request=1_000)
    assert windows == [(1_500, 2_499), (500, 1_499), (0, 499)]
    assert plan_windows(10, 9, 1, 5) == []


def test_funding_refills_windows_when_history_used_a_shorter_interval(tmp_path):
    start, end = 0, 100 * DAY_MS - 1
    # Hourly settlements for the first 50 days, 8-hourly afterwards; the instrument now reports 8h.
    stamps = list(range(0, 50 * DAY_MS, HOUR_MS)) + list(range(50 * DAY_MS, 100 * DAY_MS, 8 * HOUR_MS))
    client = FakeClient(stamps, [])
    instrument = Instrument("XUSDT", "Trading", launch_ms=0, funding_interval_minutes=480)

    report = asyncio.run(
        download_funding(
            client, [("X", instrument)], start, end, tmp_path,
            skip_existing=False, concurrency=4, show_progress=False,
        )
    )

    saved = pd.read_csv(tmp_path / "X_funding.csv")
    assert report.results[0].rows == len(stamps) == len(saved)
    assert len(client.calls) > len(plan_windows(start, end, 8 * HOUR_MS, 200))


def test_klines_keep_only_requested_range_and_report_gaps(tmp_path):
    start, end = 10 * DAY_MS, 1_209 * DAY_MS
    stamps = [s for s in range(0, 1_300 * DAY_MS, DAY_MS) if s != 500 * DAY_MS]
    client = FakeClient([], stamps)
    instrument = Instrument("XUSDT", "Trading", launch_ms=0, funding_interval_minutes=480)

    report = asyncio.run(
        download_klines(
            client, [("X", instrument)], "D", start, end, tmp_path,
            skip_existing=False, concurrency=4, show_progress=False,
        )
    )

    frame = pd.read_csv(tmp_path / "X_1440.csv")
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert frame["timestamp"].iloc[0] == "1970-01-11"
    assert len(frame) == 1_199
    assert report.results[0].missing == 1
    assert len(client.calls) == 2
