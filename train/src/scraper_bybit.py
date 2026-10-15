"""Download Bybit USDT linear-perpetual candles and, optionally, funding history.

Ported from quasar/tools/scraper_bybit.py and reduced to public market data:
no pybit session, credentials, order or WebSocket code. Every request goes
through one adaptive rate limiter shared by all symbols, and each symbol's
history is split into fixed request windows that are fetched concurrently.

Candles are written exactly in the training/live panel format
(`<SYMBOL>_<minutes>.csv`: timestamp, open, high, low, close, volume) and only
confirmed bars through the requested UTC day are kept.

Funding history (`--funding`) is written to `<SYMBOL>_funding.csv`
(timestamp, funding_rate) with one row per settlement. Bybit returns at most
200 settlements per request, newest first. Windows are sized from the current
`fundingInterval`; a window that returns a full page while older settlements
remain (the interval was shorter in the past) is refilled recursively, so the
result is complete regardless of historical interval changes.

    python src/scraper_bybit.py --data-dir data/1d --days 1000 --symbols-from data/1d --funding
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from tqdm import tqdm

API_ROOT = "https://api.bybit.com/v5/market"
CATEGORY = "linear"
QUOTE = "USDT"
KLINE_LIMIT = 1000
FUNDING_LIMIT = 200
MINUTE_MS = 60_000
DAY_MS = 86_400_000
DEFAULT_FUNDING_INTERVAL_MINUTES = 480
KLINE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
FUNDING_COLUMNS = ("timestamp", "funding_rate")
INTERVAL_MINUTES = {
    "1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60,
    "120": 120, "240": 240, "360": 360, "720": 720, "D": 1440,
}
# 10006: too many visits; 10018: IP rate limit exceeded.
RATE_LIMIT_CODES = frozenset({10006, 10018})


class BybitError(RuntimeError):
    pass


class _Retryable(Exception):
    pass


class AdaptiveRateLimiter:
    """Global async request pacer that backs off on Bybit rate-limit responses."""

    def __init__(
        self,
        target_rps: float,
        *,
        min_rps: float = 2.0,
        backoff: float = 0.7,
        recovery: float = 1.02,
        cooldown_s: float = 2.0,
        forbidden_block_s: float = 60.0,
    ) -> None:
        if target_rps <= 0:
            raise ValueError("target_rps must be positive")
        self.target_rps = float(target_rps)
        self.rps = float(target_rps)
        self.min_rps = min(float(min_rps), self.target_rps)
        self.backoff = backoff
        self.recovery = recovery
        self.cooldown_s = cooldown_s
        self.forbidden_block_s = forbidden_block_s
        self.rate_limit_hits = 0
        self.forbidden_hits = 0
        self._next_slot = 0.0
        self._block_until = 0.0
        self._last_limit = float("-inf")
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot, self._block_until)
            self._next_slot = slot + 1.0 / self.rps
        if slot > now:
            await asyncio.sleep(slot - now)

    def on_success(self) -> None:
        if self.rps < self.target_rps and time.monotonic() - self._last_limit >= self.cooldown_s:
            self.rps = min(self.target_rps, self.rps * self.recovery)

    def on_rate_limit(self) -> None:
        self.rate_limit_hits += 1
        now = time.monotonic()
        self._last_limit = now
        self.rps = max(self.min_rps, self.rps * self.backoff)
        self._block_until = max(self._block_until, now + 1.0)

    def on_forbidden(self) -> None:
        self.forbidden_hits += 1
        now = time.monotonic()
        self._last_limit = now
        self.rps = self.min_rps
        self._block_until = max(self._block_until, now + self.forbidden_block_s)


class BybitPublicClient:
    def __init__(self, session: aiohttp.ClientSession, limiter: AdaptiveRateLimiter, retries: int = 8) -> None:
        self.session = session
        self.limiter = limiter
        self.retries = retries
        self.requests = 0
        self.retried = 0

    async def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = {key: str(value) for key, value in params.items()}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            if attempt:
                self.retried += 1
                await asyncio.sleep(min(8.0, 0.25 * 2**attempt) * (0.5 + random.random()))
            await self.limiter.wait()
            try:
                async with self.session.get(f"{API_ROOT}/{endpoint}", params=query) as response:
                    if response.status == 403:
                        self.limiter.on_forbidden()
                        raise _Retryable("HTTP 403 (IP temporarily blocked)")
                    if response.status == 429:
                        self.limiter.on_rate_limit()
                        raise _Retryable("HTTP 429")
                    if response.status >= 500:
                        raise _Retryable(f"HTTP {response.status}")
                    if response.status != 200:
                        raise BybitError(f"{endpoint} {query}: HTTP {response.status}")
                    payload = await response.json(content_type=None)
            except (TimeoutError, aiohttp.ClientError, _Retryable) as exc:
                last_error = exc
                continue
            self.requests += 1
            code = int(payload.get("retCode", -1))
            if code == 0:
                self.limiter.on_success()
                result = payload.get("result")
                return result if isinstance(result, dict) else {}
            if code in RATE_LIMIT_CODES:
                self.limiter.on_rate_limit()
                last_error = BybitError(f"retCode={code} {payload.get('retMsg')}")
                continue
            raise BybitError(f"{endpoint} {query}: retCode={code} {payload.get('retMsg')}")
        raise BybitError(f"{endpoint} {query}: failed after {self.retries} attempts: {last_error}")


@dataclass(frozen=True)
class Instrument:
    symbol: str
    status: str
    launch_ms: int
    funding_interval_minutes: int

    @property
    def base(self) -> str:
        return self.symbol.removesuffix(QUOTE)


@dataclass
class SymbolResult:
    base: str
    saved: bool = False
    rows: int = 0
    first: str | None = None
    last: str | None = None
    missing: int = 0
    note: str | None = None
    error: str | None = None


@dataclass
class PhaseReport:
    name: str
    elapsed_s: float = 0.0
    requests: int = 0
    results: list[SymbolResult] = field(default_factory=list)


async def fetch_instruments(client: BybitPublicClient) -> dict[str, Instrument]:
    instruments: dict[str, Instrument] = {}
    cursor = ""
    while True:
        params: dict[str, Any] = {"category": CATEGORY, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        result = await client.get("instruments-info", params)
        for item in result.get("list") or []:
            if item.get("contractType") != "LinearPerpetual" or item.get("quoteCoin") != QUOTE:
                continue
            if item.get("settleCoin") != QUOTE:
                continue
            instruments[str(item["symbol"])] = Instrument(
                symbol=str(item["symbol"]),
                status=str(item.get("status", "")),
                launch_ms=int(item.get("launchTime") or 0),
                funding_interval_minutes=int(item.get("fundingInterval") or DEFAULT_FUNDING_INTERVAL_MINUTES),
            )
        cursor = str(result.get("nextPageCursor") or "")
        if not cursor:
            return instruments


def plan_windows(start_ms: int, end_ms: int, step_ms: int, per_request: int) -> list[tuple[int, int]]:
    """Split an inclusive range into adjacent windows holding at most `per_request` steps, newest first."""
    if start_ms > end_ms:
        return []
    span = step_ms * per_request
    windows: list[tuple[int, int]] = []
    high = end_ms
    while high >= start_ms:
        low = max(start_ms, high - span + 1)
        windows.append((low, high))
        high = low - 1
    return windows


def _floor(value: int, step: int) -> int:
    return value - value % step


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _format_timestamps(ms: Iterable[int], daily: bool) -> list[str]:
    stamps = pd.to_datetime(list(ms), unit="ms", utc=True)
    return list(stamps.strftime("%Y-%m-%d" if daily else "%Y-%m-%d %H:%M:%S"))


class _Progress:
    """tqdm wrapper whose total can grow when windows are refilled."""

    def __init__(self, total: int, desc: str, client: BybitPublicClient, enabled: bool) -> None:
        self.bar = tqdm(total=total, desc=desc, unit="req", disable=not enabled, dynamic_ncols=True)
        self.client = client
        self.started = time.monotonic()
        self.base_requests = client.requests
        self.symbols_done = 0
        self.symbol_count = 0
        self.errors = 0

    def request_done(self) -> None:
        self.bar.update(1)
        if self.bar.n % 10 == 0:
            self._postfix()

    def add_requests(self, count: int) -> None:
        self.bar.total += count
        self.bar.refresh()

    def skip(self, count: int) -> None:
        self.bar.update(count)

    def symbol_done(self, error: bool = False) -> None:
        self.symbols_done += 1
        self.errors += int(error)
        self._postfix()

    def _postfix(self) -> None:
        elapsed = max(1e-3, time.monotonic() - self.started)
        self.bar.set_postfix(
            sym=f"{self.symbols_done}/{self.symbol_count}",
            rps=f"{(self.client.requests - self.base_requests) / elapsed:.0f}",
            err=self.errors,
            refresh=False,
        )

    def close(self) -> None:
        self._postfix()
        self.bar.close()


async def download_klines(
    client: BybitPublicClient,
    targets: list[tuple[str, Instrument | None]],
    interval: str,
    start_ms: int,
    end_ms: int,
    output_dir: Path,
    *,
    skip_existing: bool,
    concurrency: int,
    show_progress: bool,
) -> PhaseReport:
    minutes = INTERVAL_MINUTES[interval]
    step = minutes * MINUTE_MS
    daily = minutes == 1440
    output_dir.mkdir(parents=True, exist_ok=True)
    report = PhaseReport(name=f"klines {interval}")
    requests_before = client.requests
    started = time.monotonic()

    plans: list[tuple[str, Instrument | None, list[tuple[int, int]], int]] = []
    for base, instrument in targets:
        first = max(start_ms, _floor(instrument.launch_ms, step)) if instrument else start_ms
        plans.append((base, instrument, plan_windows(first, end_ms, step, KLINE_LIMIT), first))

    progress = _Progress(sum(len(plan[2]) for plan in plans), f"klines {interval}", client, show_progress)
    progress.symbol_count = len(plans)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(base: str, instrument: Instrument | None, windows: list[tuple[int, int]], first: int) -> SymbolResult:
        result = SymbolResult(base=base)
        path = output_dir / f"{base}_{minutes}.csv"
        if skip_existing and path.exists():
            progress.skip(len(windows))
            result.note = "exists"
            progress.symbol_done()
            return result
        if not windows:
            result.note = "listed after range"
            progress.symbol_done()
            return result
        async with semaphore:
            rows: dict[int, list[Any]] = {}
            completed = 0

            async def fetch(window: tuple[int, int]) -> None:
                nonlocal completed
                payload = await client.get(
                    "kline",
                    {
                        "category": CATEGORY,
                        "symbol": f"{base}{QUOTE}",
                        "interval": interval,
                        "start": window[0],
                        "end": window[1],
                        "limit": KLINE_LIMIT,
                    },
                )
                completed += 1
                progress.request_done()
                for item in payload.get("list") or []:
                    opened = int(item[0])
                    if start_ms <= opened <= end_ms:
                        rows[opened] = item

            tasks = [asyncio.create_task(fetch(window)) for window in windows]
            try:
                await asyncio.gather(*tasks)
            except Exception as exc:
                for task in tasks:
                    task.cancel()
                progress.skip(len(windows) - completed)
                result.error = str(exc)
                progress.symbol_done(error=True)
                return result

        if not rows:
            result.note = "no candles"
            progress.symbol_done()
            return result
        stamps = sorted(rows)
        frame = pd.DataFrame(
            {
                "timestamp": _format_timestamps(stamps, daily),
                **{
                    name: [float(rows[stamp][index]) for stamp in stamps]
                    for index, name in enumerate(KLINE_COLUMNS[1:], start=1)
                },
            },
            columns=list(KLINE_COLUMNS),
        )
        _write_csv_atomic(frame, path)
        result.saved = True
        result.rows = len(stamps)
        result.first, result.last = frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1]
        result.missing = (stamps[-1] - stamps[0]) // step + 1 - len(stamps)
        if stamps[-1] < end_ms:
            result.note = f"ends {result.last}"
        progress.symbol_done()
        return result

    report.results = list(await asyncio.gather(*(one(*plan) for plan in plans)))
    progress.close()
    report.elapsed_s = time.monotonic() - started
    report.requests = client.requests - requests_before
    return report


async def download_funding(
    client: BybitPublicClient,
    targets: list[tuple[str, Instrument | None]],
    start_ms: int,
    end_ms: int,
    output_dir: Path,
    *,
    skip_existing: bool,
    concurrency: int,
    show_progress: bool,
) -> PhaseReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = PhaseReport(name="funding")
    requests_before = client.requests
    started = time.monotonic()

    plans: list[tuple[str, list[tuple[int, int]]]] = []
    for base, instrument in targets:
        interval = instrument.funding_interval_minutes if instrument else DEFAULT_FUNDING_INTERVAL_MINUTES
        first = max(start_ms, instrument.launch_ms) if instrument else start_ms
        plans.append((base, plan_windows(first, end_ms, interval * MINUTE_MS, FUNDING_LIMIT)))

    progress = _Progress(sum(len(windows) for _, windows in plans), "funding", client, show_progress)
    progress.symbol_count = len(plans)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(base: str, windows: list[tuple[int, int]]) -> SymbolResult:
        result = SymbolResult(base=base)
        path = output_dir / f"{base}_funding.csv"
        if skip_existing and path.exists():
            progress.skip(len(windows))
            result.note = "exists"
            progress.symbol_done()
            return result
        if not windows:
            result.note = "listed after range"
            progress.symbol_done()
            return result
        async with semaphore:
            rates: dict[int, float] = {}
            planned, completed = len(windows), 0

            async def fetch(low: int, high: int) -> None:
                nonlocal planned, completed
                payload = await client.get(
                    "funding/history",
                    {
                        "category": CATEGORY,
                        "symbol": f"{base}{QUOTE}",
                        "startTime": low,
                        "endTime": high,
                        "limit": FUNDING_LIMIT,
                    },
                )
                completed += 1
                progress.request_done()
                page = payload.get("list") or []
                stamps = []
                for item in page:
                    stamp = int(item["fundingRateTimestamp"])
                    if low <= stamp <= high:
                        rates[stamp] = float(item["fundingRate"])
                        stamps.append(stamp)
                if len(page) >= FUNDING_LIMIT and stamps and min(stamps) > low:
                    # A full page means older settlements in this window may be missing.
                    planned += 1
                    progress.add_requests(1)
                    await fetch(low, min(stamps) - 1)

            tasks = [asyncio.create_task(fetch(low, high)) for low, high in windows]
            try:
                await asyncio.gather(*tasks)
            except Exception as exc:
                for task in tasks:
                    task.cancel()
                progress.skip(planned - completed)
                result.error = str(exc)
                progress.symbol_done(error=True)
                return result

        if not rates:
            result.note = "no funding"
            progress.symbol_done()
            return result
        stamps = sorted(rates)
        frame = pd.DataFrame(
            {"timestamp": _format_timestamps(stamps, daily=False), "funding_rate": [rates[s] for s in stamps]},
            columns=list(FUNDING_COLUMNS),
        )
        _write_csv_atomic(frame, path)
        result.saved = True
        result.rows = len(stamps)
        result.first, result.last = frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1]
        progress.symbol_done()
        return result

    report.results = list(await asyncio.gather(*(one(*plan) for plan in plans)))
    progress.close()
    report.elapsed_s = time.monotonic() - started
    report.requests = client.requests - requests_before
    return report


@dataclass(frozen=True)
class SeriesSpec:
    """A daily public market series fetched in fixed request windows (newest first, capped pages)."""

    name: str
    endpoint: str
    per_request: int
    columns: tuple[str, ...]
    daily_bars: bool
    params: Callable[[str, int, int], dict[str, Any]]
    parse: Callable[[Any], tuple[int, list[float]]]


SERIES: dict[str, SeriesSpec] = {
    # Snapshots stamped at 00:00 UTC: the value at 00:00 of day d+1 is known at the close of day d.
    "oi": SeriesSpec(
        "oi", "open-interest", 200, ("open_interest",), False,
        lambda symbol, low, high: {"category": CATEGORY, "symbol": symbol, "intervalTime": "1d",
                                   "startTime": low, "endTime": high, "limit": 200},
        lambda item: (int(item["timestamp"]), [float(item["openInterest"])]),
    ),
    "lsratio": SeriesSpec(
        "lsratio", "account-ratio", 500, ("buy_ratio", "sell_ratio"), False,
        lambda symbol, low, high: {"category": CATEGORY, "symbol": symbol, "period": "1d",
                                   "startTime": low, "endTime": high, "limit": 500},
        lambda item: (int(item["timestamp"]), [float(item["buyRatio"]), float(item["sellRatio"])]),
    ),
    # Daily premium-index candles (perp vs index); a bar is known at its close.
    "premium": SeriesSpec(
        "premium", "premium-index-price-kline", 1000, ("open", "high", "low", "close"), True,
        lambda symbol, low, high: {"category": CATEGORY, "symbol": symbol, "interval": "D",
                                   "start": low, "end": high, "limit": 1000},
        lambda item: (int(item[0]), [float(value) for value in item[1:5]]),
    ),
}


async def download_series(
    client: BybitPublicClient,
    targets: list[tuple[str, Instrument | None]],
    spec: SeriesSpec,
    start_ms: int,
    end_ms: int,
    output_dir: Path,
    *,
    skip_existing: bool,
    concurrency: int,
    show_progress: bool,
) -> PhaseReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = PhaseReport(name=spec.name)
    requests_before = client.requests
    started = time.monotonic()
    # launchTime can be a contract relaunch date with older history behind it, so it does not clip series.
    plans = [(base, plan_windows(start_ms, end_ms, DAY_MS, spec.per_request)) for base, _ in targets]
    progress = _Progress(sum(len(windows) for _, windows in plans), spec.name, client, show_progress)
    progress.symbol_count = len(plans)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(base: str, windows: list[tuple[int, int]]) -> SymbolResult:
        result = SymbolResult(base=base)
        path = output_dir / f"{base}_{spec.name}.csv"
        if skip_existing and path.exists():
            progress.skip(len(windows))
            result.note = "exists"
            progress.symbol_done()
            return result
        if not windows:
            result.note = "listed after range"
            progress.symbol_done()
            return result
        async with semaphore:
            rows: dict[int, list[float]] = {}
            planned, completed = len(windows), 0

            async def fetch(low: int, high: int) -> None:
                nonlocal planned, completed
                payload = await client.get(spec.endpoint, spec.params(f"{base}{QUOTE}", low, high))
                completed += 1
                progress.request_done()
                page = payload.get("list") or []
                stamps = []
                for item in page:
                    stamp, values = spec.parse(item)
                    if low <= stamp <= high:
                        rows[stamp] = values
                        stamps.append(stamp)
                if len(page) >= spec.per_request and stamps and min(stamps) > low:
                    planned += 1
                    progress.add_requests(1)
                    await fetch(low, min(stamps) - 1)

            tasks = [asyncio.create_task(fetch(low, high)) for low, high in windows]
            try:
                await asyncio.gather(*tasks)
            except Exception as exc:
                for task in tasks:
                    task.cancel()
                progress.skip(planned - completed)
                result.error = str(exc)
                progress.symbol_done(error=True)
                return result

        if not rows:
            result.note = f"no {spec.name}"
            progress.symbol_done()
            return result
        stamps = sorted(rows)
        frame = pd.DataFrame(
            {
                "timestamp": _format_timestamps(stamps, daily=spec.daily_bars),
                **{column: [rows[s][i] for s in stamps] for i, column in enumerate(spec.columns)},
            }
        )
        _write_csv_atomic(frame, path)
        result.saved = True
        result.rows = len(stamps)
        result.first, result.last = frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1]
        progress.symbol_done()
        return result

    report.results = list(await asyncio.gather(*(one(*plan) for plan in plans)))
    progress.close()
    report.elapsed_s = time.monotonic() - started
    report.requests = client.requests - requests_before
    return report


def resolve_targets(
    instruments: dict[str, Instrument],
    symbols: list[str] | None,
    symbols_from: Path | None,
    suffix: int,
) -> list[tuple[str, Instrument | None]]:
    """Return (base symbol, current instrument or None) pairs; None means not currently listed."""
    if symbols:
        bases = [symbol.upper().removesuffix(QUOTE) for symbol in symbols]
    elif symbols_from is not None:
        bases = sorted(path.name.removesuffix(f"_{suffix}.csv") for path in symbols_from.glob(f"*_{suffix}.csv"))
        if not bases:
            raise SystemExit(f"no *_{suffix}.csv files in {symbols_from}")
    else:
        bases = sorted(item.base for item in instruments.values() if item.status == "Trading")
    return [(base, instruments.get(f"{base}{QUOTE}")) for base in dict.fromkeys(bases)]


def _print_report(report: PhaseReport, limit: int = 12) -> None:
    saved = [item for item in report.results if item.saved]
    errors = [item for item in report.results if item.error]
    empty = [item for item in report.results if not item.saved and not item.error and item.note != "exists"]
    rows = sum(item.rows for item in saved)
    rps = report.requests / max(report.elapsed_s, 1e-3)
    print(
        f"{report.name}: saved={len(saved)} rows={rows:,} requests={report.requests:,} "
        f"elapsed={report.elapsed_s:.1f}s ({rps:.0f} req/s) errors={len(errors)}"
    )
    gaps = [item for item in saved if item.missing]
    stale = [item for item in saved if item.note and item.note.startswith("ends")]
    for label, items, describe in (
        ("errors", errors, lambda item: f"{item.base}: {item.error}"),
        ("unavailable", empty, lambda item: f"{item.base} ({item.note})"),
        ("internal gaps", gaps, lambda item: f"{item.base} ({item.missing} missing)"),
        ("stopped early", stale, lambda item: f"{item.base} ({item.note})"),
    ):
        if items:
            shown = ", ".join(describe(item) for item in items[:limit])
            more = f" … +{len(items) - limit}" if len(items) > limit else ""
            print(f"  {label}={len(items)}: {shown}{more}")


def _write_manifest(directory: Path, name: str, payload: dict[str, Any], report: PhaseReport) -> None:
    manifest = {
        **payload,
        "phase": report.name,
        "elapsed_s": round(report.elapsed_s, 2),
        "requests": report.requests,
        "symbols": {
            item.base: {
                "rows": item.rows, "first": item.first, "last": item.last,
                "missing": item.missing, "note": item.note, "error": item.error,
            }
            for item in report.results
        },
    }
    (directory / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    through: date = args.through or (datetime.now(UTC).date() - timedelta(days=1))
    minutes = INTERVAL_MINUTES[args.interval]
    step = minutes * MINUTE_MS
    start_day = through - timedelta(days=args.days - 1)
    start_ms = int(datetime.combine(start_day, datetime.min.time(), UTC).timestamp() * 1000)
    range_end_ms = int(datetime.combine(through + timedelta(days=1), datetime.min.time(), UTC).timestamp() * 1000) - 1
    last_bar_ms = _floor(range_end_ms + 1, step) - step
    show = not args.no_progress

    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10)
    connector = aiohttp.TCPConnector(limit=args.connections, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        limiter = AdaptiveRateLimiter(args.rps)
        client = BybitPublicClient(session, limiter)
        instruments = await fetch_instruments(client)
        targets = resolve_targets(instruments, args.symbols, args.symbols_from, minutes)
        unlisted = [base for base, instrument in targets if instrument is None]
        print(
            f"range {start_day}..{through} ({args.days} days) symbols={len(targets)} "
            f"listed={len(targets) - len(unlisted)} rps={args.rps:g}"
        )
        if unlisted:
            print(f"  not currently listed (attempted anyway): {', '.join(unlisted)}")
        settings = {
            "source": API_ROOT,
            "category": CATEGORY,
            "start": start_day.isoformat(),
            "through": through.isoformat(),
            "days": args.days,
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
        }
        reports: list[PhaseReport] = []

        if not (args.funding_only or args.no_candles):
            report = await download_klines(
                client, targets, args.interval, start_ms, last_bar_ms, args.data_dir,
                skip_existing=args.skip_existing, concurrency=args.symbol_concurrency, show_progress=show,
            )
            _write_manifest(args.data_dir, f"_manifest_{minutes}.json", {**settings, "interval": args.interval}, report)
            reports.append(report)
        if args.funding or args.funding_only:
            report = await download_funding(
                client, targets, start_ms, range_end_ms, args.funding_dir,
                skip_existing=args.skip_existing, concurrency=args.symbol_concurrency, show_progress=show,
            )
            _write_manifest(args.funding_dir, "_manifest_funding.json", settings, report)
            reports.append(report)
        for key in args.series:
            spec = SERIES[key]
            # Snapshot series include the 00:00 value after the last day; bar series end at its open.
            end = start_ms + args.days * DAY_MS - DAY_MS if spec.daily_bars else range_end_ms + 1
            directory = args.series_root / spec.name
            report = await download_series(
                client, targets, spec, start_ms, end, directory,
                skip_existing=args.skip_existing, concurrency=args.symbol_concurrency, show_progress=show,
            )
            _write_manifest(directory, f"_manifest_{spec.name}.json", settings, report)
            reports.append(report)

    print()
    for report in reports:
        _print_report(report)
    print(
        f"rate limiter: final_rps={limiter.rps:.0f} rate_limit_hits={limiter.rate_limit_hits} "
        f"forbidden_hits={limiter.forbidden_hits} retries={client.retried}"
    )
    return int(any(item.error for report in reports for item in report.results))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Bybit USDT perpetual candles and funding history")
    parser.add_argument("--data-dir", type=Path, default=Path("data/1d"), help="Candle output directory")
    parser.add_argument("--funding-dir", type=Path, default=Path("data/funding"), help="Funding output directory")
    parser.add_argument("--days", type=int, default=1000, help="Number of UTC days ending at --through")
    parser.add_argument("--through", type=date.fromisoformat, help="Last confirmed UTC day; default yesterday")
    parser.add_argument("--interval", choices=sorted(INTERVAL_MINUTES), default="D", help="Bybit kline interval")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--symbols", type=lambda raw: [s for s in raw.split(",") if s], help="BTC,ETH,...")
    selection.add_argument("--symbols-from", type=Path, help="Use the symbols of existing candle files in DIR")
    parser.add_argument("--funding", action="store_true", help="Also download funding-rate history")
    parser.add_argument("--funding-only", action="store_true", help="Download funding history without candles")
    parser.add_argument("--no-candles", action="store_true", help="Skip candles (download only selected series)")
    parser.add_argument("--open-interest", dest="series", action="append_const", const="oi",
                        help="Also download daily open interest")
    parser.add_argument("--ls-ratio", dest="series", action="append_const", const="lsratio",
                        help="Also download the daily long/short account ratio")
    parser.add_argument("--premium", dest="series", action="append_const", const="premium",
                        help="Also download daily premium-index candles")
    parser.add_argument("--series-root", type=Path, default=Path("data"), help="Parent directory of series outputs")
    parser.add_argument("--skip-existing", action="store_true", help="Keep files that already exist")
    parser.add_argument("--rps", type=float, default=80.0, help="Target requests per second (Bybit IP cap is 120)")
    parser.add_argument("--connections", type=int, default=64, help="Maximum open HTTP connections")
    parser.add_argument("--symbol-concurrency", type=int, default=48, help="Symbols downloaded at once")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    args.series = list(dict.fromkeys(args.series or []))
    if args.days < 1:
        parser.error("--days must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(run(parse_args(argv))))


if __name__ == "__main__":
    main()
