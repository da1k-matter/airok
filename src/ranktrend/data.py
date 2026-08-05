from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def list_data_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*_1440.csv"))
    if not files:
        raise FileNotFoundError(f"No *_1440.csv files in {data_dir}")
    return files


@lru_cache(maxsize=8)
def _stable_data_fingerprint(resolved_directory: str) -> str:
    directory = Path(resolved_directory)
    digest = hashlib.sha256()
    for path in list_data_files(directory):
        digest.update(path.name.encode())
        digest.update(b"\0")
        stat = path.stat()
        digest.update(str(stat.st_size).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            digest.update(handle.read(4096))
            if stat.st_size > 4096:
                handle.seek(max(0, stat.st_size - 4096))
                digest.update(handle.read(4096))
    return digest.hexdigest()[:16]


def data_fingerprint(data_dir: Path) -> str:
    """Fast stable-enough cache fingerprint based on names, sizes and file edges."""
    return _stable_data_fingerprint(str(data_dir.resolve()))


def validate_data(data_dir: Path) -> dict[str, Any]:
    files = list_data_files(data_dir)
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    symbols: list[str] = []
    errors: list[str] = []

    for path in files:
        try:
            frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
            timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
            if timestamps.isna().any() or not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
                errors.append(f"{path.name}: invalid, duplicate or non-monotonic timestamps")
                continue
            values = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)
            if np.any(values[:, :4] <= 0) or np.any(values[:, 4] < 0):
                errors.append(f"{path.name}: non-positive OHLC or negative volume")
                continue
            starts.append(timestamps.iloc[0])
            ends.append(timestamps.iloc[-1])
            symbols.append(path.name.removesuffix("_1440.csv"))
        except Exception as exc:  # pragma: no cover
            errors.append(f"{path.name}: {exc}")

    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors[:20]))
    if "BTC" not in symbols:
        raise ValueError("BTC_1440.csv is required")
    return {
        "file_count": len(files),
        "asset_count": len(symbols),
        "start": str(min(starts).date()),
        "end": str(max(ends).date()),
        "fingerprint": data_fingerprint(data_dir),
    }


def load_panel(data_dir: Path) -> dict[str, Any]:
    files = list_data_files(data_dir)
    frames: list[tuple[Path, pd.DataFrame]] = []
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    for path in files:
        frame = pd.read_csv(path, parse_dates=["timestamp"], usecols=list(REQUIRED_COLUMNS))
        frames.append((path, frame))
        start = frame["timestamp"].iloc[0] if start is None else min(start, frame["timestamp"].iloc[0])
        end = frame["timestamp"].iloc[-1] if end is None else max(end, frame["timestamp"].iloc[-1])

    assert start is not None and end is not None
    dates = pd.date_range(start, end, freq="D")
    date0 = dates[0]
    symbols = [path.name.removesuffix("_1440.csv") for path, _ in frames]
    t_count, n_assets = len(dates), len(symbols)
    arrays = {
        name: np.full((t_count, n_assets), np.nan, np.float32)
        for name in ("open", "high", "low", "close", "volume")
    }
    first_idx = np.full(n_assets, t_count, np.int32)

    for j, (_, frame) in enumerate(frames):
        idx = (frame["timestamp"] - date0).dt.days.to_numpy(np.int32)
        for name, array in arrays.items():
            array[idx, j] = frame[name].to_numpy(np.float32)
        first_idx[j] = int(idx[0])

    return {"dates": dates, "symbols": symbols, **arrays, "first_idx": first_idx}


def save_context(ctx: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "dates": [str(x.date()) for x in ctx["dates"]],
        "symbols": list(ctx["symbols"]),
        "feature_names": list(ctx["feature_names"]),
        "btc_idx": int(ctx["btc_idx"]),
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for key, value in ctx.items():
        if isinstance(value, np.ndarray):
            np.save(directory / f"{key}.npy", value, allow_pickle=False)


def load_context(directory: Path, mmap: bool = True) -> dict[str, Any]:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    ctx: dict[str, Any] = {
        "dates": pd.DatetimeIndex(meta["dates"]),
        "symbols": meta["symbols"],
        "feature_names": meta["feature_names"],
        "btc_idx": int(meta["btc_idx"]),
    }
    for path in directory.glob("*.npy"):
        ctx[path.stem] = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    return ctx
