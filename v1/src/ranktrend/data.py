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
    """Content fingerprint stable across copying, ZIP/TAR extraction and machines."""
    directory = Path(resolved_directory)
    digest = hashlib.sha256()
    for path in list_data_files(directory):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def data_fingerprint(data_dir: Path) -> str:
    return _stable_data_fingerprint(str(data_dir.resolve()))


def validate_data(data_dir: Path) -> dict[str, Any]:
    files = list_data_files(data_dir)
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    bad: list[str] = []
    symbols: list[str] = []
    for path in files:
        try:
            frame = pd.read_csv(path, nrows=5)
            missing = set(REQUIRED_COLUMNS) - set(frame.columns)
            if missing:
                bad.append(f"{path.name}: missing {sorted(missing)}")
                continue
            timestamps = pd.read_csv(path, usecols=["timestamp"])["timestamp"]
            dt = pd.to_datetime(timestamps, errors="coerce")
            if dt.isna().any() or not dt.is_monotonic_increasing or dt.duplicated().any():
                bad.append(f"{path.name}: invalid/duplicate/non-monotonic timestamps")
                continue
            starts.append(dt.iloc[0])
            ends.append(dt.iloc[-1])
            symbols.append(path.name.removesuffix("_1440.csv"))
        except Exception as exc:  # pragma: no cover - surfaced to doctor
            bad.append(f"{path.name}: {exc}")
    if bad:
        raise ValueError("Dataset validation failed:\n" + "\n".join(bad[:20]))
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
    meta: list[tuple[Path, pd.Timestamp, pd.Timestamp]] = []
    for path in files:
        ts = pd.read_csv(path, usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"]
        meta.append((path, ts.iloc[0], ts.iloc[-1]))
    start = min(x[1] for x in meta)
    end = max(x[2] for x in meta)
    dates = pd.date_range(start, end, freq="D")
    date0 = dates[0]
    symbols = [path.name.removesuffix("_1440.csv") for path in files]
    t_count, n_assets = len(dates), len(symbols)
    arrays = {name: np.full((t_count, n_assets), np.nan, np.float32) for name in ("open", "high", "low", "close", "volume")}
    first_idx = np.full(n_assets, t_count, np.int32)
    for j, path in enumerate(files):
        frame = pd.read_csv(path, parse_dates=["timestamp"], usecols=list(REQUIRED_COLUMNS))
        idx = (frame["timestamp"] - date0).dt.days.to_numpy(np.int32)
        for name in arrays:
            arrays[name][idx, j] = frame[name].to_numpy(np.float32)
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
