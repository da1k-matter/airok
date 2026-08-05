from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-12


def date_mask(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def monthly_returns(returns: np.ndarray, dates: pd.DatetimeIndex, mask: np.ndarray) -> pd.Series:
    series = pd.Series(returns[mask], index=dates[mask], dtype=np.float64)
    return (1.0 + series).groupby(series.index.to_period("M")).prod() - 1.0


def metrics(
    returns: np.ndarray,
    turnover: np.ndarray,
    dates: pd.DatetimeIndex,
    start: str,
    end: str,
    periods_per_year: int = 365,
) -> tuple[dict[str, float | int | str], pd.Series]:
    mask = date_mask(dates, start, end)
    r = np.asarray(returns[mask], np.float64)
    tr = np.asarray(turnover[mask], np.float64)
    good = np.isfinite(r)
    r, tr = r[good], tr[good]
    if r.size < 2:
        raise ValueError(f"Not enough observations for {start}..{end}")

    equity = np.cumprod(1.0 + r)
    years = r.size / float(periods_per_year)
    total_return = float(equity[-1] - 1.0)
    cagr = float(equity[-1] ** (1.0 / years) - 1.0)
    std = float(np.std(r, ddof=1))
    volatility = std * math.sqrt(periods_per_year)
    sharpe = float(np.mean(r) / (std + EPS) * math.sqrt(periods_per_year))
    downside = float(np.std(np.minimum(r, 0.0), ddof=1) * math.sqrt(periods_per_year))
    sortino = float(np.mean(r) * periods_per_year / (downside + EPS))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    max_dd = float(-drawdown.min())
    monthly = monthly_returns(returns, dates, mask)
    return {
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "n_days": int(r.size),
        "return": total_return,
        "cagr": cagr,
        "vol": float(volatility),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": float(cagr / (max_dd + EPS)),
        "turnover_pa": float(np.nansum(tr) / years),
        "positive_months": int((monthly > 0).sum()),
        "n_months": int(monthly.size),
        "positive_month_ratio": float((monthly > 0).mean()),
        "worst_month": float(monthly.min()),
        "median_month": float(monthly.median()),
    }, monthly


def yearly_table(
    returns: np.ndarray,
    turnover: np.ndarray,
    dates: pd.DatetimeIndex,
    start: str,
    end: str,
    periods_per_year: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    overall = date_mask(dates, start, end)
    for year in sorted(set(dates[overall].year)):
        year_dates = dates[overall & (dates.year == year)]
        if year_dates.empty:
            continue
        row, monthly = metrics(
            returns,
            turnover,
            dates,
            str(year_dates[0].date()),
            str(year_dates[-1].date()),
            periods_per_year,
        )
        row["year"] = int(year)
        row["best_month"] = float(monthly.max())
        rows.append(row)
    columns = [
        "year",
        "start",
        "end",
        "return",
        "cagr",
        "vol",
        "sharpe",
        "sortino",
        "max_dd",
        "calmar",
        "positive_months",
        "n_months",
        "turnover_pa",
        "worst_month",
        "best_month",
    ]
    return pd.DataFrame(rows)[columns]


def save_plots(daily: pd.DataFrame, output_dir: Path) -> None:
    dates = pd.to_datetime(daily["date"])
    plt.figure(figsize=(11, 6))
    plt.plot(dates, daily["equity"])
    plt.axhline(1.0, linewidth=1)
    plt.title("RankTrend OOS equity")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(output_dir / "equity.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(dates, daily["drawdown"])
    plt.axhline(0.0, linewidth=1)
    plt.title("RankTrend drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(output_dir / "drawdown.png", dpi=160)
    plt.close()


def environment_text() -> str:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "numba", "lightgbm", "catboost", "yaml", "matplotlib"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            packages[name] = "not installed"
    return json.dumps(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        },
        indent=2,
    )


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
