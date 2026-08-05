from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def simulate_pnl(weights: np.ndarray, open_to_open_return: np.ndarray, one_way_cost: float):
    t_count, n_assets = weights.shape
    pnl = np.zeros(t_count, dtype=np.float64)
    turnover = np.zeros(t_count, dtype=np.float64)
    previous = np.zeros(n_assets, dtype=np.float64)
    for t in range(t_count - 1):
        gross_return = 0.0
        traded = 0.0
        for j in range(n_assets):
            weight = weights[t, j]
            traded += abs(weight - previous[j])
            asset_return = open_to_open_return[t, j]
            if np.isfinite(asset_return):
                gross_return += weight * asset_return
            previous[j] = weight
        turnover[t] = traded
        pnl[t] = gross_return - one_way_cost * traded
    return pnl, turnover


def _side_weights(volatility: np.ndarray, selected: np.ndarray, mode: str) -> np.ndarray:
    if mode == "equal":
        values = np.ones(selected.size, np.float64)
    elif mode == "invvol":
        vol = volatility[selected].astype(np.float64)
        low, high = np.nanquantile(vol, [0.10, 0.90])
        values = 1.0 / np.clip(vol, max(low, 1e-5), max(high, low + 1e-5))
    else:
        raise ValueError(f"Unknown weighting: {mode}")
    return values / values.sum()


def _project_dollar_and_beta(weights: np.ndarray, beta: np.ndarray) -> np.ndarray:
    selected = np.flatnonzero(weights)
    if selected.size < 4:
        return np.zeros_like(weights)
    matrix = np.vstack([np.ones(selected.size), beta[selected].astype(np.float64)])
    raw = weights[selected]
    correction = matrix.T @ np.linalg.solve(
        matrix @ matrix.T + np.eye(matrix.shape[0]) * 1e-6,
        matrix @ raw,
    )
    neutral = raw - correction
    gross = np.abs(neutral).sum()
    output = np.zeros_like(weights)
    if gross > 1e-8:
        output[selected] = neutral / gross
    return output


def build_daily_signal(ctx: dict[str, Any], scores: np.ndarray, portfolio: dict[str, Any]) -> np.ndarray:
    """Create one close-t signal, executed at open t+1, for every available date."""
    liq_rank = ctx["liq_rank"]
    eligible = ctx["eligible_max"]
    beta = ctx["beta60"]
    volatility = ctx["vol20"]
    t_count, n_assets = scores.shape
    signal = np.zeros((t_count, n_assets), np.float32)
    universe_size = int(portfolio["universe_size"])
    tail_fraction = float(portfolio["tail_fraction"])
    weighting = str(portfolio["weighting"])

    for t in range(t_count - 1):
        valid = (
            eligible[t]
            & (liq_rank[t] <= universe_size)
            & np.isfinite(scores[t])
            & np.isfinite(volatility[t])
            & (volatility[t] > 1e-5)
            & np.isfinite(beta[t])
        )
        index = np.flatnonzero(valid)
        if index.size < 20:
            continue
        order = np.argsort(scores[t, index].astype(np.float64))
        k = max(3, int(math.floor(index.size * tail_fraction)))
        k = min(k, index.size // 2)
        short_index = index[order[:k]]
        long_index = index[order[-k:]]
        weights = np.zeros(n_assets, np.float64)
        weights[long_index] = 0.5 * _side_weights(volatility[t], long_index, weighting)
        weights[short_index] = -0.5 * _side_weights(volatility[t], short_index, weighting)
        if bool(portfolio.get("dollar_neutral", True)) or bool(portfolio.get("btc_beta_neutral", True)):
            weights = _project_dollar_and_beta(weights, beta[t])
        signal[t + 1] = weights.astype(np.float32)
    return signal


def smooth_fixed_window(signal: np.ndarray, days: int) -> np.ndarray:
    if days <= 1:
        return signal.copy()
    cumulative = np.cumsum(signal.astype(np.float64), axis=0)
    output = np.zeros_like(signal, dtype=np.float32)
    for t in range(signal.shape[0]):
        total = cumulative[t].copy()
        left = t - days
        if left >= 0:
            total -= cumulative[left]
        output[t] = (total / float(days)).astype(np.float32)
    return output


def overlay_multiplier(ctx: dict[str, Any], overlay: dict[str, Any]) -> np.ndarray:
    """Reproduce the selected legacy BTC-volatility exposure policy exactly.

    The signal known at close ``t`` is already shifted into the open-``t+1`` target
    portfolio by :func:`build_daily_signal`.  The legacy overlay therefore compares
    the current panel row's BTC 20-day volatility with the *lagged* trailing median
    and applies that multiplier to the same target-weight row.
    """
    t_count = len(ctx["dates"])
    if not bool(overlay.get("enabled", False)):
        return np.ones(t_count, np.float32)

    vol = pd.Series(np.asarray(ctx["btc_vol20"], dtype=np.float64))
    median = (
        vol.rolling(
            int(overlay.get("median_lookback", 180)),
            min_periods=int(overlay.get("median_min_periods", 90)),
        )
        .median()
        .shift(int(overlay.get("median_lag", 1)))
    )
    low = float(overlay.get("low_vol_multiplier", 1.10))
    high = float(overlay.get("high_vol_multiplier", 0.75))
    regime_low = vol < median
    multiplier = np.where(regime_low.fillna(False), low, high).astype(np.float32)

    # Keep base gross until both sides of the regime comparison exist.
    unavailable = vol.isna() | median.isna()
    multiplier[unavailable.to_numpy()] = 1.0
    return multiplier


def build_weights(ctx: dict[str, Any], scores: np.ndarray, portfolio: dict[str, Any]) -> np.ndarray:
    signal = build_daily_signal(ctx, scores, portfolio)
    weights = smooth_fixed_window(signal, int(portfolio["smoothing_days"]))
    multiplier = overlay_multiplier(ctx, portfolio.get("overlay", {}))
    return (weights * multiplier[:, None]).astype(np.float32)
