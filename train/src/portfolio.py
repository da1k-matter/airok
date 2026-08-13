from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def simulate_target_weight_pnl(weights: np.ndarray, open_to_open_return: np.ndarray, one_way_cost: float):
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


@dataclass(frozen=True)
class BybitExecutionSimulation:
    pnl: np.ndarray
    turnover: np.ndarray
    held_weights: np.ndarray
    rejected_notional: np.ndarray
    attempted_orders: np.ndarray
    executed_orders: np.ndarray
    skipped_orders: np.ndarray


def simulate_bybit_market_pnl(
    target_weights: np.ndarray,
    open_prices: np.ndarray,
    open_to_open_return: np.ndarray,
    one_way_cost: float,
    initial_equity_usd: float,
    gross_leverage: float,
    qty_step: np.ndarray,
    min_order_qty: np.ndarray,
    min_notional_value: np.ndarray,
    max_market_order_qty: np.ndarray,
) -> BybitExecutionSimulation:
    """Execute one Bybit linear-perpetual market order per symbol and day.

    The order quantity is rounded down to the instrument qty step before every
    Bybit lot-size validation. Rejected deltas remain as tracking error until a
    later rebalance is valid under the same current rule snapshot.
    """
    if initial_equity_usd <= 0.0 or gross_leverage <= 0.0:
        raise ValueError("Execution equity and gross leverage must be positive")

    t_count, n_assets = target_weights.shape
    pnl = np.zeros(t_count, dtype=np.float64)
    turnover = np.zeros(t_count, dtype=np.float64)
    held_weights = np.zeros((t_count, n_assets), dtype=np.float32)
    rejected_notional = np.zeros(t_count, dtype=np.float64)
    attempted_orders = np.zeros(t_count, dtype=np.int32)
    executed_orders = np.zeros(t_count, dtype=np.int32)
    skipped_orders = np.zeros(t_count, dtype=np.int32)
    held_quantity = np.zeros(n_assets, dtype=np.float64)
    equity = float(initial_equity_usd)

    for t in range(t_count - 1):
        if equity <= 0.0:
            break
        prices = open_prices[t].astype(np.float64)
        valid_price = np.isfinite(prices) & (prices > 0.0)
        target_notional = target_weights[t].astype(np.float64) * equity * gross_leverage
        current_notional = held_quantity * prices
        delta_notional = target_notional - current_notional
        active = valid_price & (np.abs(delta_notional) > 1e-12)
        attempted_orders[t] = int(active.sum())
        requested_quantity = np.zeros(n_assets, dtype=np.float64)
        requested_quantity[active] = delta_notional[active] / prices[active]
        rounded_quantity = np.sign(requested_quantity) * (
            np.floor(np.abs(requested_quantity) / qty_step + 1e-6) * qty_step
        )
        rounded_notional = np.abs(rounded_quantity) * prices
        executable = (
            active
            & (np.abs(rounded_quantity) + 1e-12 >= min_order_qty)
            & (rounded_notional + 1e-8 * np.maximum(1.0, min_notional_value) >= min_notional_value)
            & (np.abs(rounded_quantity) <= max_market_order_qty + 1e-12)
        )
        rejected = active & ~executable
        held_quantity[executable] += rounded_quantity[executable]
        executed_notional = float(rounded_notional[executable].sum())
        rejected_notional[t] = float(np.abs(delta_notional[rejected]).sum())
        executed_orders[t] = int(executable.sum())
        skipped_orders[t] = int(rejected.sum())
        turnover[t] = executed_notional / equity
        held_weights[t] = (held_quantity * prices / (equity * gross_leverage)).astype(np.float32)

        returns = open_to_open_return[t].astype(np.float64)
        valid = valid_price & np.isfinite(returns)
        gross_pnl = float(np.dot((held_quantity * prices)[valid], returns[valid]))
        fees = one_way_cost * executed_notional
        pnl[t] = (gross_pnl - fees) / equity
        equity += gross_pnl - fees
        # Quantity is invariant between rebalances; next day's price marks it.

    return BybitExecutionSimulation(
        pnl=pnl,
        turnover=turnover,
        held_weights=held_weights,
        rejected_notional=rejected_notional,
        attempted_orders=attempted_orders,
        executed_orders=executed_orders,
        skipped_orders=skipped_orders,
    )


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


def _tail_size(portfolio: dict[str, Any], universe_count: int) -> int:
    if portfolio.get("tail_count") is not None:
        count = int(portfolio["tail_count"])
    else:
        count = max(1, int(math.floor(universe_count * float(portfolio["tail_fraction"]))))
    return min(count, universe_count // 2)


def build_daily_signal(ctx: dict[str, Any], scores: np.ndarray, portfolio: dict[str, Any]) -> np.ndarray:
    """Create close-t signals shifted to execution at open t+1."""
    liq_rank = np.asarray(ctx["liq_rank"])
    eligible = np.asarray(ctx["eligible_max"])
    beta = np.asarray(ctx["beta60"])
    volatility = np.asarray(ctx["vol20"])
    t_count, n_assets = scores.shape
    signal = np.zeros((t_count, n_assets), np.float32)
    universe_size = int(portfolio["universe_size"])
    weighting = str(portfolio["weighting"])
    minimum_universe = int(portfolio.get("minimum_universe", 20))

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
        if index.size < minimum_universe:
            continue
        order = np.argsort(scores[t, index].astype(np.float64), kind="stable")
        k = _tail_size(portfolio, index.size)
        if k < 1:
            continue
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
    multiplier = np.where((vol < median).fillna(False), low, high).astype(np.float32)
    unavailable = vol.isna() | median.isna()
    multiplier[unavailable.to_numpy()] = 1.0
    return multiplier


def build_weights(ctx: dict[str, Any], scores: np.ndarray, portfolio: dict[str, Any]) -> np.ndarray:
    signal = build_daily_signal(ctx, scores, portfolio)
    weights = smooth_fixed_window(signal, int(portfolio["smoothing_days"]))
    multiplier = overlay_multiplier(ctx, portfolio.get("overlay", {}))
    return (weights * multiplier[:, None]).astype(np.float32)


def average_portfolios(portfolios: list[np.ndarray], renormalize: bool = False) -> np.ndarray:
    weights = np.mean(np.stack(portfolios).astype(np.float32), axis=0).astype(np.float32)
    if not renormalize:
        return weights
    gross = np.abs(weights).sum(axis=1)
    good = gross > 1e-8
    weights[good] /= gross[good, None]
    return weights
