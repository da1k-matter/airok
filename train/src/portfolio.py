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


def target_weight_funding(weights: np.ndarray, funding_factor: np.ndarray) -> np.ndarray:
    """Funding return of ideal target weights held from open t to open t+1."""
    return -np.nansum(weights.astype(np.float64) * funding_factor, axis=1)


@dataclass(frozen=True)
class BybitExecutionSimulation:
    pnl: np.ndarray
    turnover: np.ndarray
    held_weights: np.ndarray
    rejected_notional: np.ndarray
    attempted_orders: np.ndarray
    executed_orders: np.ndarray
    skipped_orders: np.ndarray
    funding_pnl: np.ndarray
    funding_uncovered_notional: np.ndarray


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
    funding_factor: np.ndarray | None = None,
) -> BybitExecutionSimulation:
    """Execute one Bybit linear-perpetual market order per symbol and day.

    The order quantity is rounded down to the instrument qty step before every
    Bybit lot-size validation. Rejected deltas remain as tracking error until a
    later rebalance is valid under the same current rule snapshot.

    With `funding_factor` (see funding.load_funding_factors) every position held
    through day t pays `notional * factor[t]`; held notional on days without
    funding coverage is reported in `funding_uncovered_notional` and pays none.
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
    funding_pnl = np.zeros(t_count, dtype=np.float64)
    funding_uncovered_notional = np.zeros(t_count, dtype=np.float64)
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
        position_notional = np.where(valid_price, held_quantity * prices, 0.0)
        held_weights[t] = (position_notional / (equity * gross_leverage)).astype(np.float32)

        returns = open_to_open_return[t].astype(np.float64)
        valid = valid_price & np.isfinite(returns)
        gross_pnl = float(np.dot(position_notional[valid], returns[valid]))
        fees = one_way_cost * executed_notional
        funding = 0.0
        if funding_factor is not None:
            factor = funding_factor[t].astype(np.float64)
            covered = np.isfinite(factor)
            funding = -float(np.dot(position_notional[covered], factor[covered]))
            funding_uncovered_notional[t] = float(np.abs(position_notional[~covered]).sum())
            funding_pnl[t] = funding / equity
        pnl[t] = (gross_pnl - fees + funding) / equity
        equity += gross_pnl - fees + funding
        # Quantity is invariant between rebalances; next day's price marks it.

    return BybitExecutionSimulation(
        pnl=pnl,
        turnover=turnover,
        held_weights=held_weights,
        rejected_notional=rejected_notional,
        attempted_orders=attempted_orders,
        executed_orders=executed_orders,
        skipped_orders=skipped_orders,
        funding_pnl=funding_pnl,
        funding_uncovered_notional=funding_uncovered_notional,
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


def _cross_sectional_z(ctx: dict[str, Any], name: str) -> np.ndarray:
    source = {"size": "dvol20", "vol": "vol20"}[name]
    values = np.asarray(ctx[source], np.float64)
    if name == "vol":
        values = np.log(np.maximum(values, 1e-6))
    eligible = np.asarray(ctx["eligible_max"])
    masked = np.where(eligible, values, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(masked, axis=1, keepdims=True)
        std = np.nanstd(masked, axis=1, keepdims=True)
    return np.nan_to_num((values - mean) / np.where(std > 1e-12, std, 1.0))


def pca_loadings(ctx: dict[str, Any], components: int, window: int = 90) -> list[np.ndarray]:
    """Loadings on principal components 2..components+1 of trailing eligible returns, known at each close.

    Component 1 is close to the equal-weighted market and is handled by the alt-index beta.
    """
    key = f"pca_{components}_{window}"
    if key not in ctx:
        returns = np.nan_to_num(np.asarray(ctx["ret1"], np.float64))
        eligible = np.asarray(ctx["eligible_max"])
        t_count, n_assets = returns.shape
        loadings = [np.zeros((t_count, n_assets)) for _ in range(components)]
        for t in range(window, t_count):
            members = np.flatnonzero(eligible[t])
            if members.size <= components + 1:
                continue
            block = returns[t - window + 1 : t + 1][:, members]
            block = block - block.mean(axis=0)
            _, _, vt = np.linalg.svd(block, full_matrices=False)
            for k in range(components):
                loadings[k][t, members] = vt[k + 1] * np.sqrt(members.size)
        ctx[key] = loadings
    return ctx[key]


def alt_index_beta(ctx: dict[str, Any], window: int = 60, top: int | None = None) -> np.ndarray:
    """Rolling beta of every asset to an equal-weighted universe return, known at each close.

    The index holds the eligible universe, or only its `top` most liquid names.
    """
    key = "alt_beta60" if top is None else f"alt_beta60_top{top}"
    if key not in ctx:
        returns = np.asarray(ctx["ret1"], np.float64)
        eligible = np.asarray(ctx["eligible_max"])
        if top is not None:
            eligible = eligible & (np.asarray(ctx["liq_rank"]) <= top)
        with np.errstate(invalid="ignore"):
            index = np.nanmean(np.where(eligible, returns, np.nan), axis=1)
        frame = pd.DataFrame(returns)
        market = pd.Series(index)
        minimum = window // 2
        cov = frame.mul(market, axis=0).rolling(window, min_periods=minimum).mean() - frame.rolling(
            window, min_periods=minimum
        ).mean().mul(market.rolling(window, min_periods=minimum).mean(), axis=0)
        var = market.rolling(window, min_periods=minimum).var(ddof=0)
        ctx[key] = np.clip(cov.div(var, axis=0).to_numpy(), -3.0, 5.0)
    return ctx[key]


def _project_dollar_and_beta(
    weights: np.ndarray, beta: np.ndarray, extra_betas: list[np.ndarray] | None = None
) -> np.ndarray:
    selected = np.flatnonzero(weights)
    if selected.size < 4:
        return np.zeros_like(weights)
    rows = [np.ones(selected.size), beta[selected].astype(np.float64)]
    rows += [np.nan_to_num(extra[selected].astype(np.float64)) for extra in extra_betas or []]
    matrix = np.vstack(rows)
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


def trailing_funding(ctx: dict[str, Any], lookback: int) -> np.ndarray:
    """Annualized mean funding rate over the last `lookback` holding days, known at each close."""
    if "funding_rate_day" not in ctx:
        raise KeyError("funding-aware portfolio options need ctx['funding_rate_day']; see pipeline.attach_funding")
    daily = pd.DataFrame(np.asarray(ctx["funding_rate_day"], np.float64))
    return (daily.rolling(lookback, min_periods=1).mean().to_numpy() * 365.0).astype(np.float64)


def _funding_adjusted(scores: np.ndarray, ctx: dict[str, Any], portfolio: dict[str, Any]):
    """Return (scores, long_ok, short_ok) after optional funding-aware selection."""
    options = portfolio.get("funding_select") or {}
    mode = str(options.get("mode", "off"))
    if mode == "off":
        return scores, None, None
    funding = np.nan_to_num(trailing_funding(ctx, int(options.get("lookback", 7))))
    if mode == "penalty":
        return scores - float(options["kappa"]) * funding, None, None
    if mode == "filter":
        threshold = float(options["threshold"])
        return scores, funding <= threshold, funding >= -threshold
    raise ValueError(f"Unknown portfolio.funding_select.mode: {mode}")


def build_daily_signal(
    ctx: dict[str, Any],
    scores: np.ndarray,
    portfolio: dict[str, Any],
    short_scores: np.ndarray | None = None,
) -> np.ndarray:
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
    scores, long_ok, short_ok = _funding_adjusted(scores, ctx, portfolio)
    alt_strength = float(portfolio.get("alt_beta_neutral", 0.0))
    alt_top = portfolio.get("alt_index_top")
    alt_beta = alt_index_beta(ctx, top=int(alt_top) if alt_top else None) if alt_strength > 0 else None
    # Optional characteristic exposures neutralized in the same blended book (cross-sectional z-scores).
    characteristic_names = list(portfolio.get("neutral_characteristics", []))
    characteristics = [_cross_sectional_z(ctx, name) for name in characteristic_names]
    characteristics += pca_loadings(ctx, int(portfolio["pca_neutral"])) if portfolio.get("pca_neutral") else []

    trade_mask = portfolio.get("_trade_mask")

    for t in range(t_count - 1):
        valid = (
            eligible[t]
            & np.isfinite(scores[t])
            & np.isfinite(volatility[t])
            & (volatility[t] > 1e-5)
            & np.isfinite(beta[t])
        )
        if short_scores is not None:
            valid &= np.isfinite(short_scores[t])
        if trade_mask is None:
            valid &= liq_rank[t] <= universe_size
        else:
            # Restricted trading: the liquidity-ranked universe is formed after the allowlist filter.
            candidates = np.flatnonzero(valid & trade_mask)
            chosen = candidates[np.argsort(liq_rank[t, candidates], kind="stable")][:universe_size]
            valid = np.zeros(n_assets, bool)
            valid[chosen] = True
        index = np.flatnonzero(valid)
        if index.size < minimum_universe:
            continue
        order = np.argsort(scores[t, index].astype(np.float64), kind="stable")
        k = _tail_size(portfolio, index.size)
        if k < 1:
            continue
        ranked = index[order]
        short_pool = ranked if short_ok is None else ranked[short_ok[t, ranked]]
        long_pool = ranked if long_ok is None else ranked[long_ok[t, ranked]]
        if short_pool.size < k or long_pool.size < k:
            continue
        long_index = long_pool[-k:]
        hedge_count = int(portfolio.get("hedge_count", 0))
        if hedge_count:
            # Replace model longs with the most liquid eligible names: a low-noise beta hedge for the shorts.
            long_index = index[np.argsort(liq_rank[t, index], kind="stable")][:hedge_count]
            short_pool = short_pool[~np.isin(short_pool, long_index)]
        if short_scores is None:
            if short_pool.size < k:
                continue
            short_index = short_pool[:k]
        else:
            # A dedicated short-side model ranks shorts; low score = short. Never short a selected long.
            by_short = index[np.argsort(short_scores[t, index].astype(np.float64), kind="stable")]
            by_short = by_short[~np.isin(by_short, long_index)]
            if short_ok is not None:
                by_short = by_short[short_ok[t, by_short]]
            if by_short.size < k:
                continue
            short_index = by_short[:k]
        weights = np.zeros(n_assets, np.float64)
        weights[long_index] = 0.5 * _side_weights(volatility[t], long_index, weighting)
        weights[short_index] = -0.5 * _side_weights(volatility[t], short_index, weighting)
        if bool(portfolio.get("dollar_neutral", True)) or bool(portfolio.get("btc_beta_neutral", True)):
            btc_only = _project_dollar_and_beta(weights, beta[t])
            if alt_beta is None:
                weights = btc_only
            else:
                # Partial neutralization blends the BTC-only and the BTC+alt-index neutral books.
                blended = (1.0 - alt_strength) * btc_only + alt_strength * _project_dollar_and_beta(
                    weights, beta[t], [alt_beta[t]] + [values[t] for values in characteristics]
                )
                gross = np.abs(blended).sum()
                weights = blended / gross if gross > 1e-8 else blended
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


def smooth_with_fast_exit(
    ctx: dict[str, Any], signal: np.ndarray, days: int, options: dict[str, Any]
) -> np.ndarray:
    """Boxcar smoothing in which costly shorts are held for fewer days at the same capital-days.

    A short whose trailing funding (annualized, known at the signal close) is at or below
    `funding_below` enters with weight days/fast_days and lives fast_days instead of days.
    Each smoothed book is then made dollar- and BTC-beta-neutral again at its original gross.
    """
    fast_days = int(options.get("days", 7))
    signal_close = np.roll(trailing_funding(ctx, int(options.get("lookback", 7))), 1, axis=0)
    fast = (signal < 0) & (np.nan_to_num(signal_close, nan=0.0) <= float(options["funding_below"]))
    slow_part = np.where(fast, 0.0, signal)
    fast_part = np.where(fast, signal, 0.0) * (days / fast_days)
    combined = smooth_fixed_window(slow_part, days) + smooth_fixed_window(fast_part, fast_days)
    beta = np.asarray(ctx["beta60"])
    output = np.zeros_like(combined)
    for t in range(combined.shape[0]):
        gross = np.abs(combined[t]).sum()
        if gross <= 1e-8 or t == 0:
            continue
        # Weights executed at open t were formed at close t-1, with the betas known then.
        neutral = _project_dollar_and_beta(combined[t].astype(np.float64), np.nan_to_num(beta[t - 1]))
        output[t] = (neutral * gross).astype(np.float32)
    return output


def smooth_ewma(signal: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted average of daily signals, normalized like the fixed window."""
    decay = 0.5 ** (1.0 / halflife)
    output = np.zeros_like(signal, dtype=np.float32)
    state = np.zeros(signal.shape[1], np.float64)
    for t in range(signal.shape[0]):
        state = decay * state + (1.0 - decay) * signal[t]
        output[t] = state.astype(np.float32)
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


def vol_target_multiplier(ctx: dict[str, Any], weights: np.ndarray, options: dict[str, Any]) -> np.ndarray:
    """Scale exposure to an ex-ante portfolio volatility target.

    Weights executed at open t are priced on close-to-close returns of the
    previous `lookback` days (rows t-lookback .. t-1), all known at that open.
    """
    returns = np.nan_to_num(np.asarray(ctx["ret1"], np.float64))
    lookback = int(options.get("lookback", 60))
    target = float(options["target_vol"]) / math.sqrt(365.0)
    low, high = float(options.get("min_scale", 0.5)), float(options.get("max_scale", 1.5))
    multiplier = np.ones(weights.shape[0], np.float64)
    for t in range(lookback, weights.shape[0]):
        if not np.any(weights[t]):
            continue
        history = returns[t - lookback : t] @ weights[t].astype(np.float64)
        sigma = history.std(ddof=1)
        if sigma > 1e-9:
            multiplier[t] = min(high, max(low, target / sigma))
    return multiplier


def build_weights(
    ctx: dict[str, Any],
    scores: np.ndarray,
    portfolio: dict[str, Any],
    short_scores: np.ndarray | None = None,
) -> np.ndarray:
    signal = build_daily_signal(ctx, scores, portfolio, short_scores)
    halflife = portfolio.get("smoothing_halflife")
    fast = portfolio.get("fast_exit") or {}
    if halflife:
        weights = smooth_ewma(signal, float(halflife))
    elif fast:
        weights = smooth_with_fast_exit(ctx, signal, int(portfolio["smoothing_days"]), fast)
    else:
        weights = smooth_fixed_window(signal, int(portfolio["smoothing_days"]))
    vol_target = portfolio.get("vol_target") or {}
    if bool(vol_target.get("enabled", False)):
        multiplier = vol_target_multiplier(ctx, weights, vol_target)
    else:
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
