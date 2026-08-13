import numpy as np
import pandas as pd

from portfolio import (
    _project_dollar_and_beta,
    average_portfolios,
    overlay_multiplier,
    simulate_bybit_market_pnl,
    smooth_fixed_window,
)
from reporting import metrics
from research import aggregate_seed_ranks, rolling_rank_pct


def test_rank_ties_use_average_percentile():
    values = np.array([[1.0, 2.0, 2.0, 4.0]], dtype=np.float32)
    result = rolling_rank_pct(values)
    np.testing.assert_allclose(result, [[0.25, 0.625, 0.625, 1.0]])


def test_projection_is_dollar_and_beta_neutral():
    raw = np.array([0.3, 0.2, -0.25, -0.25], dtype=float)
    beta = np.array([1.2, 0.8, 1.5, 0.4], dtype=float)
    output = _project_dollar_and_beta(raw, beta)
    assert abs(output.sum()) < 2e-6
    assert abs(output @ beta) < 2e-6
    np.testing.assert_allclose(np.abs(output).sum(), 1.0, atol=2e-6)


def test_smoothing_uses_fixed_denominator():
    signal = np.zeros((4, 1), np.float32)
    signal[:2, 0] = 1.0
    got = smooth_fixed_window(signal, 4).ravel()
    np.testing.assert_allclose(got, [0.25, 0.5, 0.5, 0.5])


def test_seed_rank_aggregations():
    first = np.array([[0.1, 0.8], [0.4, np.nan]], dtype=np.float32)
    second = np.array([[0.3, 0.6], [0.8, 0.2]], dtype=np.float32)
    third = np.array([[0.2, 0.9], [0.6, 0.4]], dtype=np.float32)
    mean = aggregate_seed_ranks([first, second, third], "mean_rank")
    median = aggregate_seed_ranks([first, second, third], "median_rank")
    np.testing.assert_allclose(mean[0], [0.2, 0.7666667], rtol=1e-5)
    np.testing.assert_allclose(median[0], [0.2, 0.8])
    np.testing.assert_allclose(mean[1, 1], 0.3)


def test_portfolio_average_can_preserve_or_renormalize_gross():
    left = np.array([[0.5, -0.5]], np.float32)
    right = np.array([[0.0, 0.0]], np.float32)
    raw = average_portfolios([left, right], renormalize=False)
    normalized = average_portfolios([left, right], renormalize=True)
    np.testing.assert_allclose(np.abs(raw).sum(), 0.5)
    np.testing.assert_allclose(np.abs(normalized).sum(), 1.0)


def test_metrics_compound_calendar_months():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    returns = np.full(40, 0.001)
    turnover = np.zeros(40)
    result, monthly = metrics(returns, turnover, dates, "2024-01-01", "2024-02-09", 365)
    assert result["positive_months"] == 2
    assert len(monthly) == 2
    assert result["max_dd"] == 0.0


def test_btc_vol_overlay_uses_lagged_median():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    ctx = {"dates": dates, "btc_vol20": np.array([1.0, 2.0, 3.0, 1.0, 4.0, 0.5])}
    config = {
        "enabled": True,
        "median_lookback": 3,
        "median_min_periods": 2,
        "median_lag": 1,
        "low_vol_multiplier": 1.1,
        "high_vol_multiplier": 0.75,
    }
    got = overlay_multiplier(ctx, config)
    np.testing.assert_allclose(got, [1.0, 1.0, 0.75, 1.1, 0.75, 1.1])


def test_bybit_execution_rounds_quantity_and_retains_invalid_deltas():
    weights = np.array([[0.02], [0.025], [0.0], [0.0]], dtype=np.float32)
    returns = np.zeros_like(weights)
    result = simulate_bybit_market_pnl(
        weights,
        np.full_like(weights, 100.0),
        returns,
        one_way_cost=0.0,
        initial_equity_usd=1_000.0,
        gross_leverage=1.0,
        qty_step=np.array([0.1]),
        min_order_qty=np.array([0.1]),
        min_notional_value=np.array([10.0]),
        max_market_order_qty=np.array([100.0]),
    )
    np.testing.assert_allclose(result.held_weights[:3, 0], [0.02, 0.02, 0.0])
    np.testing.assert_allclose(result.turnover[:3], [0.02, 0.0, 0.02])
    np.testing.assert_allclose(result.rejected_notional[:3], [0.0, 5.0, 0.0], atol=1e-5)
    np.testing.assert_array_equal(result.executed_orders[:3], [1, 0, 1])
    np.testing.assert_array_equal(result.skipped_orders[:3], [0, 1, 0])
