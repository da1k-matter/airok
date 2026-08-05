import numpy as np
import pandas as pd

from ranktrend.portfolio import _project_dollar_and_beta, overlay_multiplier, smooth_fixed_window
from ranktrend.reporting import metrics
from ranktrend.research import rolling_rank_pct


def test_rank_ties_use_average_percentile():
    values = np.array([[1.0, 2.0, 2.0, 4.0]], dtype=np.float32)
    result = rolling_rank_pct(values)
    np.testing.assert_allclose(result, [[0.25, 0.625, 0.625, 1.0]])


def test_projection_is_dollar_and_beta_neutral():
    raw = np.array([0.3, 0.2, -0.25, -0.25], dtype=float)
    beta = np.array([1.2, 0.8, 1.5, 0.4], dtype=float)
    out = _project_dollar_and_beta(raw, beta)
    assert abs(out.sum()) < 2e-6
    assert abs(out @ beta) < 2e-6
    np.testing.assert_allclose(np.abs(out).sum(), 1.0, atol=2e-6)


def test_smoothing_uses_fixed_denominator():
    signal = np.zeros((4, 1), np.float32)
    signal[0, 0] = 1.0
    signal[1, 0] = 1.0
    got = smooth_fixed_window(signal, 4).ravel()
    np.testing.assert_allclose(got, [0.25, 0.5, 0.5, 0.5])


def test_metrics_compound_calendar_months():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    returns = np.full(40, 0.001)
    turnover = np.zeros(40)
    result, monthly = metrics(returns, turnover, dates, "2024-01-01", "2024-02-09", 365)
    assert result["positive_months"] == 2
    assert len(monthly) == 2
    assert result["max_dd"] == 0.0


def test_btc_vol_overlay_uses_current_vol_and_lagged_median():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    ctx = {"dates": dates, "btc_vol20": np.array([1.0, 2.0, 3.0, 1.0, 4.0, 0.5])}
    cfg = {
        "enabled": True,
        "median_lookback": 3,
        "median_min_periods": 2,
        "median_lag": 1,
        "low_vol_multiplier": 1.1,
        "high_vol_multiplier": 0.75,
    }
    got = overlay_multiplier(ctx, cfg)
    # Lagged rolling medians are [nan, nan, 1.5, 2.0, 2.0, 3.0].
    np.testing.assert_allclose(got, [1.0, 1.0, 0.75, 1.1, 0.75, 1.1])
