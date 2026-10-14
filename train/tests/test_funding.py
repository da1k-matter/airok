import numpy as np
import pandas as pd

from funding import load_funding_factors
from portfolio import simulate_bybit_market_pnl, target_weight_funding


def _write(tmp_path, symbol, rows):
    pd.DataFrame(rows, columns=["timestamp", "funding_rate"]).to_csv(tmp_path / f"{symbol}_funding.csv", index=False)


def test_settlements_map_to_the_holding_day_that_pays_them(tmp_path):
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    opens = np.array([[100.0], [200.0], [200.0], [200.0]])
    _write(
        tmp_path,
        "X",
        [
            ("2024-01-01 00:00:00", 0.5),  # paid by the position held during 2023-12-31: outside the panel
            ("2024-01-01 12:00:00", 0.01),  # day 0, price halfway from 100 to 200
            ("2024-01-02 00:00:00", 0.02),  # rebalance-time settlement: still day 0's position
            ("2024-01-02 08:00:00", 0.03),  # day 1
        ],
    )
    loaded = load_funding_factors(tmp_path, dates, ["X", "MISSING"], np.column_stack([opens, opens]))
    factor = loaded["factor"]
    np.testing.assert_allclose(factor[:2, 0], [0.01 * 1.5 + 0.02 * 2.0, 0.03])
    assert np.isnan(factor[2:, 0]).all()
    assert np.isnan(factor[:, 1]).all()
    assert loaded["settlements"] == 3


def test_positive_funding_is_paid_by_longs_and_received_by_shorts():
    weights = np.array([[0.5, -0.5], [0.5, -0.5], [0.0, 0.0]], np.float32)
    factor = np.array([[0.001, 0.001], [0.002, np.nan], [0.0, 0.0]])
    result = simulate_bybit_market_pnl(
        weights,
        np.full((3, 2), 100.0),
        np.zeros((3, 2)),
        one_way_cost=0.0,
        initial_equity_usd=1_000.0,
        gross_leverage=1.0,
        qty_step=np.full(2, 0.001),
        min_order_qty=np.full(2, 0.001),
        min_notional_value=np.full(2, 1.0),
        max_market_order_qty=np.full(2, 1e9),
        funding_factor=factor,
    )
    # Day 0: long pays 0.1%, short receives 0.1% -> net zero. Day 1: long pays 0.2% of half the book.
    np.testing.assert_allclose(result.funding_pnl[:2], [0.0, -0.001], atol=1e-12)
    np.testing.assert_allclose(result.funding_uncovered_notional[1], 500.0)
    np.testing.assert_allclose(result.pnl[:2], result.funding_pnl[:2])
    np.testing.assert_allclose(target_weight_funding(weights, factor)[:2], [0.0, -0.001], atol=1e-9)


def test_net_funding_target_subtracts_settlements_of_the_holding_days():
    from research import make_targets

    t_count, horizon = 12, 3
    opens = np.ones((t_count, 2))
    funding = np.zeros((t_count, 2))
    funding[:, 0] = np.arange(t_count) * 0.01  # day d pays 0.01 * d
    ctx = {
        "open": opens,
        "beta60": np.zeros((t_count, 2), np.float32),
        "eligible_max": np.ones((t_count, 2), bool),
        "btc_idx": 1,
        "funding_rate_day": funding,
    }
    plain = make_targets(ctx, horizon)["residual_rank"]
    net = make_targets(ctx, horizon, "residual_net_funding")["residual_rank"]
    # Decision at close 0 holds days 1..3 and pays 0.06, so asset 0 ranks below the zero-funding asset.
    assert plain[0, 0] == plain[0, 1]
    assert net[0, 0] < net[0, 1]
    assert np.isnan(net[t_count - horizon - 1 :]).all()
