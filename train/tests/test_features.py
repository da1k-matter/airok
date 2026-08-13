import numpy as np
import pandas as pd

from research import build_features, make_targets


def test_feature_and_target_shapes():
    days = 240
    assets = 4
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    base = np.linspace(100.0, 150.0, days, dtype=np.float32)[:, None]
    scales = np.array([[1.0, 0.8, 1.2, 0.6]], dtype=np.float32)
    close = base * scales
    panel = {
        "dates": dates,
        "symbols": ["BTC", "ETH", "SOL", "XRP"],
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full((days, assets), 1_000_000.0, np.float32),
        "first_idx": np.zeros(assets, np.int32),
    }
    context = build_features(panel, min_age=30, max_universe=4)
    target = make_targets(context, 7)
    assert context["feature_stack"].shape[:2] == (days, assets)
    assert context["feature_stack"].shape[2] > 20
    assert target["decile"].shape == (days, assets)
