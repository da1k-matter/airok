import numpy as np
import pandas as pd

from series import load_daily_series


def test_snapshots_shift_to_the_previous_close_and_bars_stay_on_their_day(tmp_path):
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    pd.DataFrame({"timestamp": ["2024-01-02 00:00:00", "2024-01-03 00:00:00"], "open_interest": [5.0, 6.0]}).to_csv(
        tmp_path / "X_oi.csv", index=False
    )
    pd.DataFrame({"timestamp": ["2024-01-02"], "close": [0.01]}).to_csv(tmp_path / "X_premium.csv", index=False)

    oi = load_daily_series(tmp_path, "oi", "open_interest", dates, ["X", "Y"], snapshot=True)
    premium = load_daily_series(tmp_path, "premium", "close", dates, ["X"], snapshot=False)

    np.testing.assert_allclose(oi[:2, 0], [5.0, 6.0])  # state at 00:00 Jan 2 is the close of Jan 1
    assert np.isnan(oi[2, 0]) and np.isnan(oi[:, 1]).all()
    assert np.isnan(premium[0, 0]) and premium[1, 0] == 0.01
