from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
EPS = 1e-12
FEATURE_VERSION = "v2_numeric_cross_section"


@dataclass(frozen=True)
class ModelSpec:
    horizon: int


def rolling_rank_pct(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    frame = pd.DataFrame(x)
    if mask is not None:
        frame = frame.where(mask)
    return frame.rank(axis=1, method="average", pct=True).to_numpy(dtype=np.float32)


def rolling_mean(x: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    minimum = min_periods if min_periods is not None else max(2, window // 2)
    return pd.DataFrame(x).rolling(window, min_periods=minimum).mean().to_numpy(dtype=np.float32)


def rolling_std(x: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    minimum = min_periods if min_periods is not None else max(3, window // 2)
    return pd.DataFrame(x).rolling(window, min_periods=minimum).std(ddof=0).to_numpy(dtype=np.float32)


def rolling_max(x: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    minimum = min_periods if min_periods is not None else max(2, window // 2)
    return pd.DataFrame(x).rolling(window, min_periods=minimum).max().to_numpy(dtype=np.float32)


def build_features(panel: dict[str, Any], min_age: int = 180, max_universe: int = 150) -> dict[str, Any]:
    symbols: list[str] = panel["symbols"]
    o = panel["open"]
    h = panel["high"]
    l = panel["low"]
    c = panel["close"]
    v = panel["volume"]
    first_idx = panel["first_idx"]
    t_count, _ = c.shape

    logc = np.log(c, where=np.isfinite(c), out=np.full_like(c, np.nan))
    ret1 = np.full_like(c, np.nan)
    ret1[1:] = logc[1:] - logc[:-1]
    oo_ret = np.full_like(c, np.nan)
    oo_ret[:-1] = o[1:] / o[:-1] - 1.0

    dollar_volume = c * v
    log_dollar_volume = np.log1p(dollar_volume)
    dvol20 = rolling_mean(log_dollar_volume, 20, 10)
    dvol60 = rolling_mean(log_dollar_volume, 60, 30)
    age = np.arange(t_count, dtype=np.int32)[:, None] - first_idx[None, :]
    base_valid = (age >= min_age) & np.isfinite(dvol20) & np.isfinite(c)
    liq_rank = (
        pd.DataFrame(dvol20)
        .where(base_valid)
        .rank(axis=1, ascending=False, method="first")
        .to_numpy(np.float32)
    )
    eligible_max = base_valid & (liq_rank <= max_universe)

    if "BTC" not in symbols:
        raise ValueError("BTC contract is required")
    btc_idx = symbols.index("BTC")
    btc_return = ret1[:, btc_idx].astype(np.float64)
    btc_2d = np.broadcast_to(btc_return[:, None], ret1.shape)
    mean_r60 = rolling_mean(ret1, 60, 30)
    mean_b60 = pd.Series(btc_return).rolling(60, min_periods=30).mean().to_numpy()
    mean_rb60 = rolling_mean(ret1 * btc_2d, 60, 30)
    mean_b2_60 = pd.Series(btc_return * btc_return).rolling(60, min_periods=30).mean().to_numpy()
    var_b60 = np.maximum(mean_b2_60 - mean_b60 * mean_b60, 1e-8)
    beta60 = (mean_rb60 - mean_r60 * mean_b60[:, None]) / var_b60[:, None]
    beta60 = np.clip(beta60, -3.0, 5.0).astype(np.float32)

    raw: dict[str, np.ndarray] = {}
    for window in (1, 2, 3, 5, 7, 14, 21, 30, 60, 90, 120):
        values = np.full_like(c, np.nan)
        if window == 1:
            values = ret1.copy()
        else:
            values[window:] = logc[window:] - logc[:-window]
        raw[f"ret_{window}"] = values

    for window in (5, 10, 20, 60):
        raw[f"vol_{window}"] = rolling_std(ret1, window)
    negative = np.minimum(ret1, 0.0)
    raw["downvol_20"] = np.sqrt(rolling_mean(negative * negative, 20, 10)).astype(np.float32)
    raw["downvol_60"] = np.sqrt(rolling_mean(negative * negative, 60, 30)).astype(np.float32)

    log_range = np.log(np.maximum(h, EPS) / np.maximum(l, EPS))
    raw["range_1"] = log_range
    raw["range_5"] = rolling_mean(log_range, 5, 3)
    raw["range_20"] = rolling_mean(log_range, 20, 10)
    denominator = np.maximum(h - l, EPS)
    raw["close_loc_1"] = (c - l) / denominator - 0.5
    raw["close_loc_5"] = rolling_mean(raw["close_loc_1"], 5, 3)
    raw["body_1"] = (c - o) / denominator
    raw["body_5"] = rolling_mean(raw["body_1"], 5, 3)

    raw["volume_5_20"] = rolling_mean(log_dollar_volume, 5, 3) - dvol20
    raw["volume_20_60"] = dvol20 - dvol60
    raw["dvol_level"] = dvol20
    raw["amihud_20"] = rolling_mean(np.abs(ret1) / np.maximum(dollar_volume, 1.0), 20, 10)
    for window in (20, 60, 120):
        raw[f"dist_high_{window}"] = c / rolling_max(c, window) - 1.0

    raw["beta_60"] = beta60
    raw["resmom_21"] = raw["ret_21"] - beta60 * raw["ret_21"][:, btc_idx][:, None]
    raw["resmom_60"] = raw["ret_60"] - beta60 * raw["ret_60"][:, btc_idx][:, None]
    raw["short_reversal"] = -raw["ret_3"]
    raw["mom_21_ex_3"] = raw["ret_21"] - raw["ret_3"]
    raw["mom_60_ex_7"] = raw["ret_60"] - raw["ret_7"]

    features: dict[str, np.ndarray] = {
        f"{name}_xrank": (rolling_rank_pct(values, eligible_max) - 0.5).astype(np.float32)
        for name, values in raw.items()
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        breadth = np.nanmean(np.where(eligible_max, ret1 > 0, np.nan), axis=1)
        dispersion = np.nanstd(np.where(eligible_max, ret1, np.nan), axis=1)
    regimes = {
        "mkt_btc_ret7": raw["ret_7"][:, btc_idx],
        "mkt_btc_ret21": raw["ret_21"][:, btc_idx],
        "mkt_btc_vol20": raw["vol_20"][:, btc_idx],
        "mkt_breadth": breadth,
        "mkt_dispersion": dispersion,
    }
    for name, values in regimes.items():
        features[name] = np.broadcast_to(values[:, None], c.shape).astype(np.float32)

    feature_names = list(features)
    feature_stack = np.stack([features[name] for name in feature_names], axis=2).astype(np.float32)
    valid_features = np.all(np.isfinite(feature_stack), axis=2)
    return {
        **panel,
        "oo_ret": oo_ret.astype(np.float32),
        "ret1": ret1.astype(np.float32),
        "dvol20": dvol20.astype(np.float32),
        "liq_rank": liq_rank,
        "eligible_max": eligible_max,
        "beta60": beta60,
        "vol20": raw["vol_20"].astype(np.float32),
        "btc_vol20": raw["vol_20"][:, btc_idx].astype(np.float32),
        "feature_names": feature_names,
        "feature_stack": feature_stack,
        "valid_features": valid_features,
        "btc_idx": btc_idx,
    }


def make_targets(ctx: dict[str, Any], horizon: int) -> dict[str, np.ndarray]:
    o = ctx["open"]
    beta = ctx["beta60"]
    eligible = ctx["eligible_max"]
    t_count, n_assets = o.shape
    future = np.full((t_count, n_assets), np.nan, np.float32)
    future[: -(horizon + 1)] = np.log(o[horizon + 1 :] / o[1:-horizon])
    btc_future = future[:, int(ctx["btc_idx"])]
    residual = future - beta * btc_future[:, None]
    mask = eligible & np.isfinite(future)
    residual_rank = rolling_rank_pct(residual, mask)
    decile = np.floor(np.clip(residual_rank, 0.0, 0.999999) * 10.0).astype(np.float32)
    decile[~np.isfinite(residual_rank)] = np.nan
    return {"residual_rank": residual_rank, "decile": decile}


def panel_to_rows(ctx: dict[str, Any]) -> dict[str, np.ndarray]:
    mask = ctx["eligible_max"] & ctx["valid_features"]
    date_idx, asset_idx = np.where(mask)
    return {
        "date_idx": date_idx.astype(np.int32),
        "asset_idx": asset_idx.astype(np.int32),
        "X": np.asarray(ctx["feature_stack"])[date_idx, asset_idx].astype(np.float32),
    }


def _groups_from_sorted_dates(date_idx: np.ndarray) -> np.ndarray:
    return np.unique(date_idx, return_counts=True)[1].astype(np.int32)


def _fit_predict_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    dates_train: np.ndarray,
    x_predict: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    from lightgbm import LGBMRanker

    defaults: dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "label_gain": list(range(10)),
        "lambdarank_truncation_level": 30,
        "n_estimators": 160,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 6.0,
        "max_bin": 127,
        "n_jobs": 4,
        "force_col_wise": True,
        "verbosity": -1,
    }
    defaults.update(params)
    defaults.update(
        {
            "random_state": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
        }
    )
    model = LGBMRanker(**defaults)
    model.fit(
        x_train,
        y_train.astype(np.int32),
        group=_groups_from_sorted_dates(dates_train),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRanker was fitted with feature names",
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )
        return model.predict(x_predict).astype(np.float32)


def _fit_predict_catboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    dates_train: np.ndarray,
    x_predict: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    from catboost import CatBoostRanker, Pool

    defaults: dict[str, Any] = {
        "loss_function": "YetiRankPairwise:mode=NDCG;top=8;type=Base",
        "iterations": 160,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "rsm": 0.80,
        "border_count": 127,
        "thread_count": 4,
        "allow_writing_files": False,
        "verbose": False,
    }
    defaults.update(params)
    defaults["random_seed"] = seed
    model = CatBoostRanker(**defaults)
    pool = Pool(x_train, label=y_train.astype(np.int32), group_id=dates_train)
    model.fit(pool, verbose=False)
    return model.predict(x_predict).astype(np.float32)


def train_walk_forward(
    ctx: dict[str, Any],
    rows: dict[str, np.ndarray],
    spec: ModelSpec,
    backend: str,
    backend_params: dict[str, Any],
    training: dict[str, Any],
    seed: int,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> np.ndarray:
    dates: pd.DatetimeIndex = ctx["dates"]
    t_count, n_assets = ctx["close"].shape
    date_rows = rows["date_idx"]
    asset_rows = rows["asset_idx"]
    x_all = rows["X"]
    target = make_targets(ctx, spec.horizon)["decile"]
    y_all = target[date_rows, asset_rows]
    scores = np.full((t_count, n_assets), np.nan, np.float32)

    retrain_days = int(training["retrain_every_days"])
    starts = pd.date_range(prediction_start.normalize(), prediction_end.normalize(), freq=f"{retrain_days}D")
    if len(starts) == 0 or starts[0] != prediction_start.normalize():
        starts = starts.insert(0, prediction_start.normalize())

    for fold, pstart in enumerate(starts, start=1):
        pend = min(pstart + pd.Timedelta(days=retrain_days - 1), prediction_end)
        p0 = int(np.searchsorted(dates.values, np.datetime64(pstart), side="left"))
        p1 = int(np.searchsorted(dates.values, np.datetime64(pend), side="right")) - 1
        if p0 >= t_count or p1 < 0 or p0 > p1:
            continue
        train_end = p0 - spec.horizon - 2
        train_start = max(0, train_end - int(training["train_window_days"]) + 1)
        train_mask = (
            (date_rows >= train_start)
            & (date_rows <= train_end)
            & np.isfinite(y_all)
            & (((date_rows - train_start) % int(training["train_stride"])) == 0)
        )
        predict_mask = (date_rows >= p0) & (date_rows <= p1)
        train_rows = np.flatnonzero(train_mask)
        predict_rows = np.flatnonzero(predict_mask)
        if train_rows.size < int(training.get("minimum_train_rows", 5000)) or predict_rows.size == 0:
            continue

        fit_predict = _fit_predict_lightgbm if backend == "lightgbm" else _fit_predict_catboost
        prediction = fit_predict(
            x_all[train_rows],
            y_all[train_rows],
            date_rows[train_rows],
            x_all[predict_rows],
            backend_params,
            seed,
        )
        scores[date_rows[predict_rows], asset_rows[predict_rows]] = prediction
        LOG.info(
            "%s h%d seed=%d fold=%02d %s..%s train=%s predict=%s",
            backend,
            spec.horizon,
            seed,
            fold,
            pstart.date(),
            pend.date(),
            f"{train_rows.size:,}",
            f"{predict_rows.size:,}",
        )
    return scores


def rank_model_scores(scores: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    return rolling_rank_pct(scores, eligible & np.isfinite(scores))


def aggregate_seed_ranks(seed_scores: list[np.ndarray], method: str) -> np.ndarray:
    stack = np.stack(seed_scores).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if method == "mean_rank":
            result = np.nanmean(stack, axis=0)
        elif method == "median_rank":
            result = np.nanmedian(stack, axis=0)
        else:
            raise ValueError(f"Score aggregation is unavailable for method={method!r}")
    return result.astype(np.float32)
