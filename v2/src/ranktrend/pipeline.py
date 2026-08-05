from __future__ import annotations

import hashlib
import importlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import ExperimentConfig
from .data import data_fingerprint, load_context, load_panel, save_context, validate_data
from .portfolio import average_portfolios, build_weights, simulate_pnl
from .reporting import environment_text, git_commit, metrics, save_plots, yearly_table
from .research import (
    FEATURE_VERSION,
    ModelSpec,
    aggregate_seed_ranks,
    blend_horizons,
    build_features,
    panel_to_rows,
    train_walk_forward,
)

LOG = logging.getLogger(__name__)


def model_specs(config: ExperimentConfig) -> list[ModelSpec]:
    return [
        ModelSpec(horizon=int(item["horizon"]), weight=float(item["weight"]))
        for item in config.raw["models"]
    ]


def feature_cache_dir(config: ExperimentConfig) -> Path:
    data = config.section("data")
    key = (
        f"{data_fingerprint(config.data_dir)}_{FEATURE_VERSION}"
        f"_age{data['min_history_days']}_u{data['feature_universe_max']}"
    )
    return config.cache_dir / f"features_{key}"


def prediction_cache_file(config: ExperimentConfig, spec: ModelSpec, seed: int) -> Path:
    relevant = {
        "backend": config.section("backend"),
        "data": config.section("data"),
        "training": config.section("training"),
        "horizon": spec.horizon,
        "seed": seed,
        "feature_version": FEATURE_VERSION,
        "data_fingerprint": data_fingerprint(config.data_dir),
    }
    digest = hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return config.cache_dir / "predictions" / f"{config.backend}_h{spec.horizon}_s{seed}_{digest}.npz"


def get_context(config: ExperimentConfig, reuse: bool = True) -> dict[str, Any]:
    cache = feature_cache_dir(config)
    if reuse and (cache / "meta.json").exists():
        LOG.info("Feature cache hit: %s", cache)
        return load_context(cache, mmap=True)

    LOG.info("Loading data from %s", config.data_dir)
    panel = load_panel(config.data_dir)
    data = config.section("data")
    LOG.info("Building features for %d assets x %d days", len(panel["symbols"]), len(panel["dates"]))
    ctx = build_features(
        panel,
        min_age=int(data["min_history_days"]),
        max_universe=int(data["feature_universe_max"]),
    )
    if cache.exists():
        shutil.rmtree(cache)
    save_context(ctx, cache)
    LOG.info("Feature cache saved: %s", cache)
    return ctx


def train_models(
    config: ExperimentConfig,
    ctx: dict[str, Any],
    reuse: bool = True,
) -> dict[int, list[np.ndarray]]:
    rows = panel_to_rows(ctx)
    training = config.section("training")
    prediction_start = pd.Timestamp(training["prediction_start"])
    prediction_end = pd.Timestamp(training.get("prediction_end", ctx["dates"][-1]))
    backend_params = config.section("backend").get("params", {})
    specs = model_specs(config)
    outputs: dict[int, list[np.ndarray]] = {}

    for seed in [int(value) for value in training["seeds"]]:
        seed_outputs: list[np.ndarray] = []
        for spec in specs:
            path = prediction_cache_file(config, spec, seed)
            if reuse and path.exists():
                LOG.info("Prediction cache hit: %s", path)
                scores = np.load(path, allow_pickle=False)["scores"]
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                scores = train_walk_forward(
                    ctx=ctx,
                    rows=rows,
                    spec=spec,
                    backend=config.backend,
                    backend_params=backend_params,
                    training=training,
                    seed=seed,
                    prediction_start=prediction_start,
                    prediction_end=prediction_end,
                )
                np.savez_compressed(path, scores=scores.astype(np.float32))
                LOG.info("Prediction cache saved: %s", path)
            seed_outputs.append(scores)
        outputs[seed] = seed_outputs
    return outputs


def build_ensemble_weights(
    config: ExperimentConfig,
    ctx: dict[str, Any],
    predictions: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    specs = model_specs(config)
    seed_scores = {
        seed: blend_horizons(arrays, specs, np.asarray(ctx["eligible_max"]))
        for seed, arrays in predictions.items()
    }
    ensemble = config.section("ensemble")
    method = str(ensemble["method"])
    portfolio = config.section("portfolio")

    if method == "portfolio_average":
        portfolios = [build_weights(ctx, score, portfolio) for score in seed_scores.values()]
        weights = average_portfolios(
            portfolios,
            renormalize=bool(ensemble.get("renormalize_portfolio_average", False)),
        )
    else:
        combined = aggregate_seed_ranks(list(seed_scores.values()), method)
        weights = build_weights(ctx, combined, portfolio)
    return weights.astype(np.float32), seed_scores


def _run_directory(config: ExperimentConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = config.root / "outputs" / "runs" / f"{stamp}_{config.name}"
    suffix = 1
    while output.exists():
        output = config.root / "outputs" / "runs" / f"{stamp}_{config.name}_{suffix}"
        suffix += 1
    output.mkdir(parents=True)
    return output


def _latest_weights_table(ctx: dict[str, Any], weights: np.ndarray, row: int) -> pd.DataFrame:
    symbols = np.asarray(ctx["symbols"], dtype=object)
    selected = np.flatnonzero(np.abs(weights[row]) > 1e-8)
    frame = pd.DataFrame(
        {
            "symbol": symbols[selected],
            "weight": weights[row, selected],
            "side": np.where(weights[row, selected] > 0, "long", "short"),
            "liquidity_rank": np.asarray(ctx["liq_rank"])[row, selected],
            "beta60": np.asarray(ctx["beta60"])[row, selected],
            "vol20": np.asarray(ctx["vol20"])[row, selected],
        }
    )
    return frame.sort_values("weight", ascending=False).reset_index(drop=True)


def backtest(
    config: ExperimentConfig,
    ctx: dict[str, Any],
    predictions: dict[int, list[np.ndarray]],
) -> tuple[Path, dict[str, Any]]:
    evaluation = config.section("evaluation")
    periods_per_year = int(evaluation.get("periods_per_year", 365))
    oos_start = str(evaluation["oos_start"])
    oos_end = str(evaluation.get("oos_end", ctx["dates"][-2]))
    latest_start = str(evaluation["latest_year_start"])
    dates: pd.DatetimeIndex = ctx["dates"]

    weights, seed_scores = build_ensemble_weights(config, ctx, predictions)
    start_idx = int(np.searchsorted(dates.values, np.datetime64(oos_start), side="left"))
    end_idx = min(
        int(np.searchsorted(dates.values, np.datetime64(oos_end), side="right")) - 1,
        len(dates) - 2,
    )
    weights[:start_idx] = 0.0
    weights[end_idx + 1 :] = 0.0

    cost = float(config.section("costs")["one_way_bps"]) / 10_000.0
    pnl, turnover = simulate_pnl(weights, np.asarray(ctx["oo_ret"]), cost)
    effective_end = str(dates[end_idx].date())
    full_metrics, full_monthly = metrics(pnl, turnover, dates, oos_start, effective_end, periods_per_year)
    latest_metrics, _ = metrics(pnl, turnover, dates, latest_start, effective_end, periods_per_year)
    summary: dict[str, Any] = {
        "model": {
            "backend": config.backend,
            "horizons": [spec.horizon for spec in model_specs(config)],
            "seeds": [int(seed) for seed in config.section("training")["seeds"]],
            "seed_aggregation": config.section("ensemble")["method"],
        },
        "full": full_metrics,
        "latest_year": latest_metrics,
    }

    output = _run_directory(config)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config.raw, sort_keys=False), encoding="utf-8")
    (output / "environment.txt").write_text(environment_text(), encoding="utf-8")

    mask = (dates >= pd.Timestamp(oos_start)) & (dates <= pd.Timestamp(effective_end))
    dated_weights = weights[mask]
    beta_exposure = np.nansum(dated_weights * np.asarray(ctx["beta60"])[mask], axis=1)
    daily = pd.DataFrame(
        {
            "date": dates[mask],
            "net_return": pnl[mask],
            "turnover": turnover[mask],
            "gross_exposure": np.abs(dated_weights).sum(axis=1),
            "net_exposure": dated_weights.sum(axis=1),
            "beta_exposure": beta_exposure,
        }
    )
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    daily.to_csv(output / "daily.csv", index=False)
    full_monthly.rename("return").to_csv(output / "monthly.csv", header=True)
    yearly_table(pnl, turnover, dates, oos_start, effective_end, periods_per_year).to_csv(
        output / "yearly.csv", index=False
    )

    fee_rows: list[dict[str, Any]] = []
    for bps in evaluation.get("fee_stress_bps", [5, 10, 15, 20, 30]):
        fee_pnl, fee_turnover = simulate_pnl(weights, np.asarray(ctx["oo_ret"]), float(bps) / 10_000.0)
        for label, start in (("full", oos_start), ("latest_year", latest_start)):
            row, _ = metrics(fee_pnl, fee_turnover, dates, start, effective_end, periods_per_year)
            fee_rows.append({"one_way_bps": bps, "period": label, **row})
    pd.DataFrame(fee_rows).to_csv(output / "fee_stress.csv", index=False)

    latest = _latest_weights_table(ctx, weights, end_idx)
    latest.to_csv(output / "latest_weights.csv", index=False)

    summary["exposures"] = {
        "average_gross": float(daily["gross_exposure"].mean()),
        "average_net": float(daily["net_exposure"].mean()),
        "average_absolute_beta": float(daily["beta_exposure"].abs().mean()),
        "mean_active_assets": float((np.abs(dated_weights) > 1e-8).sum(axis=1).mean()),
    }
    summary["seed_rank_correlation"] = _seed_rank_correlation(seed_scores, mask)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.config_hash,
        "data_fingerprint": data_fingerprint(config.data_dir),
        "git_commit": git_commit(config.root),
        "output": str(output),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_plots(daily, output)
    return output, summary


def _seed_rank_correlation(seed_scores: dict[int, np.ndarray], mask: np.ndarray) -> float | None:
    scores = list(seed_scores.values())
    if len(scores) < 2:
        return None
    correlations: list[float] = []
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            a = scores[left][mask].ravel()
            b = scores[right][mask].ravel()
            good = np.isfinite(a) & np.isfinite(b)
            if good.sum() > 100:
                correlations.append(float(np.corrcoef(a[good], b[good])[0, 1]))
    return float(np.mean(correlations)) if correlations else None


def doctor(config: ExperimentConfig) -> dict[str, Any]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    (config.root / "outputs").mkdir(parents=True, exist_ok=True)
    module = importlib.import_module(config.backend)
    diagnostics = validate_data(config.data_dir)
    diagnostics.update(
        {
            "config": str(config.path),
            "project_root": str(config.root),
            "cache_dir": str(config.cache_dir),
            "backend": config.backend,
            "backend_version": str(getattr(module, "__version__", "unknown")),
            "horizons": [spec.horizon for spec in model_specs(config)],
            "seeds": [int(seed) for seed in config.section("training")["seeds"]],
            "aggregation": config.section("ensemble")["method"],
        }
    )
    return diagnostics


def run(
    config: ExperimentConfig,
    reuse_features: bool = True,
    reuse_predictions: bool = True,
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    ctx = get_context(config, reuse=reuse_features)
    predictions = train_models(config, ctx, reuse=reuse_predictions)
    output, summary = backtest(config, ctx, predictions)
    LOG.info("Total elapsed: %.1fs", time.perf_counter() - started)
    return output, summary
