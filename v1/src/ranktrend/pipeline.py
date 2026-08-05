from __future__ import annotations

import hashlib
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
from .portfolio import build_weights, simulate_pnl
from .reporting import environment_text, git_commit, metrics, monthly_returns, save_plots, yearly_table
from .research import ModelSpec, build_features, ensemble_scores, panel_to_rows, train_walk_forward

LOG = logging.getLogger(__name__)


def feature_cache_dir(config: ExperimentConfig) -> Path:
    data = config.section("data")
    key = f"{data_fingerprint(config.data_dir)}_age{data['min_history_days']}_u{data['feature_universe_max']}"
    return config.cache_dir / f"features_{key}"


def prediction_cache_file(config: ExperimentConfig, model: dict[str, Any]) -> Path:
    relevant = {
        "data": config.section("data"),
        "training": config.section("training"),
        "model": model,
        "data_fingerprint": data_fingerprint(config.data_dir),
    }
    digest = hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return config.cache_dir / "predictions" / f"{model['name']}_{digest}.npz"


def get_context(config: ExperimentConfig, reuse: bool = True) -> dict[str, Any]:
    cache = feature_cache_dir(config)
    if reuse and (cache / "meta.json").exists():
        LOG.info("Feature cache hit: %s", cache)
        return load_context(cache, mmap=True)
    LOG.info("Loading %s", config.data_dir)
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


def train_models(config: ExperimentConfig, ctx: dict[str, Any], reuse: bool = True) -> list[np.ndarray]:
    rows = panel_to_rows(ctx)
    training = config.section("training")
    prediction_start = pd.Timestamp(training["prediction_start"])
    prediction_end = pd.Timestamp(ctx["dates"][-1])
    outputs: list[np.ndarray] = []
    for model_cfg in config.raw["models"]:
        path = prediction_cache_file(config, model_cfg)
        if reuse and path.exists():
            LOG.info("Prediction cache hit: %s", path)
            scores = np.load(path, allow_pickle=False)["scores"]
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            spec = ModelSpec(
                name=str(model_cfg["name"]),
                horizon=int(model_cfg["horizon"]),
                weight=float(model_cfg["weight"]),
            )
            scores = train_walk_forward(ctx, rows, spec, training, prediction_start, prediction_end)
            np.savez_compressed(path, scores=scores.astype(np.float32))
            LOG.info("Prediction cache saved: %s", path)
        outputs.append(scores)
    return outputs


def _run_directory(config: ExperimentConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = config.root / "outputs" / "runs" / f"{stamp}_{config.name}"
    counter = 1
    while output.exists():
        output = config.root / "outputs" / "runs" / f"{stamp}_{config.name}_{counter}"
        counter += 1
    output.mkdir(parents=True)
    return output


def backtest(config: ExperimentConfig, ctx: dict[str, Any], score_arrays: list[np.ndarray]) -> tuple[Path, dict[str, Any]]:
    evaluation = config.section("evaluation")
    periods_per_year = int(evaluation["periods_per_year"])
    oos_start = str(evaluation["oos_start"])
    oos_end = str(evaluation["oos_end"])
    latest_start = str(evaluation["latest_year_start"])
    dates: pd.DatetimeIndex = ctx["dates"]
    weights_cfg = [float(m["weight"]) for m in config.raw["models"]]
    blended = ensemble_scores(score_arrays, weights_cfg, ctx["eligible_max"])
    weights = build_weights(ctx, blended, config.section("portfolio"))

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
    latest_metrics, latest_monthly = metrics(pnl, turnover, dates, latest_start, effective_end, periods_per_year)
    summary = {"full": full_metrics, "latest_year": latest_metrics}

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
    yearly_table(pnl, turnover, dates, oos_start, effective_end, periods_per_year).to_csv(output / "yearly.csv", index=False)

    fee_rows = []
    for bps in (5, 10, 15, 20, 30):
        fee_pnl, fee_turnover = simulate_pnl(weights, np.asarray(ctx["oo_ret"]), bps / 10_000.0)
        for label, start in (("full", oos_start), ("latest_year", latest_start)):
            row, _ = metrics(fee_pnl, fee_turnover, dates, start, effective_end, periods_per_year)
            fee_rows.append({"one_way_bps": bps, "period": label, **row})
    pd.DataFrame(fee_rows).to_csv(output / "fee_stress.csv", index=False)

    summary["exposures"] = {
        "average_gross": float(daily["gross_exposure"].mean()),
        "average_net": float(daily["net_exposure"].mean()),
        "average_absolute_beta": float(daily["beta_exposure"].abs().mean()),
        "mean_active_assets": float((np.abs(dated_weights) > 1e-8).sum(axis=1).mean()),
    }
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


def verify_reference(config: ExperimentConfig, summary: dict[str, Any]) -> tuple[bool, list[str]]:
    if not config.reference_file.exists():
        return False, [f"Missing reference file: {config.reference_file}"]
    reference = json.loads(config.reference_file.read_text(encoding="utf-8"))
    tolerance = reference.get("tolerance", {})
    failures: list[str] = []
    for period in ("full", "latest_year"):
        expected = reference[period]
        actual = summary[period]
        for key in ("cagr", "sharpe", "max_dd", "turnover_pa"):
            limit = float(tolerance.get(key, 0.0))
            diff = abs(float(actual[key]) - float(expected[key]))
            if diff > limit:
                failures.append(f"{period}.{key}: actual={actual[key]:.8f}, expected={expected[key]:.8f}, diff={diff:.8f} > {limit}")
        month_limit = int(tolerance.get("positive_months", 0))
        month_diff = abs(int(actual["positive_months"]) - int(expected["positive_months"]))
        if month_diff > month_limit:
            failures.append(
                f"{period}.positive_months: actual={actual['positive_months']}, "
                f"expected={expected['positive_months']}, diff={month_diff} > {month_limit}"
            )
        if int(actual["n_months"]) != int(expected["n_months"]):
            failures.append(
                f"{period}.n_months: actual={actual['n_months']}, expected={expected['n_months']}"
            )
    return not failures, failures


def doctor(config: ExperimentConfig) -> dict[str, Any]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    (config.root / "outputs").mkdir(parents=True, exist_ok=True)
    diagnostics = validate_data(config.data_dir)
    diagnostics.update(
        {
            "config": str(config.path),
            "project_root": str(config.root),
            "cache_dir": str(config.cache_dir),
            "reference": str(config.reference_file),
        }
    )
    return diagnostics


def reproduce(config: ExperimentConfig, reuse_features: bool, reuse_predictions: bool) -> tuple[Path, dict[str, Any], bool, list[str]]:
    started = time.perf_counter()
    ctx = get_context(config, reuse=reuse_features)
    scores = train_models(config, ctx, reuse=reuse_predictions)
    output, summary = backtest(config, ctx, scores)
    ok, failures = verify_reference(config, summary)
    LOG.info("Total elapsed: %.1fs", time.perf_counter() - started)
    return output, summary, ok, failures
