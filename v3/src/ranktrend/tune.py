from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import ExperimentConfig
from .pipeline import build_ensemble_weights, get_context, train_models
from .portfolio import build_weights, simulate_pnl
from .reporting import metrics, save_plots

LOG = logging.getLogger(__name__)


def _require_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Optuna is required for tuning. Install the project dependencies in the configured conda environment."
        ) from exc
    return optuna


def _tuning(config: ExperimentConfig) -> dict[str, Any]:
    tuning = config.raw.get("tuning")
    if not isinstance(tuning, dict):
        raise ValueError("The tune command requires a tuning mapping in the YAML config")
    if not isinstance(tuning.get("search_space"), dict):
        raise ValueError("tuning.search_space must be a mapping")
    if len(config.section("training").get("seeds", [])) != 3:
        raise ValueError("NSGA-II tuning requires exactly three fixed training seeds")
    return tuning


def _range(space: dict[str, Any], name: str) -> list[Any]:
    value = space.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"tuning.search_space.{name} must be a non-empty list")
    return value


def _study_storage(config: ExperimentConfig, tuning: dict[str, Any]) -> str:
    storage = str(tuning["storage"])
    if "://" in storage:
        return storage
    path = config.resolve(storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _output_dir(config: ExperimentConfig, tuning: dict[str, Any]) -> Path:
    output = config.resolve(tuning.get("output_directory", "outputs/optuna")) / str(tuning["study_name"])
    output.mkdir(parents=True, exist_ok=True)
    return output


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _choice(trial: Any, name: str, choices: list[Any]) -> Any:
    return trial.suggest_categorical(name, choices)


def _suggest_int(trial: Any, name: str, bounds: list[Any], *, log: bool = False) -> int:
    if len(bounds) not in {2, 3}:
        raise ValueError(f"tuning.search_space.{name} must contain two or three values")
    kwargs: dict[str, Any] = {"log": log}
    if len(bounds) == 3:
        kwargs["step"] = int(bounds[2])
    return int(trial.suggest_int(name, int(bounds[0]), int(bounds[1]), **kwargs))


def _suggest_float(trial: Any, name: str, bounds: list[Any], *, log: bool = False) -> float:
    if len(bounds) not in {2, 3}:
        raise ValueError(f"tuning.search_space.{name} must contain two or three values")
    kwargs: dict[str, Any] = {"log": log}
    if len(bounds) == 3:
        kwargs["step"] = float(bounds[2])
    return float(trial.suggest_float(name, float(bounds[0]), float(bounds[1]), **kwargs))


def _suggest_trial_config(base: ExperimentConfig, trial: Any) -> ExperimentConfig:
    raw = copy.deepcopy(base.raw)
    tuning = _tuning(base)
    space = tuning["search_space"]

    n_horizons = int(_choice(trial, "n_horizons", [1, 2]))
    if n_horizons == 1:
        models = [{"horizon": _suggest_int(trial, "single_horizon", _range(space, "single_horizon")), "weight": 1.0}]
    else:
        short = _suggest_int(trial, "short_horizon", _range(space, "short_horizon"))
        gap = _suggest_int(trial, "horizon_gap", _range(space, "horizon_gap"))
        long = short + gap
        if long > int(space["horizon_2_max"]):
            raise _require_optuna().TrialPruned("Second horizon exceeds horizon_2_max")
        weight = _suggest_float(trial, "horizon_1_weight", _range(space, "horizon_1_weight"))
        models = [
            {"horizon": short, "weight": weight},
            {"horizon": long, "weight": 1.0 - weight},
        ]

    params = raw["lightgbm"]["params"]
    params.update(
        {
            "n_estimators": _suggest_int(trial, "n_estimators", _range(space, "n_estimators")),
            "learning_rate": _suggest_float(trial, "learning_rate", _range(space, "learning_rate"), log=True),
            "num_leaves": int(_choice(trial, "num_leaves", _range(space, "num_leaves"))),
            "min_child_samples": _suggest_int(
                trial, "min_child_samples", _range(space, "min_child_samples"), log=True
            ),
            "subsample_freq": int(_choice(trial, "subsample_freq", [0, 1])),
            "colsample_bytree": _suggest_float(trial, "colsample_bytree", _range(space, "colsample_bytree")),
            "reg_alpha": _suggest_float(trial, "reg_alpha", _range(space, "reg_alpha"), log=True),
            "reg_lambda": _suggest_float(trial, "reg_lambda", _range(space, "reg_lambda"), log=True),
            "max_bin": int(_choice(trial, "max_bin", _range(space, "max_bin"))),
            "lambdarank_truncation_level": int(
                _choice(trial, "lambdarank_truncation_level", _range(space, "lambdarank_truncation_level"))
            ),
            "sigmoid": _suggest_float(trial, "sigmoid", _range(space, "sigmoid"), log=True),
        }
    )
    if int(params["subsample_freq"]) == 1:
        params["subsample"] = _suggest_float(trial, "subsample", _range(space, "subsample"))
    else:
        params["subsample"] = 1.0

    training = raw["training"]
    training.update(
        {
            "train_window_days": int(_choice(trial, "train_window_days", _range(space, "train_window_days"))),
            "retrain_every_days": int(
                _choice(trial, "retrain_every_days", _range(space, "retrain_every_days"))
            ),
            "train_stride": int(_choice(trial, "train_stride", _range(space, "train_stride"))),
            "prediction_end": str(raw["evaluation"]["validation_end"]),
        }
    )
    raw["models"] = models
    raw["ensemble"]["method"] = str(_choice(trial, "seed_aggregation", ["mean_rank", "median_rank"]))
    portfolio = raw["portfolio"]
    portfolio.update(
        {
            "universe_size": int(_choice(trial, "universe_size", _range(space, "universe_size"))),
            "tail_count": _suggest_int(trial, "tail_count", _range(space, "tail_count")),
            "weighting": str(_choice(trial, "weighting", ["equal", "invvol"])),
            "smoothing_days": int(_choice(trial, "smoothing_days", _range(space, "smoothing_days"))),
        }
    )
    portfolio.pop("tail_fraction", None)
    raw["name"] = f"{base.name}_trial_{trial.number:05d}"
    return ExperimentConfig(raw=raw, path=base.path, root=base.root)


def _evaluate_period(
    config: ExperimentConfig,
    ctx: dict[str, Any],
    start: str,
    end: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], pd.DataFrame]:
    predictions = train_models(config, ctx, reuse=True)
    weights, seed_scores = build_ensemble_weights(config, ctx, predictions)
    dates: pd.DatetimeIndex = ctx["dates"]
    start_idx = int(np.searchsorted(dates.values, np.datetime64(start), side="left"))
    end_idx = min(
        int(np.searchsorted(dates.values, np.datetime64(end), side="right")) - 1,
        len(dates) - 2,
    )
    if start_idx > end_idx:
        raise ValueError(f"Evaluation range {start}..{end} is outside the available panel")

    def score_weights(value: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        current = value.copy()
        current[:start_idx] = 0.0
        current[end_idx + 1 :] = 0.0
        pnl, turnover = simulate_pnl(
            current,
            np.asarray(ctx["oo_ret"]),
            float(config.section("costs")["one_way_bps"]) / 10_000.0,
        )
        result, _ = metrics(
            pnl,
            turnover,
            dates,
            str(dates[start_idx].date()),
            str(dates[end_idx].date()),
            int(config.section("evaluation").get("periods_per_year", 365)),
        )
        return result, current, pnl, turnover

    aggregate, aggregate_weights, pnl, turnover = score_weights(weights)
    per_seed = {
        int(seed): score_weights(build_weights(ctx, score, config.section("portfolio")))[0]
        for seed, score in seed_scores.items()
    }
    mask = np.zeros(len(dates), dtype=bool)
    mask[start_idx : end_idx + 1] = True
    daily = pd.DataFrame(
        {
            "date": dates[mask],
            "net_return": pnl[mask],
            "turnover": turnover[mask],
            "gross_exposure": np.abs(aggregate_weights[mask]).sum(axis=1),
            "net_exposure": aggregate_weights[mask].sum(axis=1),
        }
    )
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    return aggregate, per_seed, daily


def _write_period_artifacts(
    output: Path,
    label: str,
    metrics_row: dict[str, Any],
    daily: pd.DataFrame,
    per_seed: dict[int, dict[str, Any]],
) -> None:
    daily.to_csv(output / f"{label}_daily.csv", index=False)
    save_plots(daily, output, prefix=f"{label}_")
    (output / f"{label}_summary.json").write_text(
        json.dumps({"metrics": metrics_row, "per_seed_metrics": per_seed}, indent=2), encoding="utf-8"
    )


def _anchor_params() -> list[dict[str, Any]]:
    common = {
        "n_estimators": 160,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 6.0,
        "max_bin": 127,
        "lambdarank_truncation_level": 30,
        "sigmoid": 1.0,
        "train_window_days": 730,
        "retrain_every_days": 56,
        "train_stride": 3,
        "universe_size": 75,
        "weighting": "invvol",
        "smoothing_days": 21,
    }
    v1_like = {
        **common,
        "n_horizons": 2,
        "short_horizon": 7,
        "horizon_gap": 3,
        "horizon_1_weight": 0.5,
        "subsample_freq": 0,
        "seed_aggregation": "mean_rank",
        "tail_count": 5,
    }
    v2_like = {
        **common,
        "n_horizons": 1,
        "single_horizon": 7,
        "subsample_freq": 1,
        "subsample": 0.85,
        "seed_aggregation": "median_rank",
        "tail_count": 6,
    }
    return [v1_like, v2_like]


def _top_pareto_trials(study: Any, limit: int = 25) -> list[Any]:
    frontier = [trial for trial in study.best_trials if trial.values is not None]
    if len(frontier) <= limit:
        return sorted(frontier, key=lambda trial: trial.number)

    values = np.asarray([trial.values for trial in frontier], dtype=np.float64)
    low = values.min(axis=0)
    spread = values.max(axis=0) - low
    normalized = np.divide(values - low, spread, out=np.ones_like(values), where=spread > 1e-12)
    order = sorted(
        range(len(frontier)),
        key=lambda index: (
            -float(np.min(normalized[index])),
            -float(np.sum(normalized[index])),
            frontier[index].number,
        ),
    )
    return [frontier[index] for index in order[:limit]]


def _write_study_artifacts(study: Any, output: Path, config: ExperimentConfig) -> None:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {"number": trial.number, "state": trial.state.name, **trial.params}
        if trial.values is not None:
            row["validation_cagr"] = trial.values[0]
            row["validation_sharpe"] = trial.values[1]
        metrics_attr = trial.user_attrs.get("validation_metrics")
        if isinstance(metrics_attr, dict):
            row.update({f"validation_{key}": value for key, value in metrics_attr.items()})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("number") if rows else pd.DataFrame()
    table.to_csv(output / "trials.csv", index=False)
    frontier = pd.DataFrame(
        [
            {
                "number": trial.number,
                "validation_cagr": trial.values[0],
                "validation_sharpe": trial.values[1],
                **trial.params,
            }
            for trial in study.best_trials
        ]
    )
    frontier.to_csv(output / "pareto_frontier.csv", index=False)
    selected = _top_pareto_trials(study)
    pd.DataFrame(
        [
            {
                "trial": trial.number,
                "validation_cagr": trial.values[0],
                "validation_sharpe": trial.values[1],
                **trial.params,
            }
            for trial in selected
        ]
    ).to_csv(output / "pareto_top_25.csv", index=False)
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(config.raw, sort_keys=False), encoding="utf-8"
    )
    (output / "study_summary.json").write_text(
        json.dumps(
            {
                "study_name": study.study_name,
                "directions": [direction.name for direction in study.directions],
                "completed_trials": len([trial for trial in study.trials if trial.values is not None]),
                "pareto_trial_numbers": [trial.number for trial in study.best_trials],
                "test_trial_numbers": [trial.number for trial in selected],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_tuning(config: ExperimentConfig, n_trials: int | None = None) -> Path:
    optuna = _require_optuna()
    tuning = _tuning(config)
    target_trials = int(tuning["n_trials"]) if n_trials is None else int(n_trials)
    if target_trials < 1:
        raise ValueError("The number of requested trials must be positive")
    output = _output_dir(config, tuning)
    sampler = optuna.samplers.NSGAIISampler(
        seed=int(tuning.get("sampler_seed", 0)),
        population_size=int(tuning.get("population_size", 24)),
    )
    study = optuna.create_study(
        study_name=str(tuning["study_name"]),
        storage=_study_storage(config, tuning),
        directions=["maximize", "maximize"],
        sampler=sampler,
        load_if_exists=True,
    )
    if not study.trials:
        for params in _anchor_params():
            study.enqueue_trial(params, skip_if_exists=True)

    ctx = get_context(config, reuse=True)
    validation_start = str(config.section("evaluation")["validation_start"])
    validation_end = str(config.section("evaluation")["validation_end"])

    def objective(trial: Any) -> tuple[float, float]:
        trial_config = _suggest_trial_config(config, trial)
        aggregate, per_seed, _ = _evaluate_period(trial_config, ctx, validation_start, validation_end)
        trial.set_user_attr("resolved_config", _json_safe(trial_config.raw))
        trial.set_user_attr("validation_metrics", aggregate)
        trial.set_user_attr("per_seed_validation_metrics", per_seed)
        LOG.info(
            "trial=%d CAGR=%.2f%% Sharpe=%.3f horizons=%s aggregation=%s",
            trial.number,
            100.0 * float(aggregate["cagr"]),
            float(aggregate["sharpe"]),
            [item["horizon"] for item in trial_config.raw["models"]],
            trial_config.section("ensemble")["method"],
        )
        return float(aggregate["cagr"]), float(aggregate["sharpe"])

    completed = len([trial for trial in study.trials if trial.values is not None])
    remaining = max(0, target_trials - completed)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    _write_study_artifacts(study, output, config)
    selected = _top_pareto_trials(study)
    if selected:
        evaluate_trials(config, [trial.number for trial in selected])
    return output


def evaluate_trials(config: ExperimentConfig, trial_numbers: list[int]) -> Path:
    optuna = _require_optuna()
    tuning = _tuning(config)
    output = _output_dir(config, tuning) / "holdout"
    output.mkdir(parents=True, exist_ok=True)
    study = optuna.load_study(study_name=str(tuning["study_name"]), storage=_study_storage(config, tuning))
    selected = {trial.number: trial for trial in study.trials if trial.number in set(trial_numbers)}
    missing = sorted(set(trial_numbers) - set(selected))
    if missing:
        raise ValueError(f"Unknown trial numbers: {missing}")

    ctx = get_context(config, reuse=True)
    evaluation = config.section("evaluation")
    rows: list[dict[str, Any]] = []
    for number in trial_numbers:
        trial = selected[number]
        raw = trial.user_attrs.get("resolved_config")
        if not isinstance(raw, dict):
            raise ValueError(f"Trial {number} has no resolved configuration")
        validation_config = ExperimentConfig(raw=copy.deepcopy(raw), path=config.path, root=config.root)
        validation, validation_per_seed, validation_daily = _evaluate_period(
            validation_config,
            ctx,
            str(evaluation["validation_start"]),
            str(evaluation["validation_end"]),
        )
        candidate = ExperimentConfig(raw=copy.deepcopy(raw), path=config.path, root=config.root)
        candidate.raw["training"]["prediction_end"] = str(evaluation["holdout_end"])
        aggregate, per_seed, holdout_daily = _evaluate_period(
            candidate,
            ctx,
            str(evaluation["holdout_start"]),
            str(evaluation["holdout_end"]),
        )
        rows.append(
            {
                "trial": number,
                "validation_cagr": validation["cagr"],
                "validation_sharpe": validation["sharpe"],
                "holdout_cagr": aggregate["cagr"],
                "holdout_sharpe": aggregate["sharpe"],
                "holdout_max_dd": aggregate["max_dd"],
                "holdout_turnover_pa": aggregate["turnover_pa"],
                "per_seed_holdout_metrics": json.dumps(per_seed),
            }
        )
        (output / f"trial_{number:05d}_resolved.yaml").write_text(
            yaml.safe_dump(candidate.raw, sort_keys=False), encoding="utf-8"
        )
        trial_output = output / f"trial_{number:05d}"
        trial_output.mkdir(parents=True, exist_ok=True)
        _write_period_artifacts(
            trial_output,
            "validation",
            validation,
            validation_daily,
            validation_per_seed,
        )
        _write_period_artifacts(trial_output, "test", aggregate, holdout_daily, per_seed)
    pd.DataFrame(rows).to_csv(output / "selected_trials.csv", index=False)
    return output


def evaluate_trial_ensemble(config: ExperimentConfig, trial_numbers: list[int]) -> Path:
    """Evaluate an equal-weight ensemble of previously completed trial portfolios."""
    if not trial_numbers:
        raise ValueError("At least one trial is required for an ensemble")
    if len(set(trial_numbers)) != len(trial_numbers):
        raise ValueError("Trial numbers must be unique in an ensemble")

    optuna = _require_optuna()
    tuning = _tuning(config)
    root = _output_dir(config, tuning) / "holdout"
    study = optuna.load_study(study_name=str(tuning["study_name"]), storage=_study_storage(config, tuning))
    selected = {trial.number: trial for trial in study.trials if trial.number in set(trial_numbers)}
    missing = sorted(set(trial_numbers) - set(selected))
    if missing:
        raise ValueError(f"Unknown trial numbers: {missing}")

    candidates: list[tuple[int, ExperimentConfig]] = []
    for number in trial_numbers:
        raw = selected[number].user_attrs.get("resolved_config")
        if not isinstance(raw, dict):
            raise ValueError(f"Trial {number} has no resolved configuration")
        candidates.append((number, ExperimentConfig(raw=copy.deepcopy(raw), path=config.path, root=config.root)))

    ctx = get_context(config, reuse=True)
    dates: pd.DatetimeIndex = ctx["dates"]
    evaluation = config.section("evaluation")
    output = root / ("ensemble_" + "_".join(f"{number:05d}" for number in trial_numbers))
    output.mkdir(parents=True, exist_ok=True)

    def evaluate_period(label: str, start: str, end: str, prediction_end: str) -> dict[str, Any]:
        portfolios: list[np.ndarray] = []
        for number, base_candidate in candidates:
            candidate = ExperimentConfig(raw=copy.deepcopy(base_candidate.raw), path=config.path, root=config.root)
            candidate.raw["training"]["prediction_end"] = prediction_end
            predictions = train_models(candidate, ctx, reuse=True)
            weights, _ = build_ensemble_weights(candidate, ctx, predictions)
            portfolios.append(weights)
            LOG.info("ensemble %s: loaded trial=%d", label, number)

        weights = np.mean(np.stack(portfolios, axis=0), axis=0).astype(np.float32)
        start_idx = int(np.searchsorted(dates.values, np.datetime64(start), side="left"))
        end_idx = min(
            int(np.searchsorted(dates.values, np.datetime64(end), side="right")) - 1,
            len(dates) - 2,
        )
        current = weights.copy()
        current[:start_idx] = 0.0
        current[end_idx + 1 :] = 0.0
        pnl, turnover = simulate_pnl(
            current,
            np.asarray(ctx["oo_ret"]),
            float(config.section("costs")["one_way_bps"]) / 10_000.0,
        )
        metrics_row, _ = metrics(
            pnl,
            turnover,
            dates,
            str(dates[start_idx].date()),
            str(dates[end_idx].date()),
            int(evaluation.get("periods_per_year", 365)),
        )
        mask = np.zeros(len(dates), dtype=bool)
        mask[start_idx : end_idx + 1] = True
        daily = pd.DataFrame(
            {
                "date": dates[mask],
                "net_return": pnl[mask],
                "turnover": turnover[mask],
                "gross_exposure": np.abs(current[mask]).sum(axis=1),
                "net_exposure": current[mask].sum(axis=1),
            }
        )
        daily["equity"] = (1.0 + daily["net_return"]).cumprod()
        daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
        _write_period_artifacts(output, label, metrics_row, daily, {})
        return metrics_row

    validation = evaluate_period(
        "validation",
        str(evaluation["validation_start"]),
        str(evaluation["validation_end"]),
        str(evaluation["validation_end"]),
    )
    holdout = evaluate_period(
        "test",
        str(evaluation["holdout_start"]),
        str(evaluation["holdout_end"]),
        str(evaluation["holdout_end"]),
    )
    (output / "ensemble_metadata.json").write_text(
        json.dumps(
            {
                "aggregation": "equal_weight_portfolio_average",
                "trial_numbers": trial_numbers,
                "validation_metrics": validation,
                "test_metrics": holdout,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
