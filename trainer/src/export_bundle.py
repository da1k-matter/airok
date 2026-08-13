"""Export the final causal LightGBM fold as an immutable Rust-consumable bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig, load_config
from data import load_panel
from research import build_features, make_targets, panel_to_rows


def canonical_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def final_retrain_date(
    cutoff: pd.Timestamp,
    prediction_start: pd.Timestamp,
    retrain_days: int,
) -> pd.Timestamp:
    date = prediction_start
    while date + pd.Timedelta(days=retrain_days) <= cutoff:
        date += pd.Timedelta(days=retrain_days)
    return date


def parameters(config: ExperimentConfig, seed: int) -> dict[str, object]:
    defaults: dict[str, object] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "label_gain": list(range(10)),
        "lambdarank_truncation_level": 30,
        "force_col_wise": True,
        "verbosity": -1,
    }
    defaults.update(config.section("backend").get("params", {}))
    defaults.update(
        {
            "random_state": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
        }
    )
    return defaults


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/lightgbm_h7.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--bundle", type=Path, required=True, help="Destination under ../live/models/")
    arguments = parser.parse_args()
    if arguments.bundle.exists():
        raise SystemExit(f"refusing to overwrite existing bundle: {arguments.bundle}")

    from lightgbm import LGBMRanker

    config = load_config(arguments.config, data_dir_override=arguments.data)
    if config.backend != "lightgbm":
        raise SystemExit(f"expected a LightGBM config, got {config.backend}")
    if config.section("ensemble")["method"] != "median_rank":
        raise SystemExit("only median_rank bundles are accepted by the Rust live strategy")
    data = config.section("data")
    training = config.section("training")
    horizon_days = int(config.section("model")["horizon"])
    ctx = build_features(
        load_panel(config.data_dir),
        min_age=int(data["min_history_days"]),
        max_universe=int(data["feature_universe_max"]),
    )
    rows = panel_to_rows(ctx)
    target = make_targets(ctx, horizon_days)["decile"]
    labels = target[rows["date_idx"], rows["asset_idx"]]
    dates = pd.DatetimeIndex(ctx["dates"])
    cutoff = dates[-1].normalize()
    prediction_date = final_retrain_date(
        cutoff,
        pd.Timestamp(training["prediction_start"]),
        int(training["retrain_every_days"]),
    )
    prediction_index = int(np.searchsorted(dates.values, np.datetime64(prediction_date), side="left"))
    train_end = prediction_index - horizon_days - 2
    train_start = max(0, train_end - int(training["train_window_days"]) + 1)
    train_mask = (
        (rows["date_idx"] >= train_start)
        & (rows["date_idx"] <= train_end)
        & np.isfinite(labels)
        & (((rows["date_idx"] - train_start) % int(training["train_stride"])) == 0)
    )
    selected = np.flatnonzero(train_mask)
    x_train = rows["X"][selected]
    y_train = labels[selected].astype(np.int32)
    train_dates = rows["date_idx"][selected]
    groups = np.unique(train_dates, return_counts=True)[1].astype(np.int32)
    if selected.size < int(training.get("minimum_train_rows", 5000)):
        raise SystemExit(f"only {selected.size} usable training rows")

    arguments.bundle.mkdir(parents=True)
    feature_contract = {"version": "v2_numeric_cross_section", "names": list(ctx["feature_names"])}
    universe = {"symbols": list(ctx["symbols"])}
    (arguments.bundle / "feature_contract.json").write_bytes(canonical_json(feature_contract))
    (arguments.bundle / "universe.json").write_bytes(canonical_json(universe))
    models: list[dict[str, object]] = []
    for seed in (int(value) for value in training["seeds"]):
        model = LGBMRanker(**parameters(config, seed))
        model.fit(x_train, y_train, group=groups)
        name = f"seed_{seed}.txt"
        path = arguments.bundle / name
        model.booster_.save_model(str(path))
        models.append({"seed": seed, "file": name, "sha256": digest(path.read_bytes())})
    manifest = {
        "bundle_id": f"lightgbm-h7-{cutoff:%Y%m%d}",
        "backend": "lightgbm",
        "horizon_days": horizon_days,
        "aggregation": "median_rank",
        "cutoff_date": str(cutoff.date()),
        "feature_schema_sha256": digest(canonical_json(feature_contract)),
        "universe_sha256": digest(canonical_json(universe)),
        "models": models,
    }
    (arguments.bundle / "manifest.json").write_bytes(canonical_json(manifest))
    print(
        f"created {arguments.bundle} with {selected.size} rows from "
        f"{dates[train_start].date()} through {dates[train_end].date()} "
        f"for final retrain {prediction_date.date()}"
    )


if __name__ == "__main__":
    main()
