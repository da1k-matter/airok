from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALLOWED_AGGREGATIONS = {"mean_rank", "median_rank", "portfolio_average"}
ALLOWED_WEIGHTING = {"equal", "invvol"}


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path
    root: Path

    @property
    def name(self) -> str:
        return str(self.raw["name"])

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Config section {name!r} must be a mapping")
        return value

    def resolve(self, value: str | Path) -> Path:
        expanded = os.path.expandvars(os.path.expanduser(str(value)))
        path = Path(expanded)
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.section("data")["directory"])

    @property
    def cache_dir(self) -> Path:
        return self.resolve(self.section("cache").get("directory", ".cache"))

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def find_project_root(config_path: Path) -> Path:
    config_path = config_path.resolve()
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return config_path.parent.parent


def _validate(raw: dict[str, Any]) -> None:
    required = {
        "name",
        "lightgbm",
        "data",
        "models",
        "training",
        "ensemble",
        "portfolio",
        "costs",
        "evaluation",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    lightgbm = raw["lightgbm"]
    if not isinstance(lightgbm, dict) or not isinstance(lightgbm.get("params", {}), dict):
        raise TypeError("lightgbm.params must be a mapping")

    models = raw["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    horizons: list[int] = []
    weights: list[float] = []
    for model in models:
        if not isinstance(model, dict):
            raise TypeError("Each model entry must be a mapping")
        horizon = int(model["horizon"])
        if horizon <= 0:
            raise ValueError("Model horizons must be positive")
        horizons.append(horizon)
        weights.append(float(model.get("weight", 0.0)))
    if len(set(horizons)) != len(horizons):
        raise ValueError("Model horizons must be unique")
    if abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError(f"Model horizon weights must sum to 1, got {sum(weights)}")

    training = raw["training"]
    seeds = [int(seed) for seed in training.get("seeds", [])]
    if not seeds:
        raise ValueError("training.seeds must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds must be unique")
    for key in ("train_window_days", "retrain_every_days", "train_stride"):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")

    ensemble = raw["ensemble"]
    method = str(ensemble.get("method", "")).lower()
    if method not in ALLOWED_AGGREGATIONS:
        raise ValueError(f"ensemble.method must be one of {sorted(ALLOWED_AGGREGATIONS)}")

    portfolio = raw["portfolio"]
    if str(portfolio.get("weighting", "")).lower() not in ALLOWED_WEIGHTING:
        raise ValueError(f"portfolio.weighting must be one of {sorted(ALLOWED_WEIGHTING)}")
    has_count = portfolio.get("tail_count") is not None
    has_fraction = portfolio.get("tail_fraction") is not None
    if has_count == has_fraction:
        raise ValueError("Set exactly one of portfolio.tail_count or portfolio.tail_fraction")
    if has_count and int(portfolio["tail_count"]) < 1:
        raise ValueError("portfolio.tail_count must be positive")
    if has_fraction and not 0 < float(portfolio["tail_fraction"]) < 0.5:
        raise ValueError("portfolio.tail_fraction must be between 0 and 0.5")


def load_config(path: str | Path, data_dir_override: str | Path | None = None) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("YAML root must be a mapping")
    raw = deepcopy(raw)
    if data_dir_override is not None:
        raw.setdefault("data", {})["directory"] = str(data_dir_override)
    _validate(raw)
    return ExperimentConfig(raw=raw, path=config_path, root=find_project_root(config_path))
