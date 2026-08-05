from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.section("data")["directory"])

    @property
    def cache_dir(self) -> Path:
        return self.resolve(self.section("cache").get("directory", ".cache"))

    @property
    def reference_file(self) -> Path:
        return self.resolve(self.section("reference")["file"])

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


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("YAML root must be a mapping")
    required = {"name", "data", "models", "training", "portfolio", "costs", "evaluation"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    models = raw["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")
    weight_sum = sum(float(m.get("weight", 0.0)) for m in models)
    if abs(weight_sum - 1.0) > 1e-8:
        raise ValueError(f"Model ensemble weights must sum to 1, got {weight_sum}")
    return ExperimentConfig(raw=raw, path=config_path, root=find_project_root(config_path))
