from pathlib import Path

from ranktrend.config import load_config


def test_best_config_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "best_v2.yaml")
    assert cfg.name == "best_v2"
    assert cfg.data_dir.name == "1d"
    assert sum(m["weight"] for m in cfg.raw["models"]) == 1.0
