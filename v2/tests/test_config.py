from pathlib import Path

import pytest
import yaml

from ranktrend.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "backend", "horizons", "seeds"),
    [
        ("lightgbm_h7.yaml", "lightgbm", [7], 5),
        ("lightgbm_h7_h10.yaml", "lightgbm", [7, 10], 5),
        ("catboost_h7.yaml", "catboost", [7], 5),
        ("catboost_h7_h10.yaml", "catboost", [7, 10], 5),
        ("lightgbm_legacy_best_v2.yaml", "lightgbm", [7, 10], 1),
    ],
)
def test_presets_load(name, backend, horizons, seeds):
    config = load_config(ROOT / "configs" / name)
    assert config.backend == backend
    assert [item["horizon"] for item in config.raw["models"]] == horizons
    assert len(config.section("training")["seeds"]) == seeds


def test_rejects_tail_count_and_fraction_together(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "lightgbm_h7.yaml").read_text())
    raw["portfolio"]["tail_fraction"] = 0.1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="exactly one"):
        load_config(path)
