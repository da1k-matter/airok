from pathlib import Path

import pytest
import yaml

from config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "backend", "horizon", "seeds"),
    [
        ("lightgbm_h7.yaml", "lightgbm", 7, 5),
    ],
)
def test_presets_load(name, backend, horizon, seeds):
    config = load_config(ROOT / "configs" / name)
    assert config.backend == backend
    assert config.raw["model"]["horizon"] == horizon
    assert len(config.section("training")["seeds"]) == seeds


def test_rejects_tail_count_and_fraction_together(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "lightgbm_h7.yaml").read_text())
    raw["portfolio"]["tail_fraction"] = 0.1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="exactly one"):
        load_config(path)
