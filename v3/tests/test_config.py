from pathlib import Path

import pytest
import yaml

from ranktrend.config import load_config
from ranktrend.tune import _anchor_params, _suggest_trial_config, _top_pareto_trials

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "horizons", "seeds"),
    [
        ("lightgbm_h7.yaml", [7], 3),
        ("lightgbm_h7_h10.yaml", [7, 10], 3),
        ("tune_nsga2.yaml", [7, 10], 3),
    ],
)
def test_presets_load(name, horizons, seeds):
    config = load_config(ROOT / "configs" / name)
    assert [item["horizon"] for item in config.raw["models"]] == horizons
    assert len(config.section("training")["seeds"]) == seeds
    assert isinstance(config.section("lightgbm")["params"], dict)


def test_rejects_tail_count_and_fraction_together(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "lightgbm_h7.yaml").read_text())
    raw["portfolio"]["tail_fraction"] = 0.1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="exactly one"):
        load_config(path)


def test_optuna_anchors_build_v1_and_v2_like_configs():
    optuna = pytest.importorskip("optuna")
    config = load_config(ROOT / "configs" / "tune_nsga2.yaml")
    v1_like, v2_like = [_suggest_trial_config(config, optuna.trial.FixedTrial(params)) for params in _anchor_params()]

    assert v1_like.raw["models"] == [{"horizon": 7, "weight": 0.5}, {"horizon": 10, "weight": 0.5}]
    assert v1_like.section("lightgbm")["params"]["subsample_freq"] == 0
    assert v1_like.section("portfolio")["tail_count"] == 5
    assert v1_like.section("ensemble")["method"] == "mean_rank"

    assert v2_like.raw["models"] == [{"horizon": 7, "weight": 1.0}]
    assert v2_like.section("lightgbm")["params"]["subsample_freq"] == 1
    assert v2_like.section("portfolio")["tail_count"] == 6
    assert v2_like.section("ensemble")["method"] == "median_rank"


def test_pareto_test_selection_is_limited_and_deterministic():
    class Trial:
        def __init__(self, number, values):
            self.number = number
            self.values = values

    class Study:
        best_trials = [Trial(number, (float(number), float(30 - number))) for number in range(30)]

    selected = _top_pareto_trials(Study(), limit=25)
    assert len(selected) == 25
    assert len({trial.number for trial in selected}) == 25
