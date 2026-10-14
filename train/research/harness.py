"""Research harness: evaluate one configuration variant against the champion.

    python research/harness.py <name> [dotted.key=yaml_value ...] [--base CONFIG] [--holdout]

The variant config is written to research/configs/<name>.yaml, so every result
is reproducible with `python src/cli.py run --config research/configs/<name>.yaml`.
Daily returns and metrics go to research/results/<name>.{csv,json}. Bybit
instrument rules come from a frozen snapshot for deterministic execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bybit import BybitInstrumentRule  # noqa: E402
from config import load_config  # noqa: E402
from funding import load_funding_factors  # noqa: E402
from pipeline import _bybit_rule_arrays, build_ensemble_weights, get_context, train_models  # noqa: E402
from portfolio import simulate_bybit_market_pnl, simulate_target_weight_pnl, target_weight_funding  # noqa: E402

RESEARCH = ROOT / "research"
RULES = ROOT / ".cache" / "bybit_rules_20261014.json"
CHAMPION_FILE = RESEARCH / "champion.json"
DEV = ("2023-08-05", "2026-06-12")
HOLDOUT = ("2026-06-13", "2026-10-12")
SEGMENTS = {
    "23H2": ("2023-08-05", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "26H1": ("2026-01-01", "2026-06-12"),
}
LOG = logging.getLogger("harness")


def set_dotted(raw: dict[str, Any], key: str, value: Any) -> None:
    node = raw
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def stats(r: pd.Series, turnover: pd.Series | None = None, funding: pd.Series | None = None) -> dict[str, float]:
    r = r.dropna()
    equity = (1.0 + r).cumprod()
    years = len(r) / 365.0
    std = r.std(ddof=1)
    downside = np.minimum(r, 0.0).std(ddof=1)
    monthly = (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0
    max_dd = float(-(equity / equity.cummax() - 1.0).min())
    out = {
        "sharpe": float(r.mean() / std * math.sqrt(365)) if std > 0 else 0.0,
        "sortino": float(r.mean() * 365 / (downside * math.sqrt(365))) if downside > 0 else 0.0,
        "cagr": float(cagr),
        "vol": float(std * math.sqrt(365)),
        "max_dd": max_dd,
        "calmar": float(cagr / max_dd) if max_dd > 0 else 0.0,
        "pos_months": float((monthly > 0).mean()),
        "pos_days": float((r > 0).sum() / max((r != 0).sum(), 1)),
        "worst_month": float(monthly.min()),
    }
    if turnover is not None:
        out["turnover_pa"] = float(turnover.loc[r.index].sum() / years)
    if funding is not None:
        out["funding_pa"] = float(funding.loc[r.index].mean() * 365)
    return out


def load_rules() -> dict[str, BybitInstrumentRule]:
    if not RULES.exists():
        raise SystemExit(f"Frozen rule snapshot missing: {RULES}")
    snapshot = json.loads(RULES.read_text())
    return {item["symbol"]: BybitInstrumentRule(**item) for item in snapshot["rules"]}


def simulate(config, ctx, weights, end: str) -> pd.DataFrame:
    dates: pd.DatetimeIndex = ctx["dates"]
    start_idx = int(np.searchsorted(dates.values, np.datetime64(DEV[0])))
    end_idx = min(int(np.searchsorted(dates.values, np.datetime64(end), side="right")) - 1, len(dates) - 2)
    weights = weights.copy()
    weights[:start_idx] = 0.0
    weights[end_idx + 1 :] = 0.0
    cost = float(config.section("costs")["one_way_bps"]) / 10_000.0
    funding_dir = config.resolve((config.section("costs").get("funding") or {}).get("directory", "data/funding"))
    factor = load_funding_factors(funding_dir, dates, list(ctx["symbols"]), np.asarray(ctx["open"]))["factor"]

    ideal, ideal_turnover = simulate_target_weight_pnl(weights, np.asarray(ctx["oo_ret"]), cost)
    ideal_funding = target_weight_funding(weights, factor)
    tradable, qty_step, min_qty, min_notional, max_qty = _bybit_rule_arrays(list(ctx["symbols"]), load_rules())
    targets = weights.copy()
    targets[:, ~tradable] = 0.0
    execution = config.section("execution")
    executed = simulate_bybit_market_pnl(
        targets,
        np.asarray(ctx["open"]),
        np.asarray(ctx["oo_ret"]),
        cost,
        initial_equity_usd=float(execution["initial_equity_usd"]),
        gross_leverage=float(execution["gross_leverage"]),
        qty_step=qty_step,
        min_order_qty=min_qty,
        min_notional_value=min_notional,
        max_market_order_qty=max_qty,
        funding_factor=factor,
    )
    frame = pd.DataFrame(
        {
            "ideal": ideal + ideal_funding,
            "ideal_no_funding": ideal,
            "ideal_funding": ideal_funding,
            "ideal_turnover": ideal_turnover,
            "exec": executed.pnl,
            "exec_funding": executed.funding_pnl,
            "gross": np.abs(weights).sum(axis=1),
        },
        index=dates,
    )
    return frame.loc[DEV[0] : end]


def paired_t(a: pd.Series, b: pd.Series) -> float:
    diff = (a - b).dropna()
    return float(diff.mean() / diff.std(ddof=1) * math.sqrt(len(diff))) if diff.std(ddof=1) > 0 else 0.0


def report(daily: pd.DataFrame, holdout: bool) -> dict[str, Any]:
    periods = {"DEV": DEV, **SEGMENTS}
    if holdout:
        periods["HOLDOUT"] = HOLDOUT
    out: dict[str, Any] = {}
    for label, (start, end) in periods.items():
        part = daily.loc[start:end]
        out[label] = stats(part["ideal"], part["ideal_turnover"], part["ideal_funding"])
        out[label]["no_funding_sharpe"] = stats(part["ideal_no_funding"])["sharpe"]
        out[label]["exec_sharpe"] = stats(part["exec"])["sharpe"]
        out[label]["exec_cagr"] = stats(part["exec"])["cagr"]
    return out


def print_report(name: str, result: dict[str, Any], champion: dict[str, Any] | None, t_stat: float | None) -> None:
    head = f"{'period':8s} {'Sharpe':>7s} {'dSh':>6s} {'Sortino':>7s} {'CAGR':>7s} {'MaxDD':>6s} {'Calmar':>6s} {'+mon':>5s} {'+day':>5s} {'worstM':>7s} {'fund/y':>7s} {'TO/y':>5s} {'noFund':>6s} {'exec':>5s}"
    print(f"\n== {name}" + (f"   paired t vs champion (DEV): {t_stat:+.2f}" if t_stat is not None else ""))
    print(head)
    for label, m in result.items():
        delta = m["sharpe"] - champion[label]["sharpe"] if champion and label in champion else float("nan")
        print(
            f"{label:8s} {m['sharpe']:7.2f} {delta:+6.2f} {m['sortino']:7.2f} {m['cagr']:7.1%} {m['max_dd']:6.1%} "
            f"{m['calmar']:6.2f} {m['pos_months']:5.0%} {m['pos_days']:5.0%} {m['worst_month']:7.1%} "
            f"{m['funding_pa']:+7.1%} {m['turnover_pa']:5.1f} {m['no_funding_sharpe']:6.2f} {m['exec_sharpe']:5.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("overrides", nargs="*", help="dotted.key=yaml_value")
    parser.add_argument("--base", type=Path, default=None, help="Base config; default is the champion config")
    parser.add_argument("--holdout", action="store_true", help="Also report the untouched HOLDOUT period")
    parser.add_argument("--promote", action="store_true", help="Make this variant the champion")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    champion = json.loads(CHAMPION_FILE.read_text()) if CHAMPION_FILE.exists() else None
    base = args.base or (ROOT / champion["config"] if champion else ROOT / "configs/lightgbm_h7_extended.yaml")
    raw = deepcopy(yaml.safe_load(Path(base).read_text()))
    raw["name"] = args.name
    raw.setdefault("costs", {}).setdefault("funding", {}).update({"enabled": True})
    for item in args.overrides:
        key, _, value = item.partition("=")
        set_dotted(raw, key, yaml.safe_load(value))
    config_path = RESEARCH / "configs" / f"{args.name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    config = load_config(config_path)
    ctx = get_context(config)
    predictions = train_models(config, ctx)
    weights, _ = build_ensemble_weights(config, ctx, predictions)
    daily = simulate(config, ctx, weights, HOLDOUT[1] if args.holdout else DEV[1])

    results = RESEARCH / "results"
    results.mkdir(parents=True, exist_ok=True)
    daily.to_csv(results / f"{args.name}.csv")
    result = report(daily, args.holdout)
    t_stat = None
    champion_metrics = None
    if champion and champion["name"] != args.name:
        champion_daily = pd.read_csv(results / f"{champion['name']}.csv", index_col=0, parse_dates=True)
        dev = slice(*DEV)
        t_stat = paired_t(daily.loc[dev, "ideal"], champion_daily.loc[dev, "ideal"])
        champion_metrics = json.loads((results / f"{champion['name']}.json").read_text())["metrics"]
    (results / f"{args.name}.json").write_text(
        json.dumps(
            {"name": args.name, "config": str(config_path.relative_to(ROOT)), "overrides": args.overrides,
             "paired_t_vs_champion": t_stat, "champion": champion["name"] if champion else None,
             "metrics": result},
            indent=2,
        )
    )
    print_report(args.name, result, champion_metrics, t_stat)
    if args.promote or champion is None:
        CHAMPION_FILE.write_text(json.dumps({"name": args.name, "config": str(config_path.relative_to(ROOT))}, indent=2))
        print(f"\nchampion -> {args.name}")


if __name__ == "__main__":
    main()
