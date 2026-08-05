from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .pipeline import backtest, doctor, get_context, reproduce, train_models, verify_reference


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ranktrend", description="Bybit cross-sectional ranking research")
    root.add_argument("--verbose", action="store_true")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("doctor", "build-features", "train", "backtest", "reproduce"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/best_v2.yaml", type=Path)
        if name in {"build-features", "train", "backtest", "reproduce"}:
            command.add_argument("--reuse-features", action="store_true")
        if name in {"train", "backtest", "reproduce"}:
            command.add_argument("--reuse-predictions", action="store_true")
    return root


def print_summary(summary: dict) -> None:
    for label in ("full", "latest_year"):
        row = summary[label]
        print(f"\n{label}")
        print(f"  period:          {row['start']} .. {row['end']}")
        print(f"  CAGR:            {row['cagr']:.2%}")
        print(f"  Sharpe:          {row['sharpe']:.3f}")
        print(f"  max drawdown:    {row['max_dd']:.2%}")
        print(f"  positive months: {row['positive_months']}/{row['n_months']}")
        print(f"  turnover/year:   {row['turnover_pa']:.2f}x")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    configure_logging(args.verbose)
    config = load_config(args.config)
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(config), indent=2))
            return 0
        if args.command == "build-features":
            ctx = get_context(config, reuse=args.reuse_features)
            print(f"features ready: {ctx['feature_stack'].shape}; cache={config.cache_dir}")
            return 0
        if args.command == "train":
            ctx = get_context(config, reuse=args.reuse_features)
            train_models(config, ctx, reuse=args.reuse_predictions)
            print(f"predictions ready: {config.cache_dir / 'predictions'}")
            return 0
        if args.command == "backtest":
            ctx = get_context(config, reuse=args.reuse_features)
            scores = train_models(config, ctx, reuse=args.reuse_predictions)
            output, summary = backtest(config, ctx, scores)
            print_summary(summary)
            print(f"\noutputs: {output}")
            return 0
        if args.command == "reproduce":
            output, summary, ok, failures = reproduce(
                config,
                reuse_features=args.reuse_features,
                reuse_predictions=args.reuse_predictions,
            )
            print_summary(summary)
            print(f"\noutputs: {output}")
            if ok:
                print("reference check: PASS")
                return 0
            print("reference check: FAIL")
            for failure in failures:
                print(f"  - {failure}")
            return 2
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
