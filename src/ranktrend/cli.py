from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .pipeline import backtest, doctor, get_context, run, train_models


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="ranktrend",
        description="Cross-sectional crypto ranking with LightGBM or CatBoost",
    )
    root.add_argument("--verbose", action="store_true")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("doctor", "build-features", "train", "backtest", "run"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/lightgbm_h7.yaml", type=Path)
        command.add_argument("--data-dir", type=Path, help="Override data.directory from YAML")
        if name in {"build-features", "train", "backtest", "run"}:
            command.add_argument("--fresh-features", action="store_true")
        if name in {"train", "backtest", "run"}:
            command.add_argument("--fresh-predictions", action="store_true")
    return root


def print_summary(summary: dict) -> None:
    model = summary["model"]
    print(
        f"\n{model['backend']} horizon={model['horizon']} "
        f"seeds={model['seeds']} aggregation={model['seed_aggregation']}"
    )
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
    config = load_config(args.config, data_dir_override=args.data_dir)
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(config), indent=2))
            return 0

        reuse_features = not args.fresh_features
        if args.command == "build-features":
            ctx = get_context(config, reuse=reuse_features)
            print(f"features ready: {ctx['feature_stack'].shape}; cache={config.cache_dir}")
            return 0

        reuse_predictions = not args.fresh_predictions
        if args.command == "run":
            output, summary = run(
                config,
                reuse_features=reuse_features,
                reuse_predictions=reuse_predictions,
            )
        else:
            ctx = get_context(config, reuse=reuse_features)
            predictions = train_models(config, ctx, reuse=reuse_predictions)
            if args.command == "train":
                count = len(predictions)
                print(f"prediction matrices ready: {count}; cache={config.cache_dir / 'predictions'}")
                return 0
            output, summary = backtest(config, ctx, predictions)
        print_summary(summary)
        print(f"\noutputs: {output}")
        return 0
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
