# RankTrend

RankTrend has two intentionally separate systems:

| Directory | Owner | Purpose |
| --- | --- | --- |
| [`train/`](train/) | Python | Feature research, walk-forward model training, backtests, reports, and model export. |
| [`live/`](live/) | Rust | Bybit market-data ingestion, model inference, paper execution, persistent ledger, API, and dashboard. |

There is no Python runtime in the live service. Python produces an immutable LightGBM bundle; Rust loads that bundle through `lightgbm3` and performs inference only.

## Layout

```text
ranktrend/
├── train/                   Python research and training project
│   ├── configs/             Research model specifications
│   ├── data/1d/             Local daily candle source data (ignored)
│   ├── src/                 Flat Python pipeline
│   ├── tests/               Python tests
│   └── outputs/             Backtest artifacts (ignored)
└── live/                    Rust paper-trading service
    ├── app/                 Service binary and dashboard assets
    ├── crates/              Domain libraries
    ├── config/              Runtime configuration
    ├── models/              Frozen model bundles (ignored)
    └── state/               SQLite paper ledger (ignored)
```

Each subsystem is started from its own directory. Nothing at repository root is an executable entry point.

## Typical workflow

Run research and backtests:

```bash
cd train
python src/cli.py run --config configs/lightgbm_h7.yaml --data-dir data/1d
```

Export the final Python-trained bundle for the Rust service:

```bash
cd train
python src/export_bundle.py \
  --config configs/lightgbm_h7.yaml \
  --data data/1d \
  --bundle ../live/models/lightgbm_h7_YYYYMMDD
```

Point `live/config/paper.toml` at that new bundle, then start paper trading:

```bash
cd live
cargo run --release -p ranktrend-paper -- --config config/paper.toml
```

The dashboard is available at `http://127.0.0.1:8789`. The service uses public Bybit endpoints only and does not have private credentials or order-placement code.

See [`train/README.md`](train/README.md) for the research contract and [`live/README.md`](live/README.md) for live-runtime behavior.
