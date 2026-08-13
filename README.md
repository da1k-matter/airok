# RankTrend GBM

Compact, production-shaped research project for cross-sectional Bybit perpetual ranking with **LightGBM LambdaRank** and **CatBoost YetiRank NDCG**.

The package trains one forward-ranking model per run. The current presets use a 7-day horizon and support deterministic single-seed runs or multi-seed ensembles.

## Project shape

Only seven substantive modules:

```text
src/ranktrend/
├── config.py       # YAML validation and path resolution
├── data.py         # CSV validation, dense float32 panel and mmap cache
├── research.py     # features, residual labels, LGBM/CatBoost walk-forward training
├── portfolio.py    # tails, weighting, neutrality, smoothing, overlay and Numba PnL
├── reporting.py    # metrics, calendar tables and plots
├── pipeline.py     # cache keys, ensemble orchestration and run outputs
└── cli.py          # one CLI
```

No XGBoost dependency, no notebooks, no research-script dump, no hidden global state.

## Presets

| Config | Backend | Horizon | Seeds | Seed aggregation |
|---|---|---:|---:|---|
| `configs/lightgbm_h7.yaml` | LightGBM LambdaRank | 7 | 5 | median rank |
| `configs/catboost_h7.yaml` | CatBoost YetiRankPairwise NDCG | 7 | 5 | mean rank |

## Strategy

1. Build a point-in-time universe from trailing 20-day log dollar volume.
2. Require at least 180 days of history.
3. Predict deciles of future open-to-open return after removing ex-ante BTC beta.
4. Rank every eligible contract cross-sectionally each day.
5. Select six long and six short contracts from the top-75 liquid universe.
6. Use equal or inverse-volatility side weights.
7. Project weights to dollar neutrality and near-zero BTC beta.
8. Average target positions over a causal 21-day window.
9. Optionally scale gross exposure with the lagged BTC-volatility regime.
10. Execute at the next daily open and subtract one-way turnover costs.

All input arrays and feature caches use dense `float32`; PnL and turnover simulation are Numba-jitted.

## Install

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
```

## Data

Place daily CSV files in `data/1d/`, or pass another location with `--data-dir`.

File naming:

```text
BTC_1440.csv
ETH_1440.csv
SOL_1440.csv
...
```

Required schema:

```text
timestamp,open,high,low,close,volume
```

Every timestamp must be unique and increasing. `BTC_1440.csv` is mandatory.

## Quick start

```bash
ranktrend doctor \
  --config configs/lightgbm_h7.yaml \
  --data-dir /path/to/bybit/1d

ranktrend run \
  --config configs/lightgbm_h7.yaml \
  --data-dir /path/to/bybit/1d
```

CatBoost:

```bash
ranktrend run \
  --config configs/catboost_h7.yaml \
  --data-dir /path/to/bybit/1d
```

Or:

```bash
RANKTREND_DATA=/path/to/bybit/1d ./run.sh configs/catboost_h7.yaml
```

## Pipeline stages

```bash
# Validate environment and data
ranktrend doctor --config configs/lightgbm_h7.yaml --data-dir /path/to/1d

# Build or load features
ranktrend build-features --config configs/lightgbm_h7.yaml --data-dir /path/to/1d

# Train one horizon across all configured seeds
ranktrend train --config configs/lightgbm_h7.yaml --data-dir /path/to/1d

# Backtest from cached predictions
ranktrend backtest --config configs/lightgbm_h7.yaml --data-dir /path/to/1d
```

Caches are reused by default. Force rebuilds with:

```bash
ranktrend run ... --fresh-features --fresh-predictions
```

Feature and prediction caches are content-keyed. Changing backend, horizon, seed, training parameters or data invalidates only the affected cache.

## Multi-seed semantics

Each seed produces one ranked signal for the configured horizon. Seeds are then combined using one configured method:

- `mean_rank` — average seed-level rank signals;
- `median_rank` — robust median of seed-level rank signals;
- `portfolio_average` — build one portfolio per seed, then average weights.

Set `training.seeds: [42]` for a single model.

## Output

Every run creates an immutable directory under `outputs/runs/`:

```text
YYYYMMDD_HHMMSS_<config-name>/
├── config_resolved.yaml
├── run_metadata.json
├── environment.txt
├── summary.json
├── daily.csv
├── monthly.csv
├── yearly.csv
├── fee_stress.csv
├── latest_weights.csv
├── equity.png
└── drawdown.png
```

`latest_weights.csv` is the current target portfolio for the final executable row in the dataset.

## Configuration

The most commonly changed fields are intentionally exposed in YAML:

```yaml
backend:
  name: lightgbm          # or catboost
  params: {...}

model:
  horizon: 7

training:
  retrain_every_days: 56
  train_window_days: 730
  seeds: [11, 23, 37, 53, 71]

ensemble:
  method: median_rank

portfolio:
  universe_size: 75
  tail_count: 6           # alternatively use tail_fraction
  weighting: invvol
  smoothing_days: 21
```

Set exactly one of `tail_count` or `tail_fraction`.

## Research limitations

The historical research archive has known limitations:

- likely survivorship bias because historically delisted perpetuals are absent;
- no funding cash flows;
- no mark-price or liquidation path;
- no order-book impact model;
- no historical point-in-time order-book archive.

Backtest Sharpe is therefore not a live-performance guarantee. Before deployment, add point-in-time delisted contracts, funding and execution simulation.

## Rust paper-trading simulator

The live simulator is a separate Rust workspace. It subscribes to the confirmed public Bybit linear 1D candle of every currently tradable contract in the immutable model universe, waits for the configured close-coverage threshold, and uses the public REST API only to repair missing contracts. It then recomputes the full causal feature panel, which dynamically selects the same liquidity-ranked universe as research, scores the frozen ensemble, and simulates fills from one public order-book snapshot per target order. It has no private API credentials and cannot place exchange orders.

Train and export the immutable LightGBM bundle in the Python research environment:

```bash
python scripts/export_lightgbm_bundle.py \
  --config configs/lightgbm_h7.yaml \
  --data data/1d \
  --bundle models/lightgbm_h7_20260804_python
```

The exporter refuses to overwrite a bundle. The resulting manifest pins the model hashes, feature contract, universe, cutoff date, and seed aggregation. Python owns training; the Rust service uses `lightgbm3` only for inference and needs neither Python nor a runtime LightGBM dylib.

Start the paper service:

```bash
cargo run --release -p ranktrend-paper -- --config configs/paper.toml
```

Open `http://127.0.0.1:8789`. The dashboard is Rust/WASM and shows the persisted $1,000 paper account, current positions, execution tape, and equity curve. The SQLite ledger location, Bybit public-feed settings, execution rules, and risk limits are all explicit in `configs/paper.toml`.
