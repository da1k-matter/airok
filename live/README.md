# RankTrend live

The Rust workspace owns the paper-trading runtime. It subscribes to confirmed Bybit linear 1D candles for the active model universe, repairs missing daily candles via public REST, recomputes the causal panel, ranks the frozen LightGBM ensemble, and simulates fills from one L2 order-book snapshot per target order.

It is paper-only: no private Bybit credentials, signing, or order-placement code exists here.

## Layout

```text
live/
├── app/                 Axum service binary
├── crates/              Focused domain libraries
├── config/paper.toml    Runtime settings
├── ui/                  Static HTML, CSS, and JavaScript dashboard
├── models/              Immutable bundles exported by train (versioned in Git)
└── state/paper/         SQLite ledger and WAL files (ignored)
```

## Run

From `live/`:

```bash
cargo run --release -p ranktrend-paper -- --config config/paper.toml
```

Open `http://127.0.0.1:8789`. Readiness is confirmed by `GET /health` and the detailed session state at `GET /api/session`.

The default runtime configuration reads the shared historical candle directory at `../train/data/1d` for bootstrap only. It writes no research artifacts there. Change `[data].directory` if the deployed service maintains its own market-history store.

### Performance history

The ledger keeps one latest mark-to-market equity point for every observed UTC minute without a retention limit. Maximum drawdown is updated from every incoming mark, including multiple symbol updates inside the same minute. The dashboard requests only enough extrema-preserving time buckets for the current canvas width, so long histories remain inexpensive to transfer and render while the complete minute series stays in SQLite.

Sharpe and average return use consecutive minute equity returns. Profit factor and win rate use net P&L from closed position quantities, including allocated entry and exit fees; they are not derived from daily account P&L.

### First paper-session bootstrap

For a brand-new paper session, the default launch automatically bootstraps exactly one paper decision from yesterday's confirmed UTC candle. It rebuilds missing causal OHLCV history when needed, sizes and records each entry from that candle's close, and applies only the current order book's executable impact from the best quote to the fill VWAP. It then waits for subsequent 1D closes through WebSocket. No paper trades are backfilled for missed days.

```bash
cargo run --release -p ranktrend-paper -- --config config/paper.toml
```

When a paper ledger already exists, the regular launch restores it and only waits for the next confirmed close. To create a fresh empty session without its initial one-day decision, launch with `--no-bootstrap` after removing the existing paper state:

```bash
cargo run --release -p ranktrend-paper -- --config config/paper.toml --no-bootstrap
```

## Historical OOS parity replay

This short, isolated mode verifies Rust against an existing Python backtest over one complete 56-day walk-forward block. It uses local OHLCV only: no Bybit connection, order book, slippage, or live-paper ledger state. The previous fold is used only for the 21-day smoothing warm-up.

First export the two required Python folds from `train/`:

```bash
python src/export_bundle.py --config configs/lightgbm_h7.yaml --data data/1d \
  --bundle ../live/models/parity_warmup_20260310 --prediction-date 2026-03-10
python src/export_bundle.py --config configs/lightgbm_h7.yaml --data data/1d \
  --bundle ../live/models/parity_block_20260505 --prediction-date 2026-05-05
```

Then run this from `live/` and open `http://127.0.0.1:8789`:

```bash
cargo run --release -p ranktrend-paper -- --config config/paper.toml \
  --replay-start 2026-05-05 --replay-end 2026-06-29 \
  --replay-bundle models/parity_block_20260505 \
  --warmup-bundle models/parity_warmup_20260310 \
  --reference ../train/outputs/runs/20260805_075042_lightgbm_h7_multiseed/daily.csv
```

The command exits before serving if its maximum daily return or turnover difference exceeds `1e-6`. On success, the dashboard is explicitly labelled `HISTORICAL REPLAY` and `state/paper/historical_replay.csv` contains the 56 daily values.

## Verification

```bash
cargo test --workspace
cargo build --release -p ranktrend-paper
```
