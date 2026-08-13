# RankTrend live

The Rust workspace owns the paper-trading runtime. It subscribes to confirmed Bybit linear 1D candles for the active model universe, repairs missing daily candles via public REST, recomputes the causal panel, ranks the frozen LightGBM ensemble, and simulates fills from one L2 order-book snapshot per target order.

It is paper-only: no private Bybit credentials, signing, or order-placement code exists here.

## Layout

```text
live/
├── app/                 Axum service binary and Rust/WASM dashboard
├── crates/              Focused domain libraries
├── config/paper.toml    Runtime settings
├── models/              Immutable bundles exported by trainer (ignored)
└── state/paper/         SQLite ledger and WAL files (ignored)
```

## Run

From `live/`:

```bash
cargo run --release -p ranktrend-paper -- --config config/paper.toml
```

Open `http://127.0.0.1:8789`. Readiness is confirmed by `GET /health` and the detailed session state at `GET /api/session`.

The default runtime configuration reads the shared historical candle directory at `../trainer/data/1d` for bootstrap only. It writes no research artifacts there. Change `[data].directory` if the deployed service maintains its own market-history store.

## Verification

```bash
cargo test --workspace
cargo build --release -p ranktrend-paper
```
