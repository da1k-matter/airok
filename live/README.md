# airōk live

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
cargo run --release -p airok-paper -- --config config/paper.toml
```

Open `http://127.0.0.1:18789`. Readiness is confirmed by `GET /health` and the detailed session state at `GET /api/session`. In production, Nginx proxies public port `8790` to this loopback address.

The default runtime configuration reads the shared historical candle directory at `../train/data/1d` for bootstrap only. It writes no research artifacts there. Change `[data].directory` if the deployed service maintains its own market-history store.

### Strategy

The default bundle `models/lightgbm_h7_champion_20261013` is the research champion
`v2_base_neutral` (`train/configs/lightgbm_h7_champion.yaml`): the same LightGBM h7 recipe with
10 seeds, plus portfolio neutralization configured in `[strategy]`:

- `alt_beta_neutral = 0.5` blends the dollar/BTC-beta neutral book 50/50 with a book that is
  also neutral to the 60-day beta against the equal-weighted eligible-universe return;
- `size_neutral = true` additionally neutralizes the cross-sectional z-score of 20-day log
  dollar volume in that second book.

Seed ranks are aggregated with float32 rounding exactly as numpy does, so that equal ranks tie
and are ordered by column like numpy's stable sort.

### Funding

Open positions pay and receive funding as on Bybit. For every settlement of a held contract
the service reads the settled rate from `/v5/market/funding/history` and the mark price at the
settlement time from the one-minute `/v5/market/mark-price-kline`, then books
`-quantity × mark price × rate` into cash: longs pay a positive rate and shorts receive it.
The quantity is the one held at the settlement timestamp (a fill at exactly that time does not
count), taken from a persisted quantity history, so a late or retried check charges the same
amount. Settlements are polled every minute and once more right before each daily rebalance.
Every charge is stored in the `funding_events` table and served at `GET /api/funding`; the
dashboard shows cumulative funding and funding per open position. Ledgers created before this
feature start accruing funding for their open positions at the next launch.

### Copy Trading execution universe

Set `bybit.copy_trading_only = true` to query Bybit's public instrument API at
each daily refresh and trade only contracts the API reports with all of:
`status=Trading`, `contractType=LinearPerpetual`, `quoteCoin=USDT`,
`settleCoin=USDT`, and `copyTrading=both`. The full immutable model universe
is still subscribed and used for scoring; filtering it would violate a bundle's
feature contract. As in the research track B, the 75-name liquidity universe and the long and
short tails are then formed among the allowed contracts only, so the book stays neutral.
Existing positions remain reducible even if a contract later loses Copy Trading availability.
Keep a bundle trained on the full universe: research showed that training only on Copy Trading
contracts is worse when trading them.

### Performance history

The ledger keeps one latest mark-to-market equity point for every observed UTC minute without a retention limit. Maximum drawdown is updated from every incoming mark, including multiple symbol updates inside the same minute. The dashboard requests only enough extrema-preserving time buckets for the current canvas width, so long histories remain inexpensive to transfer and render while the complete minute series stays in SQLite.

Sharpe uses consecutive minute equity returns. Average daily return, profit factor, and win rate use one canonical series of completed daily portfolio-period returns from persisted decision snapshots. Zero-return periods are excluded from win-rate wins/losses; execution and closed-position P&L remain available for bookkeeping diagnostics but do not drive these headline metrics.

### First paper-session bootstrap

For a brand-new paper session, the default launch automatically bootstraps exactly one paper decision from yesterday's confirmed UTC candle. It rebuilds missing causal OHLCV history when needed, sizes and records each entry from that candle's close, and applies only the current order book's executable impact from the best quote to the fill VWAP. It then waits for subsequent 1D closes through WebSocket. No paper trades are backfilled for missed days.

```bash
cargo run --release -p airok-paper -- --config config/paper.toml
```

When a paper ledger already exists, the regular launch restores it and only waits for the next confirmed close. To create a fresh empty session without its initial one-day decision, launch with `--no-bootstrap` after removing the existing paper state:

```bash
cargo run --release -p airok-paper -- --config config/paper.toml --no-bootstrap
```

### Frozen ledger replay

Use `--ledger-replay` with a copied SQLite ledger to inspect a past paper session in the dashboard without making any network requests or changing the copied ledger. This preserves the recorded execution reports, order-book snapshots, positions, and equity curve exactly as captured.

```bash
cargo run --release -p airok-paper -- --config /path/to/replay.toml --ledger-replay
```

## Historical OOS parity replay

This short, isolated mode verifies Rust against an existing Python backtest over one complete 56-day walk-forward block. It uses local OHLCV only: no Bybit connection, order book, slippage, or live-paper ledger state. The previous fold is used only for the 21-day smoothing warm-up.

First export the two required Python folds and a funding-free Python reference from `train/`.
The replay uses the `[strategy]` rules of the given config, so the reference must come from the
same recipe (here the champion):

```bash
python src/export_bundle.py --config configs/lightgbm_h7_champion.yaml \
  --bundle research/parity_bundles/parity_champion_warmup_20260505 --prediction-date 2026-05-05
python src/export_bundle.py --config configs/lightgbm_h7_champion.yaml \
  --bundle research/parity_bundles/parity_champion_block_20260630 --prediction-date 2026-06-30
python src/cli.py run --config configs/lightgbm_h7_champion.yaml --no-funding
```

The replay compares target-weight returns, so convert the run's `ideal_net_return` and
`ideal_turnover` columns into `date,net_return,turnover` (`ideal_reference.csv`). Then run this
from `live/` and open `http://127.0.0.1:18789`:

```bash
cargo run --release -p airok-paper -- --config config/paper.toml \
  --replay-start 2026-06-30 --replay-end 2026-08-24 \
  --replay-bundle ../train/research/parity_bundles/parity_champion_block_20260630 \
  --warmup-bundle ../train/research/parity_bundles/parity_champion_warmup_20260505 \
  --reference ../train/outputs/runs/<run>/ideal_reference.csv
```

On 2026-10-16 this block matched Python within 3.6e-9 (daily return) and 3.8e-8 (turnover).

The command exits before serving if its maximum daily return or turnover difference exceeds `1e-6`. On success, the dashboard is explicitly labelled `HISTORICAL REPLAY` and `state/paper/historical_replay.csv` contains the 56 daily values.

## Verification

```bash
cargo test --workspace
cargo build --release -p airok-paper
```
