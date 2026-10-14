# airōk training

The Python training project owns the complete offline pipeline: data validation, causal feature construction, walk-forward LightGBM training, portfolio construction, backtesting, reporting, and immutable LightGBM-bundle export.

## Commands

Run these from `train/` with the Python environment already activated:

```bash
python src/cli.py doctor --config configs/lightgbm_h7.yaml --data-dir data/1d
python src/cli.py build-features --config configs/lightgbm_h7.yaml --data-dir data/1d
python src/cli.py train --config configs/lightgbm_h7.yaml --data-dir data/1d
python src/cli.py backtest --config configs/lightgbm_h7.yaml --data-dir data/1d
python src/cli.py run --config configs/lightgbm_h7.yaml --data-dir data/1d
python -m pytest -q
```

Refresh the local panel before a new production export. The command requests only
confirmed days, ending at yesterday UTC by default:

```bash
python src/update_data.py --data-dir data/1d
```

To rebuild the panel from scratch (and optionally funding history), use the
Bybit scraper. It keeps only confirmed bars through `--through` (default
yesterday UTC), fetches every symbol's request windows concurrently under one
adaptive rate limiter, and shows progress bars per phase:

```bash
python src/scraper_bybit.py --data-dir data/1d --days 1000 --symbols-from data/1d --funding
```

Without `--symbols`/`--symbols-from` it downloads every trading USDT perpetual.
`--funding` also writes `data/funding/<SYMBOL>_funding.csv` (`timestamp`,
`funding_rate`; one row per settlement, positive means longs pay shorts);
`--funding-only` skips candles. Each output directory receives a
`_manifest_*.json` with per-symbol coverage.

### Funding in backtests

With `costs.funding.enabled: true` the backtest charges Bybit funding exactly
as the exchange settles it: at each settlement the open position pays
`qty × price × funding_rate` (positive rates are paid by longs and received by
shorts). A position executed at open t is held through open t+1, so it pays
every settlement in (00:00 t, 00:00 t+1]; the 00:00 settlement on a rebalance
day is charged to the position held before the rebalance. The intraday mark
price is interpolated between the two opens. `--funding` / `--no-funding`
overrides the YAML for `run` and `backtest`.

When funding is charged, the headline metrics include it; `summary.json`
additionally holds `without_funding` metrics from the same predictions and a
`funding` attribution block (annualized total, long and short sides, coverage).
`daily.csv` gains `funding_return` and `net_return_without_funding`.

`configs/lightgbm_h7_1000d.yaml` is the production recipe shifted to start
predictions in 2025 so it runs on a 1000-day panel.

`configs/lightgbm_h7.yaml` is the production candidate: a five-seed LightGBM LambdaRank ensemble with median-rank aggregation.

`configs/lightgbm_h7_copytrading.yaml` runs the same recipe on the current
Bybit Copy Trading universe. At process start it requests Bybit linear
instrument metadata and retains only `Trading` USDT `LinearPerpetual`
contracts whose `copyTrading` value is `both`. The exact data-file universe is
saved as `training_universe.json` in the run output alongside the complete API
rule snapshot. Run it with the same command, substituting the copy-trading
config path.

## Local artifacts

`data/1d/` contains input `<SYMBOL>_1440.csv` files. Feature and prediction caches are stored in `.cache/`; run outputs are stored in `outputs/`. These directories are local and ignored by Git.

## Export to live

Only a LightGBM `median_rank` ensemble can be exported. The exporter writes model text files plus a manifest, feature contract, hashes, and the symbol universe; it refuses to overwrite an existing bundle.

```bash
python src/export_bundle.py \
  --config configs/lightgbm_h7.yaml \
  --data data/1d \
  --bundle ../live/models/lightgbm_h7_YYYYMMDD
```

After export, change `../live/config/paper.toml` to reference the exact bundle directory. Rust inference must use the identical feature contract and frozen bundle.

By default the exporter trains the newest causal fold available in the dataset.
Pass `--prediction-date YYYY-MM-DD` to export a particular earlier causal fold for
the Rust historical parity replay.
