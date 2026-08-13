# RankTrend trainer

The Python trainer owns the complete offline pipeline: data validation, causal feature construction, walk-forward LightGBM/CatBoost training, portfolio construction, backtesting, reporting, and immutable LightGBM-bundle export.

## Commands

Run these from `trainer/` with the Python environment already activated:

```bash
python src/cli.py doctor --config configs/lightgbm_h7.yaml --data-dir data/1d
python src/cli.py build-features --config configs/lightgbm_h7.yaml --data-dir data/1d
python src/cli.py train --config configs/lightgbm_h7.yaml --data-dir data/1d
python src/cli.py backtest --config configs/lightgbm_h7.yaml --data-dir data/1d
python src/cli.py run --config configs/lightgbm_h7.yaml --data-dir data/1d
python -m pytest -q
```

`configs/lightgbm_h7.yaml` is the production candidate: a five-seed LightGBM LambdaRank ensemble with median-rank aggregation. `configs/catboost_h7.yaml` is a research challenger.

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

Pass `--prediction-date YYYY-MM-DD` to export one particular scheduled 56-day walk-forward fold for the Rust historical parity replay. The date must match the configured `retrain_every_days` schedule.
