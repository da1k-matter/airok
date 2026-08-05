# RankTrend

Clean, reproducible implementation of the selected cross-sectional Bybit perpetual ranking strategy.

The project intentionally keeps the implementation compact: seven cohesive modules, not dozens of tiny files:

```text
src/ranktrend/
├── config.py       # YAML loading and path resolution
├── data.py         # validation, dense panel and cache I/O
├── research.py     # features, residual labels and LightGBM walk-forward training
├── portfolio.py    # ranking portfolio, neutralization, smoothing, overlay and PnL
├── reporting.py    # metrics, calendar tables and plots
├── pipeline.py     # cache/run orchestration
└── cli.py          # one command-line interface
```

## Strategy

- LightGBM LambdaRank models at 7-day and 10-day horizons.
- Targets are deciles of future BTC-beta-residual open-to-open returns.
- Scores are converted to daily cross-sectional percentile ranks and blended 50/50.
- Dynamic top-75 universe by trailing 20-day log dollar volume.
- Minimum history of 180 days.
- Long top 7.5%, short bottom 7.5%.
- Inverse-20-day-volatility weighting.
- Exact dollar and ex-ante BTC-beta projection for each daily signal.
- Daily signals averaged over a fixed causal 21-day window.
- Gross exposure is 1.10 in the low-BTC-volatility regime and 0.75 otherwise.
- The regime compares the panel row's BTC 20-day volatility with its one-day-lagged trailing 180-day median. This preserves the exact selected legacy implementation; signal weights themselves are already shifted to next-open execution.
- Execution is at the next daily open.
- Backtest cost is 10 bps per unit of one-way turnover.

## Important limitation

All source files end on the same date. The dataset therefore appears to contain surviving/current contracts and likely omits historically delisted contracts. That creates survivorship bias. Funding, mark-price liquidation paths, order-book impact and borrow constraints are also unavailable. The package reproduces the research result; it does not turn that result into a guaranteed live expectation.

## Quick start with the existing Conda environment

From the project directory:

```bash
conda run -n NN python -m pip install --no-build-isolation -e .
conda run -n NN python -m ranktrend doctor --config configs/best_v2.yaml
conda run -n NN python -m ranktrend reproduce --config configs/best_v2.yaml
```

Or run all three steps:

```bash
./run.sh
```

The first run parses the 686 CSV files, builds the feature cache, and trains two walk-forward models. Later runs are much faster:

```bash
conda run -n NN python -m ranktrend reproduce \
  --config configs/best_v2.yaml \
  --reuse-features \
  --reuse-predictions
```

## Separate pipeline stages

```bash
# Validate environment and input data
conda run -n NN python -m ranktrend doctor --config configs/best_v2.yaml

# Rebuild features
conda run -n NN python -m ranktrend build-features --config configs/best_v2.yaml

# Train both models; reuse an existing feature cache
conda run -n NN python -m ranktrend train \
  --config configs/best_v2.yaml --reuse-features

# Backtest from cached features and predictions
conda run -n NN python -m ranktrend backtest \
  --config configs/best_v2.yaml --reuse-features --reuse-predictions
```

To force a completely fresh run, omit both reuse flags or delete `.cache/`.

## Reference targets

The package checks a full retraining run against `reference/expected_best_v2.json`.
The exact headline targets are retained. Full retraining can drift slightly across LightGBM versions, compilers, CPU architectures and sorting/runtime details; the verifier allows a narrow cross-platform tolerance while requiring the same profitable-month counts. Cached packaged predictions provide a deterministic fast reproduction of this build.

| Period | CAGR | Sharpe | Max DD | Positive months | Turnover/year |
|---|---:|---:|---:|---:|---:|
| Full OOS, 2023-08-05–2026-08-03 | 54.45% | 2.20 | 13.85% | 32/37 | 31.7x |
| Latest year, 2025-08-05–2026-08-03 | 75.91% | 2.36 | 13.85% | 11/13 | 30.4x |

The prediction caches shipped in this archive were trained under Linux/LightGBM in the build environment and reproduce approximately **53.21% CAGR / 2.151 Sharpe** over full OOS and **74.46% CAGR / 2.316 Sharpe** over the latest year. The original research run above used a different runtime build. The economic path, drawdown, turnover and latest-year profitable-month count remain very close; one near-zero full-OOS month can flip sign. Both the original targets and the packaged-build output are retained under `reference/`.

A separate `configs/baseline_h7.yaml` reproduces the earlier single 7-day LambdaRank strategy.

## Output

Each run creates a new directory under `outputs/runs/`:

```text
YYYYMMDD_HHMMSS_best_v2/
├── config_resolved.yaml
├── run_metadata.json
├── environment.txt
├── summary.json
├── daily.csv
├── monthly.csv
├── yearly.csv
├── fee_stress.csv
├── equity.png
└── drawdown.png
```

Feature arrays and model predictions are cached independently under `.cache/`.
No run silently overwrites another run.

## Input schema

Every file in `data/1d/` must be named `<SYMBOL>_1440.csv` and contain:

```text
timestamp,open,high,low,close,volume
```

Dates must be unique, increasing daily timestamps. `BTC_1440.csv` is mandatory.

## Extending the pipeline

- Add a horizon by adding a model entry to YAML. Existing features are reused.
- Add features inside `build_features()` in `research.py`; the feature-cache key should then be versioned if semantics change.
- Add another model family beside `train_walk_forward()` and keep prediction output in the same `(date, asset)` score format.
- Change portfolio policy only in `portfolio.py`; training remains independent.
- Add overlays through `overlay_multiplier()` and expose every parameter in YAML.

## Tests

```bash
conda run -n NN python -m pip install --no-build-isolation -e '.[dev]'
conda run -n NN pytest -q
```
