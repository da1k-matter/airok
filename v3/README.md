# RankTrend v3

LightGBM LambdaRank research for cross-sectional Bybit perpetual ranking, with resumable Optuna NSGA-II optimisation over walk-forward validation.

## Included presets

| Config | Horizons | Fixed seeds | Rank aggregation |
|---|---:|---:|---|
| `configs/lightgbm_h7.yaml` | 7 | 11, 23, 37 | median |
| `configs/lightgbm_h7_h10.yaml` | 7 + 10 | 11, 23, 37 | median |
| `configs/tune_nsga2.yaml` | Optuna chooses one or two | 11, 23, 37 | mean or median |

The strategy forms a liquid point-in-time universe, predicts future BTC-beta-residual return deciles, ranks instruments cross-sectionally, builds a dollar- and BTC-beta-neutral long/short portfolio, smooths target weights causally, and charges one-way turnover costs.

## Standard run

Assuming the intended conda environment is already active:

```bash
PYTHONPATH=src python -m ranktrend run \
  --config configs/lightgbm_h7_h10.yaml \
  --data-dir /path/to/bybit/1d
```

The usual commands are also available separately:

```bash
PYTHONPATH=src python -m ranktrend doctor --config configs/lightgbm_h7_h10.yaml
PYTHONPATH=src python -m ranktrend build-features --config configs/lightgbm_h7_h10.yaml
PYTHONPATH=src python -m ranktrend train --config configs/lightgbm_h7_h10.yaml
PYTHONPATH=src python -m ranktrend backtest --config configs/lightgbm_h7_h10.yaml
```

## NSGA-II tuning

`configs/tune_nsga2.yaml` fixes the three seeds to `[11, 23, 37]`. It uses only the validation period `2023-08-05..2025-08-04` to optimise two objectives:

```text
maximize validation CAGR
maximize validation Sharpe
```

The last year, `2025-08-05..2026-08-04`, is never used as an NSGA-II objective. After every tuning run, the selected top 25 Pareto candidates are evaluated on it once as the test period.

```bash
PYTHONPATH=src python -m ranktrend tune \
  --config configs/tune_nsga2.yaml \
  --trials 200
```

The study is persisted in the SQLite path configured by `tuning.storage`, so rerunning the command continues the same study. `--trials` is the target total of completed trials, not an additional count. Outputs are written to `outputs/optuna/ranktrend_lgbm_nsga2/`:

```text
config_resolved.yaml
study_summary.json
trials.csv
pareto_frontier.csv
pareto_top_25.csv
holdout/selected_trials.csv
```

The first two queued trials are fixed anchors, evaluated with the same three seeds as every other trial:

| Anchor | Definition |
|---|---|
| v1-like | h7+h10 at 50/50, no row bagging, five instruments per side, `mean_rank` |
| v2-like | h7 only, active row bagging, six instruments per side, `median_rank` |

When the Pareto frontier has more than 25 candidates, the test set receives the 25 most balanced candidates: first by the weaker of normalised validation CAGR/Sharpe, then by their combined normalised score. This is a deterministic compromise rule; all Pareto candidates are retained in `pareto_frontier.csv`.

For every automatically selected candidate, `holdout/trial_<id>/` contains:

```text
validation_daily.csv
validation_equity.png
validation_drawdown.png
validation_summary.json
test_daily.csv
test_equity.png
test_drawdown.png
test_summary.json
```

You can also re-run test reporting for specific existing trials:

```bash
PYTHONPATH=src python -m ranktrend evaluate-trials \
  --config configs/tune_nsga2.yaml \
  --trial 7 --trial 42
```

This updates `outputs/optuna/ranktrend_lgbm_nsga2/holdout/selected_trials.csv` and the corresponding validation/test curves. Do not change the search space after examining these test results; that would turn the test period into another tuning set.

### Search space

The search uses three fixed seeds and explores:

- one horizon from 3 to 21 days, or two horizons: a short horizon from 3 to 14 days plus a 1 to 14 day gap, capped at 28 days;
- two-horizon blend weight from 20% to 80%;
- `mean_rank` or `median_rank` seed aggregation;
- LightGBM trees, learning rate, leaves, child size, row/feature sampling, L1/L2 regularisation, bins, LambdaRank truncation and sigmoid;
- train-window length, retrain frequency and training stride;
- liquid universe size, tail count, equal/inverse-volatility weights and smoothing length.

The overlay, neutrality rules, labels, costs, data, seeds, and evaluation dates are deliberately not optimised. The exact bounds are declared in [`configs/tune_nsga2.yaml`](configs/tune_nsga2.yaml).

## Data and outputs

CSV files belong under `data/1d/` and must be named `<SYMBOL>_1440.csv` with:

```text
timestamp,open,high,low,close,volume
```

`BTC_1440.csv` is mandatory. Standard runs write immutable directories under `outputs/runs/` containing resolved configuration, daily PnL, monthly/yearly tables, fee stress, current target weights and equity/drawdown images.

## Limitations

The supplied archive can have survivorship bias and does not model funding, mark-price/liquidation effects, order-book impact, or exchange execution. Backtest results are research evidence, not a live-performance guarantee.
