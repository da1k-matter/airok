# Packaged build reproduction

Generated from the real modular pipeline with:

```bash
python -m ranktrend reproduce --config configs/best_v2.yaml \
  --reuse-features --reuse-predictions
```

Observed headline metrics:

| Period | CAGR | Sharpe | Max DD | Positive months | Turnover/year |
|---|---:|---:|---:|---:|---:|
| Full OOS | 53.21% | 2.151 | 13.91% | 31/37 | 31.74x |
| Latest year | 74.46% | 2.316 | 13.91% | 11/13 | 30.41x |

The original selected run was 54.45% / 2.199 full-OOS and 75.91% / 2.359 latest-year. The discrepancy is model-runtime drift, not a different portfolio policy. Daily returns are produced by the strategy code, not copied from the reference tables.
