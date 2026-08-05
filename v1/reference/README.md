# Reference results

- `expected_best_v2.json` contains the headline metrics from the original selected research run.
- `expected_baseline_h7.json` contains the earlier 7-day single-model baseline.
- `packaged_build/` contains an actual reproduction generated from the prediction caches shipped with this archive.

A cached-score run is deterministic for a fixed NumPy/Pandas runtime. A complete LightGBM retraining may differ slightly across operating systems, compilers, CPU architectures and LightGBM versions. The CLI compares the run with the original research targets using the documented narrow tolerance; it never substitutes reference returns into the backtest.
