#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="${1:-configs/best_v2.yaml}"
conda run -n NN python -m pip install --no-build-isolation -e .
conda run -n NN python -m ranktrend doctor --config "$CONFIG"
conda run -n NN python -m ranktrend reproduce --config "$CONFIG" --reuse-features --reuse-predictions
