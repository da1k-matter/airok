#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/lightgbm_h7.yaml}"
DATA_DIR="${RANKTREND_DATA:-data/1d}"

python -m ranktrend doctor --config "$CONFIG" --data-dir "$DATA_DIR"
python -m ranktrend run --config "$CONFIG" --data-dir "$DATA_DIR"
