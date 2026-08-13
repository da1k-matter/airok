#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/lightgbm_h7.yaml}"
DATA_DIR="${RANKTREND_DATA:-data/1d}"

python src/cli.py doctor --config "$CONFIG" --data-dir "$DATA_DIR"
python src/cli.py run --config "$CONFIG" --data-dir "$DATA_DIR"
