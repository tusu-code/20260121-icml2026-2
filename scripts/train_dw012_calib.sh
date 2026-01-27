#!/usr/bin/env bash
set -euo pipefail

# TEMPLATE: run dw012 calibration training.
# Requires the full Main training codebase (mmengine/xtuner) and datasets.

CFG="${1:-}"
if [[ -z "$CFG" ]]; then
  echo "Usage: bash scripts/train_dw012_calib.sh <PATH_TO_CONFIG_PY>"
  exit 2
fi

# Example:
#   PYTHONPATH=. python tools/train.py "$CFG"

echo "Please run this inside the full Main repo with its training dependencies installed."
