#!/usr/bin/env bash
set -euo pipefail

# TEMPLATE: run ReVOS(valid) and MeVIS_U(valid_u) evaluation.
# Requires the full Main repo (evaluation scripts + dataset files).

PRED_JSON="${1:-}"
if [[ -z "$PRED_JSON" ]]; then
  echo "Usage: bash scripts/eval_revos_mevisu.sh <PREDICTIONS_JSON>"
  exit 2
fi

# Example:
#   python tools/eval/eval_revos.py "$PRED_JSON"
#   python tools/eval/eval_mevis.py "$PRED_JSON" --save_name mevis_valid_u.json

echo "Please run this inside the full Main repo with eval scripts available."
