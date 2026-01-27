#!/usr/bin/env bash
set -euo pipefail

# TEMPLATE: convert a .pth checkpoint to a HF folder.
# Requires the full Main training codebase.

CFG="${1:-}"
CKPT="${2:-}"
OUT="${3:-}"
if [[ -z "$CFG" || -z "$CKPT" || -z "$OUT" ]]; then
  echo "Usage: bash scripts/convert_to_hf.sh <CFG_PY> <CKPT_PTH> <OUT_HF_DIR>"
  exit 2
fi

# Example:
#   PYTHONPATH=. python tools/convert_to_hf.py "$CFG" "$CKPT" --save-path "$OUT"

echo "Please run this inside the full Main repo with its conversion script available."
