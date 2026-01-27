#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-}"
VIDEO_PATH="${2:-}"
PROMPT="${3:-Please segment the target object.}"
OUT_DIR="${4:-./outputs/demo1}"

if [[ -z "$MODEL_PATH" || -z "$VIDEO_PATH" ]]; then
  echo "Usage: bash scripts/run_demo.sh <HF_MODEL_PATH> <VIDEO_PATH> [PROMPT] [OUT_DIR]"
  exit 2
fi

python demo/make_readme_demo.py \
  --video "$VIDEO_PATH" \
  --model_path "$MODEL_PATH" \
  --text "$PROMPT" \
  --out_dir "$OUT_DIR" \
  --frame_interval 1 --gif_fps 8 --max_frames 60 --max_size 512 \
  --mask_select best_single --max_segs 1

echo "[OK] outputs written to: $OUT_DIR"
