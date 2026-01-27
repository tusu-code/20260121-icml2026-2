# 🎬 Anonymous Demo (Code + Assets)

This repository contains:
- **Demo assets**: `assets/demo.gif`, `assets/demo_seg.gif`
- **Reproducible demo script**: generates `demo.gif`, `demo_seg.gif`, and a `README_DEMO.md` snippet

---

## Quick Preview

**Instruction / Question:** the ship that is the farthest from the camera.

| **Original Video** | **Segmentation Overlay** |
| --- | --- |
| ![](assets/demo.gif) | ![](assets/demo_seg.gif) |

---

## Run the demo script (with your own HF model + video)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
bash scripts/run_demo.sh /path/to/hf_model /path/to/video.mp4 "Please segment the target object." ./outputs/demo1
```

Outputs are saved under `./outputs/demo1/`.

## Notes
- Model weights are **NOT** included.
- The overlay GIF renders a **single automatically selected mask** (picked to be the most "single-object" among all predicted masks).

---

## dw012 Training / Conversion / Evaluation (templates)

This repo includes **template configs and scripts** under `configs/` and `scripts/`:
- `configs/dw012_calib_template.py`
- `configs/dialogue_boost_template.py`
- `scripts/train_dw012_calib.sh`
- `scripts/convert_to_hf.sh`
- `scripts/eval_revos_mevisu.sh`

These are **command/config templates** to document how dw012-style runs were executed.
They intentionally do **not** include any model weights.

---

## Extra utilities included (anonymous)

Under `main_core/tools/` we also include:
- `tools/train/wait_any_gpu_then_run.sh`: utility to queue a job on any free GPU.
- `tools/eval/eval_revos.py`: compute ReVOS metrics from a `results.json`.
- `tools/eval/eval_mevis.py`: compute MeVIS_U metrics from a `results.json`.

These scripts are provided to document our evaluation pipeline. They may still require the full upstream Main repo and datasets.
