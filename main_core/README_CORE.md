## Core method code (REPA-Global)

This folder contains **a minimal subset** of the Main training/eval code needed to show our method implementation:

- `projects/main/models/main.py`: REPA-Global distillation loss (global embedding alignment) + optional multi-teacher + `llm_loss_weight`.
- `projects/main/hooks/distill_weight_scheduler.py`: distillation-weight scheduling hook.
- `projects/main/datasets/base.py`: distillation cache loading utilities.
- `projects/main/evaluation/main_eval_ref_vos.py`: reference VOS evaluation entry.
- `projects/main/hf/models/*`: HF runtime pieces used by evaluation (including robust `[SEG]` fallback).
- `projects/main/configs/*`: example KD-on/off configs used in experiments.

Notes:
- This is **not a standalone repo**; it is intended to be dropped into the upstream Main codebase.
- Model weights and private dataset paths are **not included**.
