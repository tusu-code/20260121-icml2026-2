"""Dialogue/QA boost config template (LLM LoRA), starting from a dw012 checkpoint.

This is a TEMPLATE. It is NOT directly runnable without the full Main training codebase.

Fill in:
- LOAD_FROM (dw012 KD-off checkpoint, e.g., iter_500.pth)
- LLaVA / VideoQA paths (optional)

Goal:
- Enable LLM LoRA
- Increase LLM loss weight
- Train on QA-heavy data
"""

LOAD_FROM = "<PATH_TO_DW012_KD_OFF_CHECKPOINT_PTH>"

# Optional datasets (set to empty strings to disable)
LLAVA_JSON = "<PATH_TO_llava_v1_5_mix665k.json>"
LLAVA_IMAGES = "<PATH_TO_llava_images_dir>"
VIDEOQA_JSON = "<PATH_TO_video_chat.json>"
VIDEOQA_VIDEOS = "<PATH_TO_Activity_Videos_dir>"

# LoRA knobs
LLM_LORA_R = 64
LLM_LORA_ALPHA = 128
LLM_LOSS_WEIGHT = 2.0

MAX_ITERS = 500
LR = 5e-5
