"""dw012 calibration config template (no weights included).

This is a TEMPLATE for reproducing the dw012-style calibration run.
It is NOT directly runnable without the full Main training codebase.

Fill in:
- PRETRAINED_PTH (official converted .pth)
- LOAD_FROM (refcoco-ft checkpoint)
- DATA_ROOT

Then run with the Main training stack (mmengine/xtuner).
"""

# TODO: replace placeholders with your local paths
PRETRAINED_PTH = "<PATH_TO_OFFICIAL_CONVERTED_PTH>"
LOAD_FROM = "<PATH_TO_REFC0CO_FT_CHECKPOINT_PTH>"
DATA_ROOT = "<PATH_TO_DATA_ROOT>"  # should contain video_datas/, ref_seg/, etc.

# Training hyperparams (example)
DISTILL_WEIGHT = 0.012
MAX_ITERS = 500
LR = 2e-5

# Notes:
# - Keep seeds fixed if you need strict reproducibility.
# - Do not commit any weights into the anonymous repo.
