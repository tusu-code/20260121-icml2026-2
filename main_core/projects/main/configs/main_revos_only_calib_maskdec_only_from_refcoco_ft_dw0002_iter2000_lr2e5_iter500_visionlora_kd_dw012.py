"""
C route (higher KD): ReVOS-only calib(maskdec_only) + DINOv2 REPA-global KD + vision LoRA.

Variant of:
  `main_revos_only_calib_maskdec_only_from_refcoco_ft_dw0002_iter2000_lr2e5_iter500_visionlora_kd.py`

Change:
- Increase distillation weight (and scheduler target) to 0.012.
"""

from mmengine.config import read_base

with read_base():
    from .main_revos_only_calib_maskdec_only_from_refcoco_ft_dw0002_iter2000_lr2e5_iter500_visionlora_kd import *  # noqa: F401,F403

from projects.main.hooks import DistillWeightSchedulerHook

# Higher KD
DISTILL_WEIGHT = 0.012

if isinstance(model, dict):
    model["distill_weight"] = float(DISTILL_WEIGHT)

# Update distill scheduler target (replace any existing DistillWeightSchedulerHook)
_hooks = list(globals().get("custom_hooks", []) or [])
custom_hooks = [h for h in _hooks if "DistillWeightSchedulerHook" not in str(h.get("type", ""))]
custom_hooks = custom_hooks + [
    dict(
        type=DistillWeightSchedulerHook,
        target=float(DISTILL_WEIGHT),
        warmup_iters=50,
        ramp_iters=100,
        log_interval=50,
    )
]

# Dedicated output directory
work_dir = "./work_dirs/main_revos_only_calib_maskdec_only_from_refcoco_ft_dw0002_iter2000_lr2e5_iter500_visionlora_kd_dw012"


