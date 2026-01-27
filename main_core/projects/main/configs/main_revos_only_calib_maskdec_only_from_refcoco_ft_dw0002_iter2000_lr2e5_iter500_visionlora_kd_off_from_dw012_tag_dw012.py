"""
KD-off control derived from the *same* dw012 KD-on config.

Goal:
- Keep everything identical to `..._visionlora_kd_dw012.py`
- Only turn KD off (distill_weight=0, remove DistillWeightSchedulerHook)

We tag this run as "dw012 KD-off" for ablation table pairing convenience.
"""

from mmengine.config import read_base

with read_base():
    from .main_revos_only_calib_maskdec_only_from_refcoco_ft_dw0002_iter2000_lr2e5_iter500_visionlora_kd_dw012 import *  # noqa: F401,F403

# KD off
if isinstance(model, dict):
    model["distill_weight"] = 0.0

_hooks = list(globals().get("custom_hooks", []) or [])
custom_hooks = [h for h in _hooks if "DistillWeightSchedulerHook" not in str(h.get("type", ""))]

# Dedicated output directory (tagged as dw012 for table pairing)
work_dir = "./work_dirs/main_revos_only_calib_maskdec_only_from_refcoco_ft_dw0002_iter2000_lr2e5_iter500_visionlora_kd_off_from_dw012_tag_dw012"


