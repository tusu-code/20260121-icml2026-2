"""
ReVOS-only calibration finetune (calib(maskdec_only)) + DINOv2 REPA-global KD + vision LoRA.

Design goal (C route):
- Keep the strong `maskdec_only` calibration recipe.
- Enable REPA-global distillation with cached DINOv2 globals so KD is *actually applied*.
- Allow ONLY vision LoRA (keep LLM frozen/no LLM LoRA) so KD gradients can update vision reps
  without turning this into a heavy full finetune.
"""

from mmengine.config import read_base
from peft import LoraConfig

with read_base():
    from .main_revos_only_calib_maskdec_only_from_refcoco_ft_dw0002_iter2000_lr2e5_iter500 import *  # noqa: F401,F403

from projects.main.hooks import DistillWeightSchedulerHook

# -----------------------
# Use local UpstreamVL3-2B weights (avoid network / hf-mirror timeouts)
# -----------------------
# NOTE (anonymous template):
# Replace with your local pretrained model path or HF repo id.
path = "<PATH_TO_PRETRAINED_MODEL_OR_HF_REPO>"
if isinstance(tokenizer, dict):
    tokenizer["pretrained_model_name_or_path"] = path
if isinstance(model, dict):
    m = model.get("mllm", {})
    if isinstance(m, dict):
        m["model_path"] = path


# -----------------------
# KD + vision LoRA switches
# -----------------------
# NOTE (anonymous template):
# Replace with your local distillation cache directory.
DISTILL_CACHE_DIR = "<PATH_TO_DISTILL_CACHE_DIR>"
DISTILL_WEIGHT = 0.002  # align with dw0002 target scale (small & safe)

if isinstance(model, dict):
    model["distill_weight"] = float(DISTILL_WEIGHT)

    m = model.get("mllm", {})
    if isinstance(m, dict):
        # Keep LLM frozen; no LLM LoRA.
        m["freeze_llm"] = True
        m.pop("llm_lora", None)

        # Keep base vision frozen but enable PEFT via vision LoRA.
        m["freeze_visual_encoder"] = True
        m["visual_encoder_lora"] = dict(
            type=LoraConfig,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
        )


# -----------------------
# Attach distill cache to ReVOS dataset so KD is actually active
# -----------------------
def _set_distill_recursively(ds_cfg):
    """Recursively attach distill cache to the actual ReVOS dataset config.

    NOTE:
    train_dataloader['dataset'] is sometimes wrapped (e.g. RepeatDataset),
    so we must walk through nested dicts/lists to find the underlying dataset
    entries (where `name=='ReVOS'` / `type==Main03RefVOS`) and set distill args.
    """
    if ds_cfg is None:
        return
    if isinstance(ds_cfg, (list, tuple)):
        for x in ds_cfg:
            _set_distill_recursively(x)
        return
    if not isinstance(ds_cfg, dict):
        return

    # If this looks like a dataset wrapper, recurse into its children.
    if "dataset" in ds_cfg:
        _set_distill_recursively(ds_cfg.get("dataset"))
    if "datasets" in ds_cfg:
        _set_distill_recursively(ds_cfg.get("datasets"))

    # Patch the actual ReVOS dataset entry.
    name = str(ds_cfg.get("name", ""))
    if name == "ReVOS":
        ds_cfg["distill_cache_dir"] = DISTILL_CACHE_DIR
        ds_cfg["distill_cache_type"] = "global"


if isinstance(train_dataloader, dict):
    _set_distill_recursively(train_dataloader.get("dataset", None))


# -----------------------
# Distill weight scheduling (warmup -> ramp -> hold)
# -----------------------
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
work_dir = "./work_dirs/main_revos_only_calib_maskdec_only_from_refcoco_ft_dw0002_iter2000_lr2e5_iter500_visionlora_kd"


# -----------------------
# AMP/GradScaler safety
# -----------------------
# On this server / torch build, GradScaler does not support BFloat16 unscale:
#   RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
#
# For robustness (and since we only run 500 iters), we bypass AmpOptimWrapper here.
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=1, error_if_nonfinite=False),
    accumulative_counts=1,
)


