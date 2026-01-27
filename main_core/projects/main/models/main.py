from typing import Literal
from collections import OrderedDict
from pycocotools import mask as _mask
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from third_parts.mmdet.models.utils.point_sample import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness

from peft import PeftModelForCausalLM

from transformers import AutoImageProcessor, AutoVideoProcessor

class MainModel(BaseModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 # Weight for language modeling loss (dialogue quality knob).
                 # NOTE: This only has effect when the LLM (or its LoRA) is trainable.
                 llm_loss_weight: float = 1.0,
                 # Optional: freeze the small text projection MLP used for seg token hidden states.
                 # Useful for "mask-decoder-only" calibration experiments.
                 freeze_text_hidden_fcs: bool = False,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 template=None,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'upstreamllm', 'llava']='intern_vl',
                 # ext
                 # preprocessor=None,
                 # bs
                 training_bs:int=0,
                 # distillation (optional): teacher global frame embedding alignment
                 distill_weight: float = 0.0,
                 # distillation (optional): teacher global embedding dim (e.g. DINOv2 ViT-B/14 = 768)
                 distill_teacher_dim: int = 768,
                 # distillation (optional): student global pooling ("mean" over vision tokens, or "cls")
                 distill_student_pool: Literal['mean', 'cls'] = 'mean',
                 # distillation (optional): loss type ("cosine" or "mse" on normalized vectors)
                 distill_loss_type: Literal['cosine', 'mse'] = 'cosine',
                 # distillation (optional): apply probability (use <1.0 to reduce KD pressure)
                 distill_prob: float = 1.0,
                 # distillation (optional): multi-teacher config
                 # Example:
                 #   distill_teachers=dict(
                 #     dinov2=dict(dim=768, weight=0.02, loss_type="cosine", student_pool="mean"),
                 #     videomae=dict(dim=768, weight=0.01, loss_type="cosine", student_pool="cls", trd_weight=0.002, trd_tau=0.07, trd_k=64),
                 #   )
                 distill_teachers: dict | None = None,
                 # debug
                 print_grad_status: bool = False,
                 ):
        super().__init__()
        if special_tokens is None:
            special_tokens = ['[SEG]']

        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens(tokenizer, special_tokens)

        if arch_type == 'upstreamllm':
            image_processor = AutoImageProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            video_processor = AutoVideoProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            self.mllm._init_processor(image_processor, video_processor)

        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        in_dim = self.mllm.get_embedding_size()
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        if bool(freeze_text_hidden_fcs):
            self.text_hidden_fcs.requires_grad_(False)
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.template = template
        self.bs = training_bs

        # Language loss weight (dialogue / instruction-following tradeoff knob)
        self.llm_loss_weight = float(llm_loss_weight)

        # Distillation (optional)
        self.distill_weight = float(distill_weight)
        self.distill_teacher_dim = int(distill_teacher_dim)
        self.distill_student_pool = str(distill_student_pool)
        self.distill_loss_type = str(distill_loss_type)
        self.distill_prob = float(distill_prob)
        # IMPORTANT:
        # - Do NOT lazily create new trainable params inside forward() in mmengine:
        #   optimizer is constructed before the first forward, so late params won't be optimized.
        # - Create projector up-front so it is included in optimizer + checkpoints.
        #
        # For UpstreamVL, `extract_feature()` outputs are already in the LLM hidden size space,
        # so student_dim equals `self.mllm.get_embedding_size()`.
        self.distill_projector = None  # default teacher projector (backward compatible)
        self.distill_projectors_extra = nn.ModuleDict()  # extra teacher projectors

        # Normalize multi-teacher config (keep backward compatibility).
        self.distill_teachers = distill_teachers if isinstance(distill_teachers, dict) else None

        if self.distill_teachers is None:
            # Old single-teacher behavior uses the default knobs above.
            if self.distill_weight > 0:
                if self.arch_type != 'intern_vl':
                    raise NotImplementedError("repa-global distill currently supports intern_vl only")
                stu_dim = int(self.mllm.get_embedding_size())
                tea_dim = int(self.distill_teacher_dim)
                if tea_dim <= 0 or stu_dim <= 0:
                    raise ValueError(f"Invalid distill dims: student={stu_dim}, teacher={tea_dim}")
                self.distill_projector = nn.Linear(stu_dim, tea_dim, bias=False)
        else:
            # Multi-teacher mode: create projectors per teacher.
            # NOTE: we keep the legacy `distill_projector` unused (None) to avoid confusion.
            if self.arch_type != 'intern_vl':
                raise NotImplementedError("repa-global distill currently supports intern_vl only")
            stu_dim = int(self.mllm.get_embedding_size())
            if stu_dim <= 0:
                raise ValueError(f"Invalid distill dims: student={stu_dim}")
            for name, cfg in self.distill_teachers.items():
                try:
                    tea_dim = int(cfg.get("dim", 0))
                except Exception:
                    tea_dim = 0
                if tea_dim <= 0:
                    raise ValueError(f"Invalid distill teacher dim for {name}: {tea_dim}")
                self.distill_projectors_extra[name] = nn.Linear(stu_dim, tea_dim, bias=False)

        self.print_grad_status = bool(print_grad_status)

        # LoRA is optional. Some finetune configs intentionally disable LoRA and
        # only train small non-LLM modules (e.g. SAM2 mask decoder calibration).
        # In that case `llm_lora_config` can be None and xtuner's UpstreamVL LoRA
        # prepare helper would crash. Guard it here for robustness.
        llm_lora_cfg = getattr(self.mllm, "llm_lora_config", None)
        if llm_lora_cfg is not None:
            self.mllm.manual_prepare_llm_for_lora()
            self.mllm.use_llm_lora = True
        else:
            self.mllm.use_llm_lora = False

        # Visual encoder LoRA is optional (for parameter-efficient KD / adaptation).
        # Enable it only when a config is provided.
        ve_lora_cfg = getattr(self.mllm, "visual_encoder_lora_config", None)
        if ve_lora_cfg is not None:
            try:
                self.mllm.manual_prepare_visual_encoder_for_lora()
                self.mllm.use_visual_encoder_lora = True
            except Exception as e:
                # Never fail init due to LoRA plumbing; surface error on logs.
                print(f"[Main] visual-encoder LoRA prepare failed: {e}")
                self.mllm.use_visual_encoder_lora = False
        else:
            self.mllm.use_visual_encoder_lora = False

        # Debug: print gradient status.
        #
        # NOTE: Printing every parameter can easily look like a "hang" in multi-GPU
        # training because stdout becomes the bottleneck. So we only print a SUMMARY
        # by default, and only print per-parameter details when explicitly enabled.
        #
        # - enable summary:
        #     export MAIN_PRINT_GRAD_STATUS=1
        # - enable full per-parameter listing (VERY SLOW):
        #     export MAIN_PRINT_GRAD_STATUS=1
        #     export MAIN_PRINT_GRAD_STATUS_FULL=1
        debug_enabled = self.print_grad_status or os.environ.get("MAIN_PRINT_GRAD_STATUS", "0") == "1"
        if debug_enabled:
            rank0 = True
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    rank0 = dist.get_rank() == 0
            except Exception:
                rank0 = True

            if rank0:
                try:
                    base_model = self.mllm.model
                    total_params = 0
                    trainable_params = 0
                    full = os.environ.get("MAIN_PRINT_GRAD_STATUS_FULL", "0") == "1"

                    if full:
                        print("\n" + "=" * 80)
                        print("GRADIENT STATUS OF MLLM.MODEL WEIGHTS")
                        print("=" * 80)

                    for name, param in base_model.named_parameters():
                        total_params += param.numel()
                        if param.requires_grad:
                            trainable_params += param.numel()

                        if full:
                            grad_status = "✓ TRAINABLE" if param.requires_grad else "✗ FROZEN"
                            print(
                                f"{name:<60} | {grad_status} | Shape: {tuple(param.shape)} | Params: {param.numel():,}"
                            )

                    print("-" * 80)
                    print("MLLM PARAM SUMMARY:")
                    print(f"  Total parameters: {total_params:,}")
                    print(f"  Trainable parameters: {trainable_params:,}")
                    print(f"  Frozen parameters: {total_params - trainable_params:,}")
                    if total_params > 0:
                        print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
                    print("=" * 80)
                except Exception as e:
                    # never fail training for debug printing
                    print(f"[Main] grad-status debug print failed: {e}")

    # ---------------------------------------------------------------------
    # mmengine init dump control
    # ---------------------------------------------------------------------
    # mmengine's BaseModule.init_weights() dumps per-parameter init info to the
    # logger file by default. For very large models (e.g., UpstreamVL3-8B + SAM2),
    # this produces massive logs and can look like a "hang" during startup.
    #
    # We disable this dump by default. To re-enable for debugging:
    #   export MAIN_DUMP_INIT_INFO=1
    def _dump_init_info(self):
        if os.environ.get("MAIN_DUMP_INIT_INFO", "0") != "1":
            return

        # Only dump on rank0 to avoid duplicated logs in distributed runs.
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass

        try:
            return super()._dump_init_info()
        except Exception:
            return


    def _add_special_tokens(self, tokenizer, special_tokens):
        self.mllm.add_special_tokens(tokenizer, special_tokens)
        self.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0] # required to make add_special_tokens to be False to avoid <bos> or <eos>

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    def _merge_lora(self):
        if isinstance(self.mllm.model, PeftModelForCausalLM):
            self.mllm.model = self.mllm.model.merge_and_unload()
            return
        
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

    def state_dict(self, *args, **kwargs):
        prefix = kwargs.pop('prefix', '')
        state_dict_mllm = self.mllm.state_dict(*args, prefix=prefix + 'mllm.', **kwargs)
        state_dict_sam2 = self.grounding_encoder.state_dict(*args, prefix=prefix + 'grounding_encoder.', **kwargs)
        state_dict_text = self.text_hidden_fcs.state_dict(*args, prefix=prefix + 'text_hidden_fcs.', **kwargs)
        state_dict_distill = OrderedDict()
        if self.distill_projector is not None:
            state_dict_distill.update(self.distill_projector.state_dict(*args, prefix=prefix + 'distill_projector.', **kwargs))
        if hasattr(self, "distill_projectors_extra") and isinstance(self.distill_projectors_extra, nn.ModuleDict):
            state_dict_distill.update(self.distill_projectors_extra.state_dict(*args, prefix=prefix + 'distill_projectors_extra.', **kwargs))
        to_return = OrderedDict()
        to_return.update(state_dict_mllm)
        to_return.update(
            {k: v
             for k, v in state_dict_sam2.items() if k.startswith('grounding_encoder.sam2_model.sam_mask_decoder')})
        to_return.update(state_dict_text)
        to_return.update(state_dict_distill)
        return to_return

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            # assert len(pred_mebeds) == len(gt_masks)
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        distill_teacher_global = data.pop('distill_teacher_global', None)
        distill_valid = data.pop('distill_valid', None)
        distill_teacher_global_map = data.pop('distill_teacher_global_map', None)
        distill_teacher_tokens_map = data.pop('distill_teacher_tokens_map', None)
        distill_valid_map = data.pop('distill_valid_map', None)
        input_ids = data['input_ids']
        output = self.mllm(data, data_samples, mode)

        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx

        hidden_states = output.hidden_states
        # NOTE: In some environments / checkpoints, the MLLM may output BF16
        # hidden states, while `text_hidden_fcs` is initialized in FP32.
        # Torch linear requires mat1/mat2 dtype match, so cast inputs to the
        # module's parameter dtype to avoid:
        #   RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
        hs_last = hidden_states[-1]
        try:
            fc_dtype = next(self.text_hidden_fcs.parameters()).dtype
            if hs_last.dtype != fc_dtype:
                hs_last = hs_last.to(fc_dtype)
        except StopIteration:
            pass
        hidden_states = self.text_hidden_fcs(hs_last)

        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
        pred_embeddings_list_video = self.generate_video_pred_embeddings(
            pred_embeddings_list, frames_per_batch)

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video)
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None]
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1)


        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss * self.llm_loss_weight,
        }

        # Optional: distillation
        if data.get('pixel_values', None) is not None:
            # Multi-teacher mode
            if self.distill_teachers is not None and (distill_teacher_global_map is not None or distill_teacher_tokens_map is not None):
                try:
                    loss_distill_total = None
                    # Global losses
                    if isinstance(distill_teacher_global_map, dict):
                        for name, cfg in self.distill_teachers.items():
                            w = float(cfg.get("weight", 0.0))
                            if w <= 0:
                                continue
                            tea_list = distill_teacher_global_map.get(name, None)
                            if tea_list is None:
                                continue
                            vmask = None
                            if isinstance(distill_valid_map, dict):
                                vmask = distill_valid_map.get(name, None)
                            loss_g = self._compute_repa_global_loss(
                                pixel_values=data['pixel_values'],
                                distill_teacher_global=tea_list,
                                distill_valid=vmask,
                                device=input_ids.device,
                                projector=(self.distill_projectors_extra[name] if name in self.distill_projectors_extra else None),
                                student_pool=str(cfg.get("student_pool", "mean")),
                                loss_type=str(cfg.get("loss_type", "cosine")),
                            )
                            loss_w = loss_g * w
                            loss_dict[f'loss_distill_global_{name}'] = loss_w
                            loss_distill_total = loss_w if loss_distill_total is None else (loss_distill_total + loss_w)

                    # TRD losses (token relation distillation)
                    if isinstance(distill_teacher_tokens_map, dict):
                        for name, cfg in self.distill_teachers.items():
                            w = float(cfg.get("trd_weight", 0.0))
                            if w <= 0:
                                continue
                            tea_list = distill_teacher_tokens_map.get(name, None)
                            if tea_list is None:
                                continue
                            vmask = None
                            if isinstance(distill_valid_map, dict):
                                vmask = distill_valid_map.get(name, None)
                            tau = float(cfg.get("trd_tau", 0.07))
                            k = int(cfg.get("trd_k", 64))
                            loss_t = self._compute_trd_loss(
                                pixel_values=data['pixel_values'],
                                distill_teacher_tokens=tea_list,
                                distill_valid=vmask,
                                device=input_ids.device,
                                projector=(self.distill_projectors_extra[name] if name in self.distill_projectors_extra else None),
                                tau=tau,
                                k=k,
                            )
                            loss_w = loss_t * w
                            loss_dict[f'loss_distill_trd_{name}'] = loss_w
                            loss_distill_total = loss_w if loss_distill_total is None else (loss_distill_total + loss_w)

                    if loss_distill_total is not None:
                        loss_dict['loss_distill'] = loss_distill_total
                except Exception as e:
                    print(f"[Main] multi-teacher distill skipped due to error: {e}")

            # Legacy single-teacher mode (backward compatible)
            elif self.distill_weight > 0 and distill_teacher_global is not None:
                # Optional Bernoulli gate (reduce distillation pressure without changing datasets).
                if self.distill_prob < 1.0:
                    try:
                        if torch.rand((), device=input_ids.device).item() > float(self.distill_prob):
                            return loss_dict
                    except Exception:
                        pass
                # Lightweight runtime stats (helps confirm distillation is actually applied).
                # - distill_teacher_global is a list (per-sample tensor or None)
                # - distill_valid may be a list/tuple of bool/tensor flags (per-sample)
                try:
                    n_total = 0
                    n_valid = 0
                    if isinstance(distill_teacher_global, (list, tuple)):
                        n_total = len(distill_teacher_global)
                        for i, tea in enumerate(distill_teacher_global):
                            if tea is None:
                                continue
                            ok = True
                            if distill_valid is not None:
                                try:
                                    ok = bool(distill_valid[i])
                                    if hasattr(distill_valid[i], "item"):
                                        ok = bool(distill_valid[i].item())
                                except Exception:
                                    ok = True
                            if ok:
                                n_valid += 1

                    if n_total > 0:
                        # Keep these as tensors so mmengine logger can pick them up.
                        loss_dict["distill_num_valid"] = torch.tensor(float(n_valid), device=input_ids.device)
                        loss_dict["distill_valid_ratio"] = torch.tensor(float(n_valid) / float(n_total), device=input_ids.device)
                except Exception:
                    # Never fail training for stats.
                    pass
                try:
                    loss_distill = self._compute_repa_global_loss(
                        pixel_values=data['pixel_values'],
                        distill_teacher_global=distill_teacher_global,
                        distill_valid=distill_valid,
                        device=input_ids.device,
                        projector=self.distill_projector,
                        student_pool=self.distill_student_pool,
                        loss_type=self.distill_loss_type,
                    )
                    loss_dict['loss_distill'] = loss_distill * self.distill_weight
                except Exception as e:
                    # Keep training robust; surface the error on rank0-ish logs
                    print(f"[Main] distill skipped due to error: {e}")
        return loss_dict

    def _compute_repa_global_loss(self, pixel_values, distill_teacher_global, distill_valid, device,
                                  projector: nn.Module = None,
                                  student_pool: str = "mean",
                                  loss_type: str = "cosine"):
        """
        Compute frame-level global cosine alignment loss between:
        - student: UpstreamVL vision tokens mean pooled per frame
        - teacher: cached global embedding per frame (loaded by dataset)
        """
        if self.arch_type != 'intern_vl':
            raise NotImplementedError("repa-global distill currently supports intern_vl only")
        if not isinstance(pixel_values, list):
            raise ValueError("pixel_values must be a list (per-sample tensors)")
        if not isinstance(distill_teacher_global, list):
            raise ValueError("distill_teacher_global must be a list (per-sample tensors/None)")

        # concat all frames across batch
        vision_dtype = getattr(getattr(self.mllm, "model", None), "vision_model", None)
        vision_dtype = getattr(vision_dtype, "dtype", self.torch_dtype)
        concat_images = torch.cat([pv.to(device=device, dtype=vision_dtype) for pv in pixel_values], dim=0)

        # [total_frames, n_tokens, C]
        vit_embeds = self.mllm.model.extract_feature(concat_images)
        # student global pooling
        # NOTE: some ViT backbones use token 0 as CLS; if not, "cls" may degrade.
        if str(student_pool) == "cls":
            student_global = vit_embeds[:, 0]
        else:
            student_global = vit_embeds.mean(dim=1)  # [total_frames, C]

        splits = [int(pv.shape[0]) for pv in pixel_values]
        student_list = list(torch.split(student_global, splits, dim=0))

        # filter valid samples
        stu_cat = []
        tea_cat = []
        for i, (stu_i, tea_i) in enumerate(zip(student_list, distill_teacher_global)):
            if tea_i is None:
                continue
            if distill_valid is not None:
                try:
                    if bool(distill_valid[i].item()) is False:
                        continue
                except Exception:
                    pass
            if not torch.is_tensor(tea_i):
                continue
            # Support both per-frame [T, D] and clip-level [D] teacher embeddings.
            if tea_i.ndim == 1:
                tea_i = tea_i.unsqueeze(0).repeat(stu_i.shape[0], 1)
            if tea_i.ndim != 2:
                continue
            if tea_i.shape[0] != stu_i.shape[0]:
                continue
            tea_i = tea_i.to(device=stu_i.device, dtype=stu_i.dtype)
            stu_cat.append(stu_i)
            tea_cat.append(tea_i)

        if len(stu_cat) == 0:
            return student_global.mean() * 0.0

        stu_all = torch.cat(stu_cat, dim=0)
        tea_all = torch.cat(tea_cat, dim=0)

        if projector is None:
            raise RuntimeError("projector is None but distill is enabled; check init/config")
        if (projector.in_features != stu_all.shape[-1]
                or projector.out_features != tea_all.shape[-1]):
            raise ValueError(
                f"distill_projector shape mismatch: proj=({projector.in_features}->{projector.out_features}) "
                f"but got student_dim={stu_all.shape[-1]}, teacher_dim={tea_all.shape[-1]}. "
                "Fix by setting distill_teacher_dim to match cached teacher embeddings."
            )
        # keep projector on the right device/dtype (no new params created)
        if projector.weight.device != stu_all.device or projector.weight.dtype != stu_all.dtype:
            projector = projector.to(device=stu_all.device, dtype=stu_all.dtype)

        stu_proj = projector(stu_all)
        stu_proj = F.normalize(stu_proj, dim=-1)
        tea_all = F.normalize(tea_all, dim=-1)
        if str(loss_type) == "mse":
            loss = F.mse_loss(stu_proj, tea_all)
        else:
            loss = (1.0 - F.cosine_similarity(stu_proj, tea_all, dim=-1)).mean()
        return loss

    def _compute_trd_loss(self, pixel_values, distill_teacher_tokens, distill_valid, device,
                          projector: nn.Module = None,
                          tau: float = 0.07,
                          k: int = 64) -> torch.Tensor:
        """
        Token Relation Distillation (TRD) loss:
        align pairwise token relations between student vision tokens and teacher tokens.

        - student tokens: UpstreamVL vision tokens per frame
        - teacher tokens: cached per-frame tokens loaded by dataset (Tensor[T, N, D])
        """
        if self.arch_type != 'intern_vl':
            raise NotImplementedError("TRD distill currently supports intern_vl only")
        if projector is None:
            raise RuntimeError("projector is None but TRD is enabled; check init/config")
        if not isinstance(pixel_values, list):
            raise ValueError("pixel_values must be a list (per-sample tensors)")
        if not isinstance(distill_teacher_tokens, list):
            raise ValueError("distill_teacher_tokens must be a list (per-sample tensors/None)")

        # concat all frames across batch
        vision_dtype = getattr(getattr(self.mllm, "model", None), "vision_model", None)
        vision_dtype = getattr(vision_dtype, "dtype", self.torch_dtype)
        concat_images = torch.cat([pv.to(device=device, dtype=vision_dtype) for pv in pixel_values], dim=0)

        # [total_frames, n_tokens, C]
        vit_embeds = self.mllm.model.extract_feature(concat_images)
        # use patch tokens if possible
        if vit_embeds.shape[1] > 1:
            vit_tokens = vit_embeds[:, 1:]
        else:
            vit_tokens = vit_embeds

        n_tok = int(vit_tokens.shape[1])
        k_use = min(int(k), n_tok)
        if k_use <= 1:
            return vit_tokens.mean() * 0.0
        # deterministic evenly-spaced token indices
        idx = torch.linspace(0, n_tok - 1, steps=k_use, device=vit_tokens.device).long()
        stu_tokens = vit_tokens.index_select(1, idx)  # [F, K, C]

        # project to teacher dim
        if projector.weight.device != stu_tokens.device or projector.weight.dtype != stu_tokens.dtype:
            projector = projector.to(device=stu_tokens.device, dtype=stu_tokens.dtype)
        stu_tokens = projector(stu_tokens)  # [F, K, Dtea]

        splits = [int(pv.shape[0]) for pv in pixel_values]
        stu_list = list(torch.split(stu_tokens, splits, dim=0))  # each [T, K, Dtea]

        loss_sum = None
        n_used = 0
        for i, (stu_i, tea_i) in enumerate(zip(stu_list, distill_teacher_tokens)):
            if tea_i is None:
                continue
            if distill_valid is not None:
                try:
                    if bool(distill_valid[i].item()) is False:
                        continue
                except Exception:
                    pass
            if not torch.is_tensor(tea_i) or tea_i.ndim != 3:
                continue
            if tea_i.shape[0] != stu_i.shape[0]:
                continue
            # tea_i: [T, N, Dtea]
            tea_i = tea_i.to(device=stu_i.device, dtype=stu_i.dtype)
            n_tok_t = int(tea_i.shape[1])
            k_t = min(k_use, n_tok_t)
            if k_t <= 1:
                continue
            idx_t = torch.linspace(0, n_tok_t - 1, steps=k_t, device=tea_i.device).long()
            tea_tok = tea_i.index_select(1, idx_t)  # [T, Kt, D]
            # match K
            if k_t != k_use:
                # truncate to common K
                kk = min(k_t, k_use)
                tea_tok = tea_tok[:, :kk]
                stu_tok = stu_i[:, :kk]
            else:
                kk = k_use
                stu_tok = stu_i

            # normalize
            stu_n = F.normalize(stu_tok, dim=-1)
            tea_n = F.normalize(tea_tok, dim=-1)

            # relation matrices per frame: [T, K, K]
            sim_s = torch.matmul(stu_n, stu_n.transpose(-1, -2)) / float(max(tau, 1e-6))
            sim_t = torch.matmul(tea_n, tea_n.transpose(-1, -2)) / float(max(tau, 1e-6))

            p_t = F.softmax(sim_t, dim=-1)
            log_p_s = F.log_softmax(sim_s, dim=-1)
            # KL per frame, average
            kl = F.kl_div(log_p_s, p_t, reduction="batchmean")
            loss_sum = kl if loss_sum is None else (loss_sum + kl)
            n_used += 1

        if n_used == 0:
            return stu_tokens.mean() * 0.0
        return loss_sum / float(n_used)


    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def generate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        assert len(pred_embeddings_list) == len(frames_per_batch)
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def preparing_for_generation(self, metainfo, **kwargs):
        raise NotImplementedError("Main does not support preparing for generation, please use predict_video instead.")

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle
