import argparse
import json
import os
import time

import mmengine
import numpy as np
from PIL import Image

import torch
import torch.distributed
import torch.utils.data
import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from projects.main.evaluation.dataset import RefVOSDataset
from projects.main.evaluation.utils import _init_dist_pytorch, _init_dist_slurm, get_dist_info, get_rank, collect_results_cpu

import concurrent.futures
from pycocotools import mask as cocomask


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle


def mask_save(item, mask_prediction, work_dir):
    vid_id = item['video_id']
    exp_id = item['exp_id']
    save_path = os.path.join(work_dir, 'Annotations', vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        mask = Image.fromarray(mask.astype(np.float32) * 255).convert('L')
        file_name = item['frames'][id_m]
        save_file = os.path.join(save_path, file_name + ".png")
        mask.save(save_file)


DATASETS_INFO = {
    'DAVIS': {
        'data_root': 'data/video_datas/davis17/',
        'image_folder': 'data/video_datas/davis17/valid/JPEGImages/',
        'expression_file': 'data/video_datas/davis17/meta_expressions/valid/meta_expressions.json',
        'mask_file': 'data/video_datas/davis17/valid/mask_dict.pkl',
    },
    'MEVIS': {
        'data_root': 'data/video_datas/mevis/valid/',
        'image_folder': 'data/video_datas/mevis/valid/JPEGImages',
        'expression_file': 'data/video_datas/mevis/valid/meta_expressions.json',
        'mask_file': None,
    },
    'MEVIS_U': {
        'data_root': 'data/video_datas/mevis/valid_u/',
        'image_folder': 'data/video_datas/mevis/valid_u/JPEGImages',
        'expression_file': 'data/video_datas/mevis/valid_u/meta_expressions.json',
        'mask_file': 'data/video_datas/mevis/valid_u/mask_dict.json',
    },
    'MEVIS_T': {
        'data_root': 'data/video_datas/mevis/test/',
        'image_folder': 'data/video_datas/mevis/test/JPEGImages',
        'expression_file': 'data/video_datas/mevis/test/meta_expressions_release.json',
        'mask_file': None,
    },
    'REFYTVOS': {
        'data_root': 'data/video_datas/rvos/',
        'image_folder': 'data/video_datas/rvos/valid/JPEGImages/',
        'expression_file': 'data/video_datas/rvos/meta_expressions/valid/meta_expressions.json',
        'mask_file': None,
    },
    'REVOS': {
        'data_root': 'data/video_datas/revos/',
        'image_folder': 'data/video_datas/revos/',
        'expression_file': 'data/video_datas/revos/meta_expressions_valid_.json',
        'mask_file': None,
    },
    'REF_SAV': {
        'data_root': 'data/ref_sav_eval/',
        'image_folder': 'data/ref_sav_eval/videos',
        'expression_file': 'data/ref_sav_eval/meta_expressions_valid.json',
        'mask_file': 'data/ref_sav_eval/mask_dict.json',
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='RefVOS')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_INFO.keys(),
        default='MEVIS',
        help='Specify a dataset')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--deepspeed', type=str, default=None) # dummy
    parser.add_argument('--data_root', default='./data', help='Root directory for all datasets.')
    # Debug / sanity-check helpers:
    # - Limit evaluation to the first N samples for quick diagnosis:
    #     --max_samples 5
    # - Print mask statistics (emptiness/area) per sample:
    #     MAIN_EVAL_DEBUG_MASK_STATS=1
    # - Abort early if masks are empty for the first N samples:
    #     MAIN_EVAL_ABORT_ON_ALL_EMPTY=1 (default)
    parser.add_argument('--max_samples', type=int, default=None, help='Only run the first N samples (debug).')
    # Resume / checkpointing for long evaluations on shared servers.
    # - Periodically dump partial results to disk so preemption/server switch won't waste hours.
    # - Re-run with --resume to skip items already in the partial file.
    parser.add_argument(
        '--save_every',
        type=int,
        default=int(os.environ.get("MAIN_EVAL_SAVE_EVERY", "50")),
        help='Save partial results every N processed samples (default: 50; env: MAIN_EVAL_SAVE_EVERY).',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from partial results in work_dir/<dataset>/results.partial.rank{rank}.json',
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    args = parse_args()

    # Update dataset paths with data_root
    for key, info in DATASETS_INFO.items():
        for path_key, path_val in info.items():
            if path_val is not None and ('folder' in path_key or 'file' in path_key or 'root' in path_key):
                DATASETS_INFO[key][path_key] = os.path.join(args.data_root, os.path.relpath(path_val, './data'))


    work_dir = args.work_dir
    if work_dir is None:
        work_dir = 'work_dirs/foobar'

    if args.launcher == 'none':
        rank = 0
        world_size = 1
    elif args.launcher == 'pytorch':
        import datetime
        _init_dist_pytorch('nccl', timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
    elif args.launcher == 'slurm':
        _init_dist_slurm('nccl')
        rank, world_size = get_dist_info()

    # NOTE:
    # - Default behavior keeps original semantics: load on CPU then move whole model to GPU via .cuda().
    # - On shared GPUs, ".cuda()" can OOM even if the GPU could otherwise fit (due to peak/fragmentation).
    #   Enable HF/Accelerate dispatch by setting:
    #     MAIN_EVAL_DEVICE_MAP=single   (load directly onto the visible GPU:0)
    #     MAIN_EVAL_DEVICE_MAP=auto     (allow CPU offload if needed)
    #   Optional:
    #     MAIN_EVAL_4BIT=1 / MAIN_EVAL_8BIT=1 (requires bitsandbytes; model support may vary)
    device_map_env = os.environ.get("MAIN_EVAL_DEVICE_MAP", "").strip().lower()
    use_4bit = os.environ.get("MAIN_EVAL_4BIT", "0") == "1"
    use_8bit = os.environ.get("MAIN_EVAL_8BIT", "0") == "1"

    # `low_cpu_mem_usage=True` loads the model with meta tensors first, then materializes weights.
    # Some remote-code models may have parameters/buffers that are not in the checkpoint state_dict;
    # if those remain on the meta device, they can crash at runtime.
    # You can disable this behavior for stability:
    #   MAIN_EVAL_LOW_CPU_MEM_USAGE=0
    low_cpu_mem_usage = os.environ.get("MAIN_EVAL_LOW_CPU_MEM_USAGE", "1") != "0"

    model_load_kwargs = dict(
        # transformers>=4.46 prefers `dtype` over the deprecated `torch_dtype`
        dtype=torch.bfloat16,
        low_cpu_mem_usage=low_cpu_mem_usage,
        use_flash_attn=True,
        trust_remote_code=True,
    )

    if device_map_env:
        # Transformers may run `caching_allocator_warmup()` automatically when `device_map` is set.
        # On shared/fragmented GPUs this warmup can itself trigger OOM before any real weights are loaded.
        # We disable it by default for evaluation stability.
        if os.environ.get("MAIN_EVAL_DISABLE_HF_WARMUP", "1") == "1":
            try:
                import transformers.modeling_utils as _main_mu  # type: ignore
                _main_mu.caching_allocator_warmup = lambda *a, **k: None  # type: ignore
            except Exception:
                pass

        # Dispatch weights at load-time (avoids a full-model .cuda()).
        if device_map_env in ("single", "cuda", "0"):
            # With CUDA_VISIBLE_DEVICES set, "0" refers to the first visible GPU.
            model_load_kwargs["device_map"] = {"": 0}
        else:
            # e.g. "auto"
            model_load_kwargs["device_map"] = device_map_env

        # Only set offload folder when using automatic dispatch that may spill to CPU.
        if device_map_env not in ("single", "cuda", "0"):
            model_load_kwargs["offload_folder"] = os.environ.get(
                "MAIN_EVAL_OFFLOAD_DIR", os.path.join(work_dir, "_hf_offload"))

        # Optional max_memory control to force CPU offload (helps when the full model doesn't fit on a single GPU).
        # Format: "0:28GiB,cpu:200GiB" (0 refers to the first visible GPU when CUDA_VISIBLE_DEVICES is set).
        max_mem_env = os.environ.get("MAIN_EVAL_MAX_MEMORY", "").strip()
        if max_mem_env:
            max_memory = {}
            for chunk in [c.strip() for c in max_mem_env.split(",") if c.strip()]:
                if ":" not in chunk:
                    continue
                k, v = [x.strip() for x in chunk.split(":", 1)]
                if k.lower() in ("cpu",):
                    max_memory["cpu"] = v
                else:
                    try:
                        max_memory[int(k)] = v
                    except Exception:
                        # allow "cuda:28GiB" shorthand
                        if k.lower() in ("cuda", "gpu"):
                            max_memory[0] = v
            if max_memory:
                model_load_kwargs["max_memory"] = max_memory

        # Optional quantization (prefer 4bit if both are set).
        if use_4bit:
            model_load_kwargs["load_in_4bit"] = True
        elif use_8bit:
            model_load_kwargs["load_in_8bit"] = True

        # Some remote-code models may reject extra kwargs depending on transformers version.
        try:
            model = AutoModel.from_pretrained(args.model_path, **model_load_kwargs).eval()
        except TypeError as e:
            # Retry without optional kwargs that are not universally supported.
            msg = str(e)
            if "offload_folder" in msg and "unexpected keyword argument" in msg:
                model_load_kwargs.pop("offload_folder", None)
                model = AutoModel.from_pretrained(args.model_path, **model_load_kwargs).eval()
            else:
                raise
    else:
        model = AutoModel.from_pretrained(args.model_path, **model_load_kwargs).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # Print effective eval knobs (helps explain sanity200 drift / speed-quality tradeoffs).
    if rank == 0:
        _mn = int(os.environ.get("MAIN_EVAL_MAX_NEW_TOKENS", "512"))
        _force_seg = os.environ.get("MAIN_EVAL_FORCE_SEG_TOKEN", "0")
        _seg_tok = os.environ.get("MAIN_EVAL_SEG_TOKEN_STR", "").strip() or os.environ.get("MAIN_EVAL_SEG_TOKEN", "").strip() or "[SEG]"
        _sort_videos = os.environ.get("MAIN_EVAL_SORT_VIDEOS", "1")
        print(
            f"[EVAL] max_samples={args.max_samples} save_every={args.save_every} resume={args.resume} "
            f"max_new_tokens(env)={_mn} force_seg(env)={_force_seg} seg_token={_seg_tok!r} "
            f"sort_videos(env)={_sort_videos}"
        )

    # Ensure model prediction config is initialized before we override seg token id.
    # Otherwise the first `predict_forward()` call may run `preparing_for_generation()`
    # internally and overwrite our override back to the default "[SEG]".
    try:
        if hasattr(model, "init_prediction_config") and (not getattr(model, "init_prediction_config")):
            if hasattr(model, "preparing_for_generation"):
                # Signature in MainChatModel: preparing_for_generation(self, tokenizer, max_new_tokens=..., ...)
                # NOTE: MainChatModel defaults to max_new_tokens=2048 which is very slow for RefVOS.
                # Use a smaller default, configurable via env.
                max_new_tokens = int(os.environ.get("MAIN_EVAL_MAX_NEW_TOKENS", "512"))
                try:
                    model.preparing_for_generation(tokenizer, max_new_tokens=max_new_tokens)
                except TypeError:
                    # Fall back for model variants that don't accept this kwarg.
                    model.preparing_for_generation(tokenizer)
    except Exception:
        # Best-effort only; model may manage init internally.
        pass

    # Some checkpoints may use a different "segmentation trigger token" than "[SEG]".
    # The Main model uses `model.seg_token_idx` to locate those token positions in the generated text
    # and then derives mask embeddings from their hidden states.
    #
    # You can override it for debugging/evaluation:
    #   MAIN_EVAL_SEG_TOKEN_STR='<ref>'
    seg_token_str = os.environ.get("MAIN_EVAL_SEG_TOKEN_STR", "").strip()
    if seg_token_str and hasattr(model, "seg_token_idx"):
        try:
            model.seg_token_idx = tokenizer.convert_tokens_to_ids(seg_token_str)
            if rank == 0:
                print(f"[EVAL] Override seg_token_idx using token {seg_token_str!r}: id={model.seg_token_idx}")
        except Exception as e:
            if rank == 0:
                print(f"[EVAL] Failed to override seg_token_idx with token {seg_token_str!r}: {e}")

    if 'upstreamllm' in args.model_path.lower():
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        processor = None

    dataset_info = DATASETS_INFO[args.dataset]

    # Ensure dataset subdir exists early (used by partial saves).
    dataset_out_dir = os.path.join(work_dir, args.dataset)
    os.makedirs(dataset_out_dir, exist_ok=True)

    def _partial_path_for_rank(r: int) -> str:
        return os.path.join(dataset_out_dir, f"results.partial.rank{r}.json")

    def _load_partial_results(path: str):
        """Load partial results and build a done-set (video_id, exp_id)."""
        if not os.path.exists(path):
            return [], set()
        try:
            obj = json.load(open(path, "r"))
            if not isinstance(obj, list):
                return [], set()
            done = set()
            for it in obj:
                try:
                    done.add((it.get("video_id"), it.get("exp_id")))
                except Exception:
                    pass
            return obj, done
        except Exception:
            return [], set()

    def _atomic_json_dump(obj, path: str):
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)


    dataset = RefVOSDataset(
        image_folder=dataset_info['image_folder'],
        expression_file=dataset_info['expression_file'],
        mask_file=dataset_info['mask_file'],
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=int(os.environ.get("REFVOS_EVAL_NUM_WORKERS", "4")),
        pin_memory=False,
        collate_fn=lambda x:x[0],
    )
    results = []
    # Resume support (rank-local).
    partial_path = _partial_path_for_rank(rank)
    done_keys = set()
    if args.resume:
        loaded, done_keys = _load_partial_results(partial_path)
        if loaded:
            results.extend(loaded)
            if rank == 0:
                print(f"[EVAL] Resuming: loaded {len(loaded)} partial results from {partial_path}")

    executor = concurrent.futures.ThreadPoolExecutor()
    debug_mask_stats = os.environ.get("MAIN_EVAL_DEBUG_MASK_STATS", "0") == "1"
    abort_on_all_empty = os.environ.get("MAIN_EVAL_ABORT_ON_ALL_EMPTY", "1") == "1"
    empty_frames_total = 0
    total_frames_total = 0
    empty_items_total = 0
    total_items_total = 0
    processed_since_save = 0
    last_save_t = time.time()
    save_every = max(int(args.save_every), 1)
    try:
        for item in tqdm.tqdm(dataloader):
            key = (item.get('video_id'), item.get('exp_id'))
            if key in done_keys:
                continue

            with torch.no_grad():
                # Compatibility:
                # - UpstreamLLM-based HF models may require `processor=...`
                # - UpstreamVL-based HF models (e.g. official AnonymousOrg/Main-UpstreamVL3-2B)
                #   may NOT accept `processor` in `predict_forward`.
                pf_kwargs = dict(
                    video=item['images'],
                    text=item['text_prompt'],
                    tokenizer=tokenizer,
                )
                if processor is not None:
                    try:
                        result = model.predict_forward(**pf_kwargs, processor=processor)
                    except TypeError as e:
                        # retry without processor for models that don't accept it
                        if "processor" in str(e) and "unexpected keyword" in str(e):
                            result = model.predict_forward(**pf_kwargs)
                        else:
                            raise
                else:
                    result = model.predict_forward(**pf_kwargs)

            text_idx = 0
            text_prediction = result['prediction']
            used_fallback_zeros = False
            if len(result.get('prediction_masks', [])) > 0:
                mask_prediction = result['prediction_masks'][text_idx]
            else:
                # If the model didn't return masks, fallback to all-zero masks.
                used_fallback_zeros = True
                mask_prediction = np.zeros((item['length'], item['ori_height'], item['ori_width']), dtype=np.uint8)

            # Optional mask sanity stats to quickly detect "all empty mask" failure mode.
            if debug_mask_stats and rank == 0:
                # mask_prediction: (T, H, W) uint8/bool
                mp = mask_prediction.astype(np.uint8)
                per_frame_sum = mp.reshape(mp.shape[0], -1).sum(axis=1)
                empty_frames = int((per_frame_sum == 0).sum())
                total_frames = int(per_frame_sum.shape[0])
                total_pos = int(per_frame_sum.sum())
                area_ratio = float(total_pos) / float(mp.size) if mp.size > 0 else 0.0
                empty_frames_total += empty_frames
                total_frames_total += total_frames
                is_all_empty_item = empty_frames == total_frames
                if is_all_empty_item:
                    empty_items_total += 1
                total_items_total += 1
                _seg_tok = os.environ.get("MAIN_EVAL_SEG_TOKEN_STR", "").strip() or "[SEG]"
                pred_has_seg = _seg_tok in (text_prediction or "")
                pred_preview = (text_prediction or "").replace("\n", "\\n")
                if len(pred_preview) > 160:
                    pred_preview = pred_preview[:160] + "..."
                print(
                    f"[MASK_STATS] vid={item.get('video_id')} exp={item.get('exp_id')} "
                    f"fallback={used_fallback_zeros} empty_frames={empty_frames}/{total_frames} "
                    f"area_ratio={area_ratio:.6f} seg_tok={_seg_tok!r} pred_has_seg={pred_has_seg} "
                    f"pred='{pred_preview}'"
                )

                # Abort early if the run is clearly broken (e.g., first N samples all-empty).
                if abort_on_all_empty and total_items_total >= 3 and empty_items_total == total_items_total:
                    raise RuntimeError(
                        "Detected all-empty masks for the first samples. "
                        "This usually means `predict_forward()` did not return valid masks, "
                        "or post-processing thresholding collapsed to empty. "
                        "Re-run with a different loading strategy (e.g. disable 4bit/offload) "
                        "and inspect `result['prediction_masks']`."
                    )

            if args.submit:
                async_func(executor, mask_save, item=item, mask_prediction=mask_prediction, work_dir=work_dir)
                encoded_mask = None
            else:
                encoded_mask = mask_to_rle(mask_prediction)

            result = {
                'index': item['index'],
                'video_id': item['video_id'],
                'exp_id': item['exp_id'],
                'text_prediction': text_prediction,
                'frames': item['frames'],
                'exp': item['text_prompt'],
                'prediction_masks': encoded_mask,
            }
            results.append(result)
            done_keys.add(key)
            processed_since_save += 1

            # Periodic partial save (rank-local).
            if processed_since_save >= save_every:
                _atomic_json_dump(results, partial_path)
                processed_since_save = 0
                last_save_t = time.time()
                if rank == 0:
                    print(f"[EVAL] Saved partial ({len(results)} items) to {partial_path}")

            # Optional early stop for debugging.
            if args.max_samples is not None and len(results) >= args.max_samples:
                break
    finally:
        # Always flush partial results if we made progress since last save.
        if processed_since_save > 0:
            try:
                _atomic_json_dump(results, partial_path)
                if rank == 0:
                    print(f"[EVAL] Saved partial ({len(results)} items) to {partial_path}")
            except Exception:
                pass
    executor.shutdown(wait=True)

    print(f'[Rank {rank}] : Finished.')
    
    if not args.submit:
        results = collect_results_cpu(results, len(dataset))
        if get_rank() == 0:
            final_results = {}
            for item in results:
                vid_id = item['video_id']
                exp_id = item['exp_id']
                if vid_id not in final_results:
                    final_results[vid_id] = {}
                assert exp_id not in final_results[vid_id]
                final_results[vid_id][exp_id] = item
            work_dir = os.path.join(work_dir, args.dataset)
            os.makedirs(work_dir, exist_ok=True)
            json.dump(final_results, open(f'{work_dir}/results.json', 'w'))
            # Also drop a "done" marker and keep partial files for possible debugging.
            with open(os.path.join(work_dir, "_eval_done.txt"), "w") as f:
                f.write("done\n")

    if rank == 0:
        print('Done')
