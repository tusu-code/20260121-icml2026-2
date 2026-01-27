###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import os.path as osp
import sys

# Ensure repo root is on sys.path so `third_parts.*` imports work even without PYTHONPATH.
# IMPORTANT: must run BEFORE importing `third_parts.*`.
_THIS_DIR = osp.dirname(__file__)
_REPO_ROOT = osp.abspath(osp.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import time
import argparse
import json
import numpy as np
from pycocotools import mask as cocomask
from third_parts.revos.utils.metircs import db_eval_iou, db_eval_boundary
import multiprocessing as mp
import queue as pyqueue

# Default 128 workers is too aggressive and `q.empty()` is racy in multiprocessing.
# Allow overriding via env var: MEVIS_EVAL_WORKERS=16
NUM_WOEKERS = int(os.environ.get("MEVIS_EVAL_WORKERS", "16"))


def eval_queue(q, rank, out_dict):
    # NOTE: do NOT use `while not q.empty()` in multiprocessing; it's racy and can deadlock.
    # Use non-blocking get and exit on Empty.
    while True:
        try:
            vid_name, exp = q.get_nowait()
        except pyqueue.Empty:
            break

        vid = exp_dict[vid_name]

        exp_name = f'{vid_name}_{exp}'

        # Skip missing predictions (e.g. partial eval / resume)
        try:
            pred = mask_pred_dict[vid_name][exp]
        except KeyError:
            continue

        h, w = pred['prediction_masks'][0]['size']
        vid_len = len(vid['frames'])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid['expressions'][exp]['anno_id']

        for frame_idx, frame_name in enumerate(vid['frames']):
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)

            pred_mask = cocomask.decode(pred['prediction_masks'][frame_idx])
            pred_masks[frame_idx] += pred_mask

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str, )
    parser.add_argument("--mevis_exp_path", type=str,
                        default="./data/video_datas/mevis/valid_u/meta_expressions.json")
    parser.add_argument("--mevis_mask_path", type=str,
                        default="./data/video_datas/mevis/valid_u/mask_dict.json")
    parser.add_argument("--save_name", type=str, default="mevis_valu.json")
    args = parser.parse_args()
    queue = mp.Queue()
    exp_dict = json.load(open(args.mevis_exp_path))['videos']
    mask_dict = json.load(open(args.mevis_mask_path))

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    output_dict = mp.Manager().dict()

    mask_pred = json.load(open(args.pred_path))
    mask_pred_dict  = mp.Manager().dict(mask_pred)

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])

    start_time = time.time()
    if NUM_WOEKERS > 1:
        processes = []
        for rank in range(NUM_WOEKERS):
            p = mp.Process(target=eval_queue, args=(queue, rank, output_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        eval_queue(queue, 0, output_dict)

    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]

    output_path = osp.join(osp.dirname(args.pred_path), args.save_name)
    results = {
        'J': round(100 * float(np.mean(j)), 2),
        'F': round(100 * float(np.mean(f)), 2),
        'J&F': round(100 * float((np.mean(j) + np.mean(f)) / 2), 2),
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % (total_time))
