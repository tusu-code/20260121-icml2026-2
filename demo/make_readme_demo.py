#!/usr/bin/env python3
"""
Generate README-friendly demo assets (original GIF + segmentation overlay GIF + markdown snippet)
for Main-style video grounding/segmentation.

This is meant to help you build an *anonymous submission* repo demo like:
https://github.com/eliot127825-rgb/20260120_icml2026

Example:
  PYTHONPATH=. python tools/demo/make_readme_demo.py \
    --video /path/to/demo.mp4 \
    --model_path /path/to/hf_model_or_hf_repo_id \
    --text "Please segment the main character." \
    --out_dir ./demo_assets/demo1 \
    --frame_interval 6 --gif_fps 8
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import cv2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor


@dataclass
class DemoOutputs:
    prediction: str
    has_seg: bool
    demo_gif: Path
    demo_seg_gif: Optional[Path]
    demo_md: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True, help="Input mp4 (or any video readable by OpenCV).")
    p.add_argument("--model_path", type=str, default="AnonymousOrg/Main-8B", help="HF repo id or local HF dir.")
    p.add_argument("--text", type=str, required=True, help="Instruction / question. '<image>' will be auto-prepended.")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory (will be created).")
    p.add_argument("--frame_interval", type=int, default=6, help="Sample every N frames from the input video.")
    p.add_argument("--max_frames", type=int, default=120, help="Max sampled frames to keep (to avoid huge GIFs).")
    p.add_argument("--gif_fps", type=int, default=8, help="GIF fps.")
    p.add_argument("--max_size", type=int, default=512, help="Resize max(H,W) for GIFs (speed/size).")
    p.add_argument(
        "--max_segs",
        type=int,
        default=1,
        help="Max number of [SEG] masks to render in overlay. Use 0 to render ALL masks.",
    )
    p.add_argument(
        "--mask_select",
        type=str,
        default="best_single",
        choices=["first", "best_single"],
        help="How to select which mask(s) to render when model outputs multiple [SEG] masks.",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    return p.parse_args()


def read_video_frames(video_path: str, frame_interval: int, max_frames: int) -> Tuple[List[Image.Image], List[np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    pil_frames: List[Image.Image] = []
    np_frames_rgb: List[np.ndarray] = []
    idx = 0
    kept = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % frame_interval == 0:
            frame_rgb = frame_bgr[:, :, ::-1]
            np_frames_rgb.append(frame_rgb)
            pil_frames.append(Image.fromarray(frame_rgb).convert("RGB"))
            kept += 1
            if kept >= max_frames:
                break
        idx += 1
    cap.release()
    if not pil_frames:
        raise ValueError(f"No frames sampled from: {video_path}")
    return pil_frames, np_frames_rgb


def resize_pil(im: Image.Image, max_size: int) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= max_size:
        return im
    scale = max_size / float(m)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return im.resize((nw, nh), resample=Image.BICUBIC)


def save_gif(frames: List[Image.Image], out_path: Path, fps: int, max_size: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames2 = [resize_pil(f, max_size=max_size) for f in frames]
    duration_ms = int(round(1000.0 / max(1, fps)))
    frames2[0].save(
        out_path,
        save_all=True,
        append_images=frames2[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def overlay_masks(
    frames_rgb: List[np.ndarray],
    masks_list: List[np.ndarray],
    alpha: float = 0.5,
) -> List[Image.Image]:
    """
    frames_rgb: list of (H,W,3) uint8
    masks_list: list of (T,H,W) uint8/bool (one per [SEG] target)
    """
    # deterministic color palette
    colors: List[Tuple[int, int, int]] = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 165, 0),
        (128, 0, 128),
    ]
    out: List[Image.Image] = []
    t = len(frames_rgb)
    for i in range(t):
        base = frames_rgb[i].astype(np.float32)
        for j, masks in enumerate(masks_list):
            if i >= masks.shape[0]:
                continue
            m = masks[i]
            if m.dtype != np.bool_:
                m = m.astype(bool)
            if not m.any():
                continue
            color = np.array(colors[j % len(colors)], dtype=np.float32)
            base[m] = base[m] * (1.0 - alpha) + color * alpha
        out.append(Image.fromarray(base.astype(np.uint8)).convert("RGB"))
    return out


def _mask_connected_components_mean(mask_t: np.ndarray) -> float:
    # mask_t: (T,H,W) bool/uint8
    comps = []
    for t in range(mask_t.shape[0]):
        m = (mask_t[t].astype(np.uint8) > 0).astype(np.uint8)
        n, _ = cv2.connectedComponents(m)
        comps.append(max(0, n - 1))
    return float(np.mean(comps)) if comps else 0.0


def _mask_area_mean(mask_t: np.ndarray) -> float:
    T, H, W = mask_t.shape
    areas = []
    for t in range(T):
        m = (mask_t[t].astype(np.uint8) > 0).astype(np.uint8)
        areas.append(float(m.sum() / (H * W + 1e-9)))
    return float(np.mean(areas)) if areas else 0.0


def select_masks_for_overlay(masks_list: list, mask_select: str, max_segs: int) -> list:
    """
    masks_list: list of (T,H,W) masks.
    - mask_select='first': keep original order
    - mask_select='best_single': pick the mask with minimal connected-components (more \"single object\"), then smaller area
    max_segs:
      - 0 => keep all
      - >0 => keep at most N
    """
    if not masks_list:
        return masks_list
    if mask_select == "best_single":
        scored = []
        for i, m in enumerate(masks_list):
            mt = np.asarray(m)
            cc = _mask_connected_components_mean(mt)
            area = _mask_area_mean(mt)
            scored.append((cc, area, i))
        scored.sort()
        best_i = scored[0][2]
        masks_list = [masks_list[best_i]]
    # apply max_segs
    if isinstance(max_segs, int) and max_segs > 0:
        masks_list = masks_list[:max_segs]
    return masks_list


def load_model_and_tokenizer(model_path: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    ).eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    if "upstreamllm" in model_path.lower():
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = None
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer, processor


def run_demo(
    video_path: str,
    model_path: str,
    text: str,
    out_dir: str,
    frame_interval: int,
    max_frames: int,
    gif_fps: int,
    max_size: int,
    max_segs: int,
    mask_select: str,
    device: str,
) -> DemoOutputs:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    pil_frames, frames_rgb = read_video_frames(video_path, frame_interval=frame_interval, max_frames=max_frames)
    demo_gif = outp / "demo.gif"
    save_gif(pil_frames, demo_gif, fps=gif_fps, max_size=max_size)

    model, tokenizer, processor = load_model_and_tokenizer(model_path, device=device)

    prompt = text.strip()
    if "<image>" not in prompt:
        prompt = "<image>" + prompt

    # `predict_forward` is provided by Main HF models with trust_remote_code=True
    result = model.predict_forward(  # type: ignore[attr-defined]
        video=pil_frames,
        text=prompt,
        tokenizer=tokenizer,
        processor=processor,
    )
    prediction = str(result.get("prediction", "")).strip()
    # Make README output cleaner
    prediction = (
        prediction.replace("<|im_end|>", "")
        .replace("<|end|>", "")
        .replace("<s>", "")
        .strip()
    )

    demo_seg_gif: Optional[Path] = None
    has_seg = bool(result.get("prediction_masks")) and ("[SEG]" in prediction)
    if has_seg:
        masks_list = result["prediction_masks"]  # list[(T,H,W)]
        masks_list = select_masks_for_overlay(masks_list, mask_select=mask_select, max_segs=max_segs)
        seg_frames = overlay_masks(frames_rgb, masks_list)
        demo_seg_gif = outp / "demo_seg.gif"
        save_gif(seg_frames, demo_seg_gif, fps=gif_fps, max_size=max_size)

    # README snippet (keep it simple and anonymous-friendly)
    demo_md = outp / "README_DEMO.md"
    md = []
    md.append("## 🎬 Demo\n")
    md.append(f"**Instruction / Question:** {text.strip()}\n")
    md.append("\n---\n")
    md.append("**Model Output:**\n\n")  # keep markdown render friendly
    md.append("```\n")
    md.append(prediction + "\n")
    md.append("```\n")
    md.append("\n---\n")
    md.append("| **Original Video** | **Segmentation Overlay** |\n")
    md.append("| --- | --- |\n")
    if demo_seg_gif is not None:
        md.append("| ![](demo.gif) | ![](demo_seg.gif) |\n")
    else:
        md.append("| ![](demo.gif) | (no `[SEG]` output) |\n")

    demo_md.write_text("".join(md), encoding="utf-8")

    return DemoOutputs(
        prediction=prediction,
        has_seg=has_seg,
        demo_gif=demo_gif,
        demo_seg_gif=demo_seg_gif,
        demo_md=demo_md,
    )


def main() -> None:
    args = parse_args()
    # clean output dir if it exists (avoid mixing assets)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        # only remove common generated files; keep user files
        for name in ["demo.gif", "demo_seg.gif", "README_DEMO.md"]:
            p = out_dir / name
            if p.exists():
                p.unlink()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    _ = run_demo(
        video_path=args.video,
        model_path=args.model_path,
        text=args.text,
        out_dir=args.out_dir,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
        gif_fps=args.gif_fps,
        max_size=args.max_size,
        max_segs=args.max_segs,
        mask_select=args.mask_select,
        device=args.device,
    )
    print(f"[OK] wrote: {out_dir / 'demo.gif'}")
    if (out_dir / "demo_seg.gif").exists():
        print(f"[OK] wrote: {out_dir / 'demo_seg.gif'}")
    print(f"[OK] wrote: {out_dir / 'README_DEMO.md'}")


if __name__ == "__main__":
    main()


