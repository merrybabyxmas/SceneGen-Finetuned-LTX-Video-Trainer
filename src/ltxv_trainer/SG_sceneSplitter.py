"""
Scene Splitter (SBD-integrated):
- Metadata-driven pipeline that reads a metadata.json of the form
  [ {"caption": caption1, "media_path": path1}, ... ]
- For each video, apply **Scene Boundary Detection (SBD)** using a pretrained ResNet50 encoder
  (torchvision ImageNet weights). The algorithm:
    1) Read frames via OpenCV (RGB) -> float tensor in [0,1].
    2) Resize to target size (default 224x224) and ImageNet-normalize.
    3) Extract features via ResNet50 (fc removed). (T, 2048)
    4) Compute adjacent-frame cosine similarity and mark boundary when sim < threshold.
    5) Convert boundaries to shot ranges and **save shot video clips**.
- Produce `split_meta.json` that repeats the original caption for each shot, with file paths & frame ranges.

Notes:
- Batch sizes for preprocessing/encoding are configurable. Long videos are processed in chunks.
- Saving is done with OpenCV VideoWriter to MP4 (mp4v). Falls back to AVI if needed.
- You can still use the old tensor-based splitter, but the default `__main__` uses SBD + metadata.
"""
from __future__ import annotations

import os
import json
import glob
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from SG_configs import PipelineConfig, SBDConfig, IOConfig
from SG_sbd import SceneBoundaryDetector


# =========================
# SBD Core
# =========================


# =========================
# Shot saving & metadata
# =========================

def save_shots_from_video(video_file: str, out_dir: str, boundaries_end: List[int],
                          split_ext: str = "mp4", default_fps: int = 24) -> Tuple[List[Tuple[int,int]], List[str]]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = float(default_fps)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*('m','p','4','v')) if split_ext.lower()=="mp4" else cv2.VideoWriter_fourcc(*('X','V','I','D'))

    # Build shot ranges from boundaries
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    boundaries_end = sorted([b for b in boundaries_end if 0 <= b < T])
    if not boundaries_end or boundaries_end[-1] != T-1:
        boundaries_end.append(T-1)
    ranges: List[Tuple[int,int]] = []
    start = 0
    for end in boundaries_end:
        ranges.append((start, end))
        start = end + 1

    shot_paths: List[str] = []
    base = Path(video_file).stem
    shot_idx = 0

    # Write frames sequentially to each writer
    current_writer = None
    current_range = ranges[0]
    current_path = str(Path(out_dir) / f"{base}_shot{shot_idx}.{split_ext}")
    current_writer = cv2.VideoWriter(current_path, fourcc, fps, (W, H))
    shot_paths.append(current_path)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # write current frame
        current_writer.write(frame)
        # if we reached end of current shot, close and open next
        if frame_id == current_range[1]:
            current_writer.release()
            shot_idx += 1
            if shot_idx < len(ranges):
                current_range = ranges[shot_idx]
                current_path = str(Path(out_dir) / f"{base}_shot{shot_idx}.{split_ext}")
                current_writer = cv2.VideoWriter(current_path, fourcc, fps, (W, H))
                shot_paths.append(current_path)
        frame_id += 1

    cap.release()
    # in case of early break
    if current_writer is not None:
        current_writer.release()

    return ranges, shot_paths

# =========================
# Metadata-driven pipeline
# =========================

def process_metadata(metadata_json: str, cfg: PipelineConfig) -> Dict[str, Any]:
    with open(metadata_json, 'r', encoding='utf-8') as f:
        items: List[Dict[str, Any]] = json.load(f)

    sbd = SceneBoundaryDetector(cfg.sbd)
    split_records: List[Dict[str, Any]] = []

    for it in items:
        caption = it.get('caption', '')
        media_path = it.get('media_path')
        if not media_path:
            print('[WARN] Missing media_path, skipping item')
            continue
        if not os.path.exists(media_path):
            print(f'[WARN] Not found: {media_path}, skipping')
            continue

        # 1) Detect boundaries (SBD)
        boundaries_end, labels, meta = sbd.run(media_path)

        # 2) Optional keyframe visualization (SOS/EOS)
        if cfg.sbd.visualize_keyframes:
            save_keyframes(labels, media_path, cfg.io.keyframes_root)

        # 3) Save shots
        video_out_dir = str(Path(cfg.io.splits_root) / Path(media_path).stem)
        ranges, shot_paths = save_shots_from_video(
            media_path, video_out_dir, boundaries_end,
            split_ext=cfg.io.split_ext, default_fps=cfg.io.default_fps
        )

        # 4) Append split metadata (caption repeated)
        for si, (s, e) in enumerate(ranges):
            split_records.append({
                'orig_media_path': media_path,
                'caption': caption,
                'shot_index': si,
                'start_frame': int(s),
                'end_frame': int(e),
                'shot_path': shot_paths[si],
                'fps': float(meta.get('fps', cfg.io.default_fps)),
                'H': int(meta.get('H', -1)),
                'W': int(meta.get('W', -1)),
                'T': int(meta.get('T', -1)),
            })

    # Write split_meta.json at root
    os.makedirs(cfg.io.splits_root, exist_ok=True)
    out_meta_path = str(Path(cfg.io.splits_root) / 'split_meta.json')
    with open(out_meta_path, 'w', encoding='utf-8') as f:
        json.dump(split_records, f, ensure_ascii=False, indent=2)

    return {'out_meta': out_meta_path, 'num_items': len(items), 'num_splits': len(split_records)}

# =========================
# Utilities
# =========================

def save_keyframes(labels: Dict[int, str], media_path: str, keyframes_root: str) -> None:
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video for keyframes: {media_path}")
        return
    base_dir = Path(keyframes_root) / Path(media_path).stem
    os.makedirs(base_dir, exist_ok=True)

    # Iterate and dump labeled frames at original resolution
    idx_to_dump = sorted(labels.keys())
    frame_id = 0
    next_idx_ptr = 0
    next_target = idx_to_dump[next_idx_ptr] if idx_to_dump else None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if next_target is not None and frame_id == next_target:
            label_text = labels[next_target]
            # Save RGB JPEG
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if "[SOS]" in label_text:
                out = base_dir / f"SOS_{frame_id}.jpg"
                cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if "[EOS]" in label_text:
                out = base_dir / f"EOS_{frame_id}.jpg"
                cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            next_idx_ptr += 1
            next_target = idx_to_dump[next_idx_ptr] if next_idx_ptr < len(idx_to_dump) else None
        frame_id += 1
    cap.release()

# (Optional) helper to quickly make a tiny demo video if needed

def _make_dummy_video_mp4(path: str, T: int = 90, H: int = 128, W: int = 128, fps: int = 12) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*('m','p','4','v'))
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for t in range(T):
        if t < T//3:
            base = np.array([255, 60, 60], dtype=np.uint8)
        elif t < 2*T//3:
            base = np.array([60, 255, 60], dtype=np.uint8)
        else:
            base = np.array([60, 60, 255], dtype=np.uint8)
        frame = np.tile(base, (H, W, 1))
        # add a moving square to create intra-shot motion
        sz = 20
        x = (t*3) % (W-sz)
        y = (t*2) % (H-sz)
        frame[y:y+sz, x:x+sz, :] = 255 - frame[y:y+sz, x:x+sz, :]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

# =========================
# Main
# =========================

if __name__ == '__main__':
    # Example metadata.json generation (only if not exists)
    demo_root = Path('./_demo')
    demo_root.mkdir(parents=True, exist_ok=True)
    demo_video = str(demo_root / 'demo.mp4')
    if not os.path.exists(demo_video):
        _make_dummy_video_mp4(demo_video)

    meta_path = str(demo_root / 'metadata.json')
    if not os.path.exists(meta_path):
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump([
                {"caption": "a toy demo scene with abrupt changes", "media_path": demo_video}
            ], f, ensure_ascii=False, indent=2)

    # Build pipeline config
    cfg = PipelineConfig(
        sbd=SBDConfig(
            preproc_batch_size=128,
            enc_batch_size=128,
            similarity_threshold=0.75,
            target_h=224,
            target_w=224,
            use_gpu=True,
            visualize_keyframes=True,
        ),
        io=IOConfig(
            splits_root='./splits',
            keyframes_root='./keyframes',
            split_ext='mp4',
            default_fps=24,
        )
    )

    summary = process_metadata(meta_path, cfg)
