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


# =========================
# SBD Core
# =========================
class SceneBoundaryDetector:
    def __init__(self, cfg: SBDConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu")

        # Pretrained ResNet50 encoder (fc removed)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]).to(self.device).eval()

        # Precompute normalization tensors
        self.imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    # ---------- IO ----------
    def _read_video_frames(self, video_file: str) -> Tuple[List[np.ndarray], int, int, int, float]:
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_file}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-3:
            fps = 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames: List[np.ndarray] = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames, height, width, len(frames), float(fps)

    # ---------- Preprocess ----------
    def _preprocess(self, frames_rgb: List[np.ndarray]) -> torch.Tensor:
        T = len(frames_rgb)
        th, tw = self.cfg.target_h, self.cfg.target_w
        out_list: List[torch.Tensor] = []
        bs = self.cfg.preproc_batch_size
        for i in range(0, T, bs):
            batch = frames_rgb[i:i+bs]
            # (B, H, W, C) -> (B, C, H, W), float in [0,1]
            arr = np.stack(batch, axis=0).astype(np.float32) / 255.0
            arr = np.transpose(arr, (0, 3, 1, 2))
            t = torch.from_numpy(arr).to(self.device)
            t = F.interpolate(t, size=(th, tw), mode='bilinear', align_corners=False)
            t = (t - self.imnet_mean) / self.imnet_std
            out_list.append(t)
        return torch.cat(out_list, dim=0)  # (T, 3, th, tw)

    # ---------- Features ----------
    @torch.no_grad()
    def _encode(self, frames_norm: torch.Tensor) -> torch.Tensor:
        T = frames_norm.shape[0]
        feats_list: List[torch.Tensor] = []
        bs = self.cfg.enc_batch_size
        for i in range(0, T, bs):
            feat = self.encoder(frames_norm[i:i+bs])  # (b, 2048, 1, 1)
            feats_list.append(feat.view(feat.size(0), -1))
        feats = torch.cat(feats_list, dim=0)  # (T, 2048)
        return feats

    # ---------- Boundaries ----------
    def _detect_boundaries_from_feats(self, feats: torch.Tensor) -> Tuple[List[int], Dict[int, str]]:
        """
        Return:
          - boundaries_end: sorted list of frame indices that are **end of a shot** (EOS)
          - labels: mapping {idx: "[EOS]" or "[SOS]" or "[EOS][SOS]"}
        """
        thr = float(self.cfg.similarity_threshold)
        f = F.normalize(feats, p=2, dim=1)  # (T, D)
        if f.shape[0] < 2:
            return [f.shape[0]-1], {f.shape[0]-1: "[EOS]"}
        sims = (f[:-1] * f[1:]).sum(dim=1)  # (T-1,)
        mask = sims < thr
        labels: Dict[int, str] = {}
        boundaries_end: List[int] = []
        for i, is_b in enumerate(mask.tolist()):
            if is_b:
                boundaries_end.append(i)        # i is EOS
                labels[i] = labels.get(i, "") + "[EOS]"
                labels[i+1] = labels.get(i+1, "") + "[SOS]"
        # ensure last frame closes a shot
        last = f.shape[0] - 1
        if (not boundaries_end) or (boundaries_end[-1] != last):
            boundaries_end.append(last)
            labels[last] = labels.get(last, "") + "[EOS]"
        # dedupe/sort
        boundaries_end = sorted(set(boundaries_end))
        return boundaries_end, labels

    # ---------- Public API ----------
    def run(self, video_file: str) -> Tuple[List[int], Dict[int, str], Dict[str, Any]]:
        frames_rgb, H, W, T, fps = self._read_video_frames(video_file)
        frames_norm = self._preprocess(frames_rgb)
        feats = self._encode(frames_norm)
        boundaries_end, labels = self._detect_boundaries_from_feats(feats)
        meta = {"T": T, "H": H, "W": W, "fps": fps}
        return boundaries_end, labels, meta