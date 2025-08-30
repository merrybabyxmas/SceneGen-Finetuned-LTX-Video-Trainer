#!/usr/bin/env python3
import os
import glob
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List


class SceneBoundaryDetector:
    def __init__(self, video_file, output_dir,
                 preproc_batch_size=128,
                 enc_batch_size=128,
                 similarity_threshold=0.8,
                 target_size=(224, 224),
                 dataset_name=None,
                 video_name=None,
                resolution_bucket=(768, 448)):
        self.video_file = video_file
        self.output_dir = Path(output_dir)
        self.preproc_batch_size = preproc_batch_size
        self.enc_batch_size = enc_batch_size
        self.similarity_threshold = similarity_threshold
        self.target_size = target_size
        self.dataset_name = dataset_name
        self.video_name = video_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")
        self.bucket_w, self.bucket_h = resolution_bucket
        self.resolution_bucket = (self.bucket_h, self.bucket_w)

    def load_video_frames(self):
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {self.video_file}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        frames_np = np.array(frames)  # (T, H, W, C)
        frames_np = np.transpose(frames_np, (0, 3, 1, 2))  # (T, C, H, W)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0
        return frames_tensor  # (T, C, H, W)

    def preprocess_frames(self, frames_tensor):
        T = frames_tensor.shape[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        out_batches = []
        for i in range(0, T, self.preproc_batch_size):
            batch = frames_tensor[i:i+self.preproc_batch_size].to(self.device)
            batch_resized = F.interpolate(batch, size=self.target_size, mode='bilinear', align_corners=False)
            batch_norm = (batch_resized - mean) / std
            out_batches.append(batch_norm)
        return torch.cat(out_batches, dim=0)  # (T, 3, H, W)

    def extract_features(self, frames_norm):
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        encoder = nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        encoder.eval()
        T = frames_norm.shape[0]
        feats = []
        with torch.no_grad():
            for i in range(0, T, self.enc_batch_size):
                batch = frames_norm[i:i+self.enc_batch_size]
                out = encoder(batch)  # (B, 2048, 1, 1)
                feats.append(out.view(out.size(0), -1))
        return torch.cat(feats, dim=0)  # (T, 2048)

    def detect_boundaries(self, features):
        feats_norm = F.normalize(features, p=2, dim=1)
        sims = (feats_norm[:-1] * feats_norm[1:]).sum(dim=1)
        mask = sims < self.similarity_threshold
        boundaries = [i for i, b in enumerate(mask.tolist()) if b]
        return boundaries  # EOS frame indices

    def save_split_videos(self, frames_tensor, boundaries, fps, meta_list):
        video_id = self.video_name
        out_dir = self.output_dir / "splits"
        out_dir.mkdir(parents=True, exist_ok=True)

        shot_indices = [0] + [b+1 for b in boundaries] + [frames_tensor.shape[0]]

        if len(shot_indices) - 1 <= 1:
            print(f"[SKIP] {video_id}: only 1 shot detected, skipping...")
            return

        for i in range(len(shot_indices)-1):
            start, end = shot_indices[i], shot_indices[i+1]

            if end - start <= 5:
                print(f"[SKIP] {video_id}_shot{i}: too short ({end-start} frames)")
                continue

            shot_frames = frames_tensor[start:end]  # (F, C, H, W)

            # --- 해상도만 bucket 크기로 통일 ---
            shot_frames_resized = F.interpolate(
                shot_frames,
                size=(self.bucket_h, self.bucket_w),  # e.g. (448, 768)
                mode="bilinear",
                align_corners=False
            )  # (F, C, H, W)

            shot_path = out_dir / f"{video_id}_shot{i}.mp4"
            print(f"shot frame resized : {shot_frames_resized.shape}")

            # save mp4
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            H, W = self.bucket_h, self.bucket_w
            writer = cv2.VideoWriter(str(shot_path), fourcc, fps, (W, H))
            for f in range(shot_frames_resized.shape[0]):
                frame = (shot_frames_resized[f].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            writer.release()

            meta_list.append({
                "orig_media_path": str(self.video_file),
                "caption": self.video_name,
                "shot_index": i,
                "start_frame": start,
                "end_frame": end-1,
                "shot_path": Path(shot_path).name,
                "fps": fps,
                "H": self.bucket_h,
                "W": self.bucket_w,
                "T": shot_frames.shape[0]   # 프레임 수는 원본 그대로
            })
    def run(self):
        # --- load frames ---
        cap = cv2.VideoCapture(self.video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0 or np.isnan(fps):
            fps = 24.0   # fallback

        frames = self.load_video_frames()
        frames_norm = self.preprocess_frames(frames)
        feats = self.extract_features(frames_norm)
        boundaries = self.detect_boundaries(feats)

        meta_list = []
        self.save_split_videos(frames, boundaries, fps, meta_list)
        return meta_list

def process_dataset(video_dataset_dir, output_dir, original_meta_path=None):
    dataset_name = Path(video_dataset_dir).name
    video_files = glob.glob(os.path.join(video_dataset_dir, "*.mp4"))
    all_meta = []
    caption_map = {}
    if original_meta_path:
        with open(original_meta_path, "r", encoding="utf-8") as f:
            orig_meta = json.load(f)
        for item in orig_meta:
            # media_path = "video0_137.72_149.44.mp4" 같은 형식
            key = Path(item["media_path"]).stem
            caption_map[key] = item["caption"]

    for vf in video_files:
        vid_name = Path(vf).stem
        detector = SceneBoundaryDetector(
            vf, output_dir,
            dataset_name=dataset_name,
            video_name=vid_name,
            resolution_bucket=(768, 448)
        )
        try:
            meta = detector.run()
            # --- caption 교체 ---
            for m in meta:
                if vid_name in caption_map:
                    # print(caption_map[vid_name])
                    m["caption"] = caption_map[vid_name]
            all_meta.extend(meta)
        except Exception as e:
            print(f"Error processing {vf}: {e}")

    # save split_meta.json
    out_path = Path(output_dir) / "splits" / "split_meta.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
    print(f"Saved split metadata to {out_path}")


    # save split_meta.json
    out_path = Path(output_dir) / "splits" / "split_meta.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
    print(f"Saved split metadata to {out_path}")


if __name__ == "__main__":
    video_dataset_dir = "dongwoo/datasets/videos/videos"
    output_dir = "datasets/videos"
    original_meta_path = "dongwoo/datasets/videos/dataset.json"

    process_dataset(video_dataset_dir, output_dir, original_meta_path)
