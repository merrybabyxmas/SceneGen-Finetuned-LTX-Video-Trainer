"""
Scene Splitter: split a long scene video tensor into shot clips.

- Input: a torch tensor saved on disk at `video_path` with shape (F, C, H, W) or (B, F, C, H, W)
         with pixel range either [0, 1] or [0, 255].
- Method: compute frame features with a configurable feature extractor (including **pretrained torchvision models**),
          compare current and previous features via cosine similarity, and mark shot boundaries when
          the similarity condition crosses a threshold.
- Output: shot video files written to `cfg.io.save_dir` and a JSON sidecar with index ranges.

Pretrained option:
  - Set `cfg.extractor.name = "tv-cnn"` and choose `cfg.extractor.tv_backbone` from
    {"resnet18", "resnet50", "mobilenet_v3_large"}. Weights are torchvision's ImageNet pretrained weights.

This file is self-contained: config, models, and a small demo in `if __name__ == "__main__"`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

# ----------------------------
# Configs
# ----------------------------

TransitionRule = Literal["greater", "less"]


@dataclass
class IOConfig:
    save_dir: str = "./shots_out"
    fps: int = 24
    basename: Optional[str] = None  # if None, derive from input file name
    out_ext: Literal["mp4", "gif"] = "mp4"  # falls back to gif if ffmpeg missing


@dataclass
class ExtractorConfig:
    # which extractor
    name: Literal["avgpool-proj", "lite-cnn", "tv-cnn"] = "avgpool-proj"
    out_dim: int = 256  # feature dim D (if projection enabled)
    stride: int = 1  # temporal stride while extracting features (>=1)
    normalize_input: bool = True  # map to [0,1]
    # chunk long sequences to save memory (per forward frames)
    max_frames_per_chunk: int = 256

    # Torchvision backbone options (used when name=="tv-cnn")
    tv_backbone: Literal["resnet18", "resnet50", "mobilenet_v3_large"] = "resnet18"
    tv_use_projection: bool = True  # map native feat dim -> out_dim via Linear


@dataclass
class SplitterConfig:
    prev_offset: int = 1  # compare frame t with frame t - prev_offset
    threshold: float = 0.6  # cosine similarity threshold
    rule: TransitionRule = "less"  # ">" means boundary if sim > thr, "less" for sim < thr
    min_shot_len: int = 6  # enforce minimum frames per shot
    pad_last_shot: bool = True  # ensure last frame closes a shot


@dataclass
class SceneSplitConfig:
    device: Literal["cpu", "cuda"] = "cpu"
    extractor: ExtractorConfig = ExtractorConfig()
    splitter: SplitterConfig = SplitterConfig()
    io: IOConfig = IOConfig()

    @staticmethod
    def demo() -> "SceneSplitConfig":
        return SceneSplitConfig()


# ----------------------------
# Feature extractors
# ----------------------------

class BaseFeatureExtractor(nn.Module):
    def __init__(self, cfg: ExtractorConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F, C, H, W) -> (B, F, D)"""
        raise NotImplementedError


class AvgPoolProj(BaseFeatureExtractor):
    """Very light feature extractor: global average over H,W + linear projection.
    No external downloads required.
    """

    def __init__(self, cfg: ExtractorConfig):
        super().__init__(cfg)
        self.proj = nn.Linear(3, cfg.out_dim)  # after mean over H,W, channels remain 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Flen, C, H, W = x.shape
        # Normalize to [0,1]
        if self.cfg.normalize_input:
            x = x.float()
            if x.max() > 1.0:
                x = x / 255.0
        # (B,F,C,H,W) -> (B,F,C)
        x = x.mean(dim=(-1, -2))
        # Per-channel projection -> (B,F,D)
        x = self.proj(x)
        return x


class LiteCNN(BaseFeatureExtractor):
    """A tiny CNN followed by global average pooling and projection.
    Avoids external pretrained weights.
    """

    def __init__(self, cfg: ExtractorConfig):
        super().__init__(cfg)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, cfg.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Flen, C, H, W = x.shape
        # Normalize to [0,1]
        if self.cfg.normalize_input:
            x = x.float()
            if x.max() > 1.0:
                x = x / 255.0
        x = x.reshape(B * Flen, C, H, W)
        x = self.backbone(x).flatten(1)
        x = self.proj(x)
        x = x.reshape(B, Flen, -1)
        return x


class TorchvisionCNN(BaseFeatureExtractor):
    """Feature extractor that uses a torchvision pretrained backbone.
    Supported backbones: resnet18/resnet50/mobilenet_v3_large.
    Outputs per-frame features of native dim (e.g., 512) or projected to cfg.out_dim.
    """

    def __init__(self, cfg: ExtractorConfig):
        super().__init__(cfg)
        try:
            import torchvision.models as tm
            from torchvision.models import (
                ResNet18_Weights,
                ResNet50_Weights,
                MobileNet_V3_Large_Weights,
            )
        except Exception as e:
            raise ImportError(
                "torchvision is required for tv-cnn. Install via `pip install torchvision`"
            ) from e

        # Select backbone + weights
        if cfg.tv_backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            backbone = tm.resnet18(weights=weights)
            feat_dim = backbone.fc.in_features
            modules = list(backbone.children())[:-1]  # -> (B, feat_dim, 1,1)
            self.backbone = nn.Sequential(*modules)
        elif cfg.tv_backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            backbone = tm.resnet50(weights=weights)
            feat_dim = backbone.fc.in_features
            modules = list(backbone.children())[:-1]
            self.backbone = nn.Sequential(*modules)
        elif cfg.tv_backbone == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT
            backbone = tm.mobilenet_v3_large(weights=weights)
            feat_dim = backbone.classifier[0].in_features
            self.backbone = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        else:
            raise ValueError(f"Unknown tv_backbone: {cfg.tv_backbone}")

        self.feat_dim = feat_dim
        self.imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.use_proj = bool(cfg.tv_use_projection)
        self.proj = nn.Linear(self.feat_dim, cfg.out_dim) if self.use_proj else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Flen, C, H, W = x.shape
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = x.reshape(B * Flen, C, H, W)
        # Resize to 224x224 if needed
        if min(H, W) < 224 or max(H, W) != 224:
            x = Fnn.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # ImageNet normalization
        x = (x - self.imnet_mean.to(x.device)) / self.imnet_std.to(x.device)
        feats = self.backbone(x).flatten(1)  # (B*F, feat_dim)
        feats = self.proj(feats)  # (B*F, D)
        feats = feats.reshape(B, Flen, -1)
        return feats


def build_extractor(cfg: ExtractorConfig) -> BaseFeatureExtractor:
    if cfg.name == "avgpool-proj":
        return AvgPoolProj(cfg)
    elif cfg.name == "lite-cnn":
        return LiteCNN(cfg)
    elif cfg.name == "tv-cnn":
        return TorchvisionCNN(cfg)
    else:
        raise ValueError(f"Unknown extractor: {cfg.name}")


# ----------------------------
# Scene Splitter
# ----------------------------

class SceneSplitter(nn.Module):
    def __init__(self, cfg: SceneSplitConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.extractor = build_extractor(cfg.extractor).to(self.device).eval()

    @torch.no_grad()
    def forward(self, video_path: Path | str, save: bool = True) -> dict:
        if video_path is None:
            raise FileNotFoundError("video path not provided!")
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video tensor not found: {video_path}")

        vid = self._load_video(video_path)  # (B,F,C,H,W)
        shot_indices = self._split(vid)  # boundaries as end indices (inclusive)
        meta = self._save_shots(vid, shot_indices, video_path, save=save)
        return meta

    # --------- IO ---------
    def _load_video(self, video_path: Path) -> torch.Tensor:
        vid = torch.load(video_path)
        if vid.ndim == 4:
            # (F,C,H,W) -> (1,F,C,H,W)
            vid = vid.unsqueeze(0)
        if vid.ndim != 5:
            raise ValueError("Expected tensor of shape (F,C,H,W) or (B,F,C,H,W)")
        return vid

    # --------- Core splitting ---------
    @torch.no_grad()
    def _extract_feats_chunked(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (1, Fs, C, H, W) -> (1, Fs, D) with chunking."""
        B, Fs, C, H, W = frames.shape
        cfg = self.cfg.extractor
        chunks: List[torch.Tensor] = []
        for start in range(0, Fs, cfg.max_frames_per_chunk):
            end = min(Fs, start + cfg.max_frames_per_chunk)
            feats = self.extractor(frames[:, start:end])  # (1, L, D)
            chunks.append(feats)
        return torch.cat(chunks, dim=1)

    @torch.no_grad()
    def _split(self, data: torch.Tensor) -> List[int]:
        """Return shot boundary indices (inclusive) for the FIRST video in the batch."""
        cfg = self.cfg
        assert data.ndim == 5, "Data must be (B,F,C,H,W)"
        B, Flen, C, H, W = data.shape
        if B != 1:
            data = data[:1]
        data = data.to(self.device)

        # Optional temporal stride
        stride = max(1, cfg.extractor.stride)
        frames = data[:, ::stride]  # (1, Fs, C, H, W)
        Fs = frames.shape[1]

        feats = self._extract_feats_chunked(frames)  # (1, Fs, D)
        feats = Fnn.normalize(feats, dim=-1)  # cosine-friendly

        k = max(1, cfg.splitter.prev_offset)
        sims: List[float] = []  # similarity between t and t-k for t=k..Fs-1
        for t in range(k, Fs):
            sim = (feats[:, t] * feats[:, t - k]).sum(dim=-1)  # (1,)
            sims.append(float(sim.item()))

        # Decide boundaries
        boundaries: List[int] = []  # store indices in original frame space (unstrided)
        thr = float(cfg.splitter.threshold)
        rule = cfg.splitter.rule
        min_len = max(1, cfg.splitter.min_shot_len)

        last_end = -1  # inclusive end of last shot in original indexing

        for i, sim in enumerate(sims, start=k):
            # frame index in strided space -> original index
            t_orig = i * stride
            cond = (sim > thr) if rule == "greater" else (sim < thr)
            if cond:
                # enforce minimum shot length
                if (t_orig - (last_end + 1)) >= min_len:
                    boundaries.append(t_orig - 1)  # end the previous shot at t-1
                    last_end = t_orig - 1

        # ensure last frame closes
        if cfg.splitter.pad_last_shot:
            if not boundaries or boundaries[-1] != (Flen - 1):
                boundaries.append(Flen - 1)

        # Remove duplicate/invalid boundaries
        boundaries = sorted(set(b for b in boundaries if 0 <= b < Flen))
        # Guarantee at least one boundary (the last frame)
        if not boundaries:
            boundaries = [Flen - 1]
        return boundaries

    # --------- Save outputs ---------
    def _save_shots(
        self, vid: torch.Tensor, idx: List[int], video_path: Path, save: bool = True
    ) -> dict:
        """Split (B,F,C,H,W) by boundary indices and save to disk.

        Returns metadata with shot ranges and file paths.
        """
        io = self.cfg.io
        save_dir = Path(io.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        B, Flen, C, H, W = vid.shape
        vid = vid[0]  # (F,C,H,W)

        base = video_path.stem if io.basename is None else io.basename

        # Build ranges from boundaries
        boundaries = sorted(idx)
        ranges: List[Tuple[int, int]] = []
        start = 0
        for end in boundaries:
            ranges.append((start, end))
            start = end + 1
        # Drop invalid ranges
        ranges = [(s, e) for (s, e) in ranges if s <= e and 0 <= s < Flen and 0 <= e < Flen]

        shot_files: List[str] = []
        for si, (s, e) in enumerate(ranges):
            clip = vid[s : e + 1]  # (L,C,H,W)
            out_name = f"{base}_shot{si}.{io.out_ext}"
            out_path = save_dir / out_name
            if save:
                self._save_video_tensor(clip, out_path, fps=io.fps)
            shot_files.append(str(out_path))

        meta = {
            "source": str(video_path),
            "bounds_inclusive": boundaries,
            "ranges": ranges,
            "files": shot_files,
            "config": {"scene_split": asdict(self.cfg)},
        }
        # Write JSON sidecar
        meta_path = save_dir / f"{base}_shots_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    def _save_video_tensor(self, clip: torch.Tensor, out_path: Path, fps: int = 24) -> None:
        """Save a (L,C,H,W) tensor to mp4 (preferred) or gif (fallback)."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to uint8 HWC
        x = clip.detach().cpu().float()
        if x.max() <= 1.0:
            x = (x * 255.0).clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.permute(0, 2, 3, 1).numpy()  # (L,H,W,C)

        # Try mp4 via imageio-ffmpeg; fallback to gif
        ext = out_path.suffix.lower()
        if ext == ".mp4":
            try:
                import imageio  # type: ignore
                import imageio_ffmpeg  # noqa: F401 (ensure backend present)
                writer = imageio.get_writer(out_path, fps=fps, format="FFMPEG", codec="libx264")
                for frame in x:
                    writer.append_data(frame)
                writer.close()
                return
            except Exception:
                # fallback to gif
                out_path = out_path.with_suffix(".gif")
        # GIF fallback
        try:
            import imageio  # type: ignore
            imageio.mimsave(out_path, list(x), fps=fps)
        except Exception:
            # As a last resort, save tensor
            torch.save(clip, out_path.with_suffix(".pt"))


# ----------------------------
# Demo / Test
# ----------------------------

def _make_dummy_video(path: Path, Flen: int = 60, C: int = 3, H: int = 64, W: int = 64) -> None:
    """Create a toy video tensor with 3 shots by changing background color abruptly."""
    torch.manual_seed(0)
    frames = []
    for t in range(Flen):
        if t < Flen // 3:
            base = torch.tensor([255, 50, 50], dtype=torch.float32).view(3, 1, 1)
        elif t < 2 * Flen // 3:
            base = torch.tensor([50, 255, 50], dtype=torch.float32).view(3, 1, 1)
        else:
            base = torch.tensor([50, 50, 255], dtype=torch.float32).view(3, 1, 1)
        noise = torch.randn(C, H, W) * 5.0
        frame = (base + noise).clamp(0, 255)
        frames.append(frame)
    vid = torch.stack(frames, dim=0)  # (F,C,H,W)
    torch.save(vid, path)


if __name__ == "__main__":
    # 1) Prepare a dummy video tensor file
    demo_dir = Path("./_demo")
    demo_dir.mkdir(exist_ok=True, parents=True)
    vid_path = demo_dir / "demo_video.pt"
    if not vid_path.exists():
        _make_dummy_video(vid_path, Flen=90, H=72, W=72)

    # 2) Build config (Torchvision pretrained backbone)
    cfg = SceneSplitConfig.demo()
    cfg.device = "cpu"  # set to "cuda" if available
    cfg.extractor = ExtractorConfig(
        name="tv-cnn",            # use pretrained torchvision backbone
        tv_backbone="resnet18",   # resnet18/resnet50/mobilenet_v3_large
        tv_use_projection=True,    # project to out_dim below
        out_dim=128,
        stride=1,
        normalize_input=True,
        max_frames_per_chunk=128,
    )
    # With pretrained features, set your decision rule/threshold
    cfg.splitter = SplitterConfig(prev_offset=1, threshold=0.75, rule="less", min_shot_len=8)
    cfg.io = IOConfig(save_dir="./shots_out", fps=12, basename=None, out_ext="mp4")

    # 3) Run splitter
    splitter = SceneSplitter(cfg)
    meta = splitter(vid_path, save=True)
    print(meta)
