from __future__ import annotations

from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- SOS token utilities ----------
def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std)
    return tensor


def _sinusoidal_pos_emb(n_positions: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(n_positions, dim)
    position = torch.arange(0, n_positions, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SOSToken(nn.Module):
    """
      - learnable base frame [C,H,W] (trunc_normal init)
      - sinusoidal temporal PE -> project to channels and add with scale alpha
      - outputs [F,C,H,W]
    """
    def __init__(self, channels: int = 3, height: int = 256, width: int = 256, pe_dim: Optional[int] = None):
        super().__init__()
        self.C, self.H, self.W = channels, height, width
        self.base = nn.Parameter(torch.empty(channels, height, width))
        _trunc_normal_(self.base, std=0.02)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        pe_dim = pe_dim or min(64, channels)
        self.pe_proj = nn.Linear(pe_dim, channels, bias=False)
        nn.init.xavier_uniform_(self.pe_proj.weight)

    @torch.no_grad()
    def _resize_if_needed(self, H: int, W: int):
        if H != self.H or W != self.W:
            x = self.base.data.unsqueeze(0)
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            self.base.data = x
            self.H, self.W = H, W

    def forward(self, F_len: int, C: int, H: int, W: int, device=None) -> torch.Tensor:
        if device is None:
            device = self.base.device
        self._resize_if_needed(H, W)

        base = self.base.to(device).unsqueeze(0).expand(F_len, -1, -1, -1)

        pe_dim = self.pe_proj.in_features
        pe = _sinusoidal_pos_emb(F_len, pe_dim).to(device)
        pe_c = self.pe_proj(pe).unsqueeze(-1).unsqueeze(-1)

        sos = base + self.alpha * pe_c
        return sos.clamp(0.0, 1.0)


PRECOMPUTED_DIR_NAME = ".precomputed"


class PrecomputedDataset(Dataset):
    def __init__(self, data_root: str, data_sources: dict[str, str] | list[str] | None = None,
                 sos_sources: Optional[set[str]] = None,
                 num_shots: int = 4) -> None:
        super().__init__()
        self.data_root = self._setup_data_root(data_root)  
        # ex) data_root = Path("dataset")  
        # 실제 사용 경로: dataset/.precomputed  ← 있으면 거기로 이동

        self.data_sources = self._normalize_data_sources(data_sources)  
        # 기본값: {"latents":"latent_conditions", "conditions":"text_conditions"}  
        # 즉, 폴더 이름(latents) ↔ 출력 key(latent_conditions)

        self.sos_sources = sos_sources or set()  
        # ex) {"latents"} → 첫 샷(prev="[SOS]")일 때 SOS 토큰으로 대체

        self.num_shots = num_shots  
        # 한 비디오 당 몇 개 샷이 저장되어 있다고 가정하는지. ex) 4

        self.source_paths = self._setup_source_paths()  
        # {"latents": Path(dataset/.precomputed/latents), ...}

        self.sample_files = self._discover_samples()  
        # {"latent_conditions": [ex1.pt, ex2.pt, ...], "text_conditions":[...]}

        self._validate_setup()  
        # 각 source의 파일 개수가 일치하는지 검사


    # ---------- setup helpers ----------
    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        data_root = Path(data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        pre = data_root / PRECOMPUTED_DIR_NAME
        if pre.exists():
            return pre
        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None = None) -> dict[str, str]:
        if data_sources is None:
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            return {src: src for src in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        source_paths: dict[str, Path] = {}
        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path
            if not source_path.exists():
                raise FileNotFoundError(f"Required '{dir_name}' directory doesn't exist: {source_path}")
        return source_paths

    # ---------- discovery ----------
    def _discover_samples(self) -> dict[str, List[Path]]:
        # choose a primary key to enumerate files
        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources))
        data_path = self.source_paths[data_key]
        data_files = list(data_path.glob("**/*.pt"))
        if not data_files:
            raise ValueError(f"No data files found in {data_path}")

        sample_files: dict[str, List[Path]] = {out_key: [] for out_key in self.data_sources.values()}

        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)
            if self._all_source_files_exist(data_file, rel_path):
                self._fill_sample_data_files(data_file, rel_path, sample_files)
        return sample_files

    def _all_source_files_exist(self, data_file: Path, rel_path: Path) -> bool:
        # all-or-nothing: ensure the same rel_path (or mapped name) exists under every source dir
        all_ok = True
        for dir_name in self.data_sources:
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            if not expected_path.exists():
                logging.warning(
                    f"No matching '{dir_name}' file for {data_file.name} (expected: {expected_path})"
                )
                all_ok = False
        return all_ok

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        source_path = self.source_paths[dir_name]

        # legacy rename: latent_XXX.pt -> condition_XXX.pt under "conditions"
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, List[Path]]) -> None:
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        if not self.sample_files:
            raise ValueError("No valid samples found - all data sources must have matching files.")

        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    # ---------- indexing ----------
    def __len__(self) -> int:
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])  
        # ex) latent_conditions에 [ex1.pt, ex2.pt, ex3.pt] 있으면 → 길이=3

    def _load_curr_prev_video_index(self, index: int) -> dict[str, Any]:
        video_index = index // self.num_shots
        shot_index = index % self.num_shots
        prev_index = index - 1 if shot_index != 0 else "[SOS]"
        return {
            "curr_index": index,
            "prev_index": prev_index,
            "video_index": video_index,
            "shot_index": shot_index,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        idx_meta = self._load_curr_prev_video_index(index)
        # ex) index=5, num_shots=4 → video_index=1, shot_index=1, prev_index=4

        result: dict[str, Any] = {"idx": index}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            # ex) "latents" → dataset/.precomputed/latents

            # 현재 샷 파일 경로
            file_rel_curr_path = self.sample_files[output_key][idx_meta["curr_index"]]  
            # ex) sample_files["latent_conditions"][5] = Path("ex5.pt")
            file_curr_path = source_path / file_rel_curr_path

            curr_data = torch.load(file_curr_path, map_location="cpu")  
            # 현재 샷 로드: Tensor or dict

            # 이전 샷 처리
            if idx_meta["prev_index"] == "[SOS]" and (dir_name in self.sos_sources):
                # ex) 첫 샷인데 source가 latents일 때
                prev_data = self._make_sos_like(curr_data)
                # → curr_data와 같은 shape의 SOS tensor 생성
            elif idx_meta["prev_index"] == "[SOS]":
                prev_data = None  # SOS 적용 안 하는 소스는 None
            else:
                file_rel_prev_path = self.sample_files[output_key][idx_meta["prev_index"]]
                # ex) sample_files["latent_conditions"][4] = Path("ex4.pt")
                file_prev_path = source_path / file_rel_prev_path
                prev_data = torch.load(file_prev_path, map_location="cpu")

            # result 구조:
            # result = {
            #   "latent_conditions": {"current_shot": Tensor, "prev_shot": Tensor or SOS},
            #   "text_conditions": {"current_shot": Tensor, "prev_shot": None or Tensor},
            #   "idx": index
            # }
            result[output_key] = {
                "current_shot": curr_data,
                "prev_shot": prev_data,
                "meta": idx_meta,
            }

        return result

    # ---------- helpers ----------
    def _make_sos_like(self, ref: torch.Tensor) -> torch.Tensor:
        """
        ref가 [F,C,H,W] 또는 [C,H,W] 또는 기타 케이스일 때도 최대한 합리적으로 SOS를 생성.
        - [F,C,H,W]: 그대로 F,C,H,W로 생성
        - [C,H,W]: F=1 가정해 [1,C,H,W] 생성
        - [*,C,H,W]: 마지막 3차원을 기준으로 C,H,W를 추론, F는 앞 차원 곱 → 1로 두는 것이 안전
        """
        device = ref.device if ref.is_cuda else "cpu"
        if ref.dim() == 4:
            F_len, C, H, W = ref.shape
        elif ref.dim() == 3:
            C, H, W = ref.shape
            F_len = 1
        else:
            # 관용 처리: 뒤에서 3차원은 [C,H,W]로 가정, 앞쪽은 F로 collapse
            C, H, W = ref.shape[-3:]
            F_len = int(ref.numel() // (C * H * W))
            if F_len <= 0:
                F_len = 1

        sos = SOSToken(channels=C, height=H, width=W).to(device)
        with torch.no_grad():
            out = sos(F_len=F_len, C=C, H=H, W=W, device=device)
        return out
