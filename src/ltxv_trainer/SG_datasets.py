from __future__ import annotations

from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
from collections import defaultdict

from pathlib import Path
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

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



class SOSTokenLatents(nn.Module):
    """
    SOS token generator for latent sequence [Seq, D].
    - fixed base token [1, D] (trunc_normal init, non-learnable)
    - sinusoidal positional embedding -> project to D and add
    - outputs [Seq, D]
    """
    def __init__(self, d_model: int = 128, pe_dim: Optional[int] = None, use_zero_init: bool = True):
        super().__init__()
        self.d_model = d_model
        
        # Fixed base token (non-learnable)
        if use_zero_init:
            # Simple zero initialization for maximum stability
            base_token = torch.zeros(1, d_model)
        else:
            # Truncated normal initialization (original approach)
            base_token = torch.empty(1, d_model)
            _trunc_normal_(base_token, std=0.02)
        self.register_buffer('base', base_token)  # Non-learnable, fixed initialization
        
        # Fixed alpha value (non-learnable) - minimal influence
        self.register_buffer('alpha', torch.tensor(0.001))  # Very small fixed value

        pe_dim = pe_dim or min(64, d_model)
        self.pe_proj = nn.Linear(pe_dim, d_model, bias=False)
        
        if use_zero_init:
            # Zero initialize projection weights for stability
            nn.init.zeros_(self.pe_proj.weight)
        else:
            # Xavier initialization (original approach)
            nn.init.xavier_uniform_(self.pe_proj.weight)
        
        # Freeze pe_proj to make everything non-learnable
        for param in self.pe_proj.parameters():
            param.requires_grad = False

    def forward(self, seq_len: int, device=None) -> torch.Tensor:
        if device is None:
            device = self.base.device

        # base: [1, D] -> expand to [Seq, D]
        base = self.base.to(device).expand(seq_len, -1)  # (Seq, D)

        # sinusoidal pos emb [Seq, pe_dim] -> project -> [Seq, D]
        pe_dim = self.pe_proj.in_features
        pe = _sinusoidal_pos_emb(seq_len, pe_dim).to(device)  # (Seq, pe_dim)
        pe_c = self.pe_proj(pe)  # (Seq, D)

        sos = base + self.alpha * pe_c
        return sos.clamp(-1.0, 1.0)  # (Seq, D) - match VAE latent range


PRECOMPUTED_DIR_NAME = ".precomputed"



class PrecomputedDataset(Dataset):
    """
    Precomputed .pt 파일명이 <orig_stem>_shot{idx}.pt 인 것을 가정.
    - 정규식으로 (video_id, shot_idx) 추출하여 가변 샷 개수 지원
    - 같은 video_id 내부에서 shot_idx 오름차순으로 정렬
    - prev 샷은 같은 video_id의 직전 shot_idx를 사용 (첫 샷은 SOS)
    """

    # ex) "myvideo_s01_shot12.pt" 같은 변형도 견고하게 잡기 위해
    # 파일 'stem'(확장자 제거)에 대해 오른쪽 끝의 `_shot{idx}`를 추출
    _SHOT_RE = re.compile(r"^(?P<base>.+?)_shot(?P<idx>\d+)(?:_.*)?$")

    def __init__(
        self,
        data_root: str,
        data_sources: Dict[str, str] | List[str] | None = None,
        sos_sources: Optional[set[str]] = None,
        num_shots: int = 0,  # 더이상 사용하지 않지만, 외부 호환을 위해 인자 유지
    ) -> None:
        super().__init__()
        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.sos_sources = sos_sources or set()
        self.num_shots = num_shots  # 미사용(호환성)

        self.source_paths = self._setup_source_paths()

        # 새 인덱스 구조
        # entries[i] = {
        #   "video_id": str,
        #   "shot_idx": int,
        #   "per_source_rel": { output_key: Path(relative_to_source_dir) }
        # }
        self.entries: List[Dict[str, Any]] = []
        # 빠른 prev 조회용: (video_id, shot_idx, output_key) -> rel_path
        self._key_to_rel: Dict[tuple, Path] = {}
        # video_id -> sorted shot list
        self._video_to_shots: Dict[str, List[int]] = defaultdict(list)

        self._discover_entries()
        self._prepare_prev_links()
        self._validate_entries()

    # ---------- setup helpers ----------
    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        data_root = Path(data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")
        pre = data_root / PRECOMPUTED_DIR_NAME
        return pre if pre.exists() else data_root

    @staticmethod
    def _normalize_data_sources(data_sources: Dict[str, str] | List[str] | None = None) -> Dict[str, str]:
        if data_sources is None:
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            return {src: src for src in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> Dict[str, Path]:
        source_paths: Dict[str, Path] = {}
        for dir_name in self.data_sources:
            p = self.data_root / dir_name
            source_paths[dir_name] = p
            if not p.exists():
                raise FileNotFoundError(f"Required '{dir_name}' directory doesn't exist: {p}")
        return source_paths

    # ---------- parsing helpers ----------
    def _parse_shot_from_stem(self, stem: str) -> Optional[tuple[str, int]]:
        """
        stem: 파일명에서 확장자 제거한 부분. 예: 'movieA_shot3'
        return: (video_id, shot_idx) or None(매칭 실패)
        """
        m = self._SHOT_RE.match(stem)
        if not m:
            return None
        base = m.group("base")
        idx = int(m.group("idx"))
        return base, idx

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """
        다른 소스(conditions 등)에서 동일한 상대경로/이름을 기대하는 규칙.
        필요한 경우 레거시 이름 치환 규칙을 여기에 추가.
        """
        source_dir = self.source_paths[dir_name]
        return source_dir / rel_path

    # ---------- discovery ----------
    def _discover_entries(self) -> None:
        primary_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources))
        primary_root = self.source_paths[primary_key]
        data_files = list(primary_root.glob("**/*.pt"))
        if not data_files:
            raise ValueError(f"No .pt files found under {primary_root}")

        # temp: video_id -> shot_idx -> per_source_rel mapping (검증 중복 방지)
        tmp_map: Dict[str, Dict[int, Dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))

        for data_file in data_files:
            rel_path = data_file.relative_to(primary_root)
            shot_info = self._parse_shot_from_stem(data_file.stem)
            if not shot_info:
                logging.warning(f"[SKIP] filename does not match '*_shot{{idx}}.pt': {data_file.name}")
                continue
            video_id, shot_idx = shot_info

            # 모든 소스에 대해 존재 확인 및 상대경로 수집
            all_ok = True
            per_source_rel: Dict[str, Path] = {}
            for dir_name, output_key in self.data_sources.items():
                expected = self._get_expected_file_path(dir_name, data_file, rel_path)
                if not expected.exists():
                    logging.warning(
                        f"[SKIP] missing '{dir_name}' for {data_file.name} (expected: {expected})"
                    )
                    all_ok = False
                    break
                per_source_rel[output_key] = expected.relative_to(self.source_paths[dir_name])

            if not all_ok:
                continue

            # 누적
            tmp_map[video_id][shot_idx] = per_source_rel

        # 정렬하여 entries 구축
        for video_id, shots_dict in tmp_map.items():
            shot_list = sorted(shots_dict.keys())
            for sidx in shot_list:
                per_source_rel = shots_dict[sidx]
                self.entries.append({
                    "video_id": video_id,
                    "shot_idx": sidx,
                    "per_source_rel": per_source_rel,
                })
                for output_key, rel in per_source_rel.items():
                    self._key_to_rel[(video_id, sidx, output_key)] = rel
            self._video_to_shots[video_id] = shot_list

        # 전역 정렬 (video_id, shot_idx)
        self.entries.sort(key=lambda e: (e["video_id"], e["shot_idx"]))

    def _prepare_prev_links(self) -> None:
        """각 entry에 prev_shot_idx를 주입 (같은 video_id 내 직전 샷)"""
        for e in self.entries:
            vid = e["video_id"]
            sidx = e["shot_idx"]
            lst = self._video_to_shots[vid]
            # 이진 탐색해도 되지만 shot 수가 크지 않다고 보고 index 사용
            pos = lst.index(sidx)
            e["prev_shot_idx"] = None if pos == 0 else lst[pos - 1]

    def _validate_entries(self) -> None:
        if not self.entries:
            raise ValueError("No valid shot entries discovered. Check filenames and directories.")
        # (선택) 각 소스별 개수 일치 여부 로그
        counts_per_source = defaultdict(int)
        for e in self.entries:
            for ok in e["per_source_rel"].keys():
                counts_per_source[ok] += 1
        logging.info(f"[PrecomputedDataset] discovered {len(self.entries)} shots "
                     f"across sources: {dict(counts_per_source)}")

    # ---------- dataset API ----------
    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import pprint
        entry = self.entries[index]
        video_id = entry["video_id"]
        shot_idx = entry["shot_idx"]
        prev_shot_idx = entry["prev_shot_idx"]
        
        # print(f"curr shot idx : {shot_idx} "
        #       f"prev shot idx : {prev_shot_idx}")

        result = {}
        
        # 현재 샷 데이터 로드
        for dir_name, output_key in self.data_sources.items():
            source_root = self.source_paths[dir_name]
            try:
                rel_data = entry["per_source_rel"][output_key]
                data_path = source_root / rel_data
                data = torch.load(data_path, map_location="cpu", weights_only=True)
                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {data_path}: {e}") from e
                
        # 이전 샷 데이터 로드 (latents만 처리)
        if "latents" in self.data_sources:
            latents_output_key = self.data_sources["latents"]
            source_root = self.source_paths["latents"]
            
            if prev_shot_idx is not None:
                # 이전 샷이 있는 경우 로드
                try:
                    key = (video_id, prev_shot_idx, latents_output_key)
                    rel_prev = self._key_to_rel[key]
                    prev_path = source_root / rel_prev
                    prev_data = torch.load(prev_path, map_location="cpu", weights_only=True)
                    result["prev_conditions"] = prev_data
                except Exception as e:
                    logging.warning(f"Failed to load prev shot {prev_shot_idx}: {e}, using SOS")
                    # 이전 샷 로딩 실패시 SOS 사용
                    curr_data = result[latents_output_key]
                    result["prev_conditions"] = self._make_sos_like(curr_data)
            else:
                # 첫 번째 샷인 경우 SOS 토큰 생성
                curr_data = result[latents_output_key]
                result["prev_conditions"] = self._make_sos_like(curr_data)
                    
        result["idx"] = index
        return result
    # ---------- helpers ----------
    def _make_sos_like(self, ref: torch.Tensor) -> torch.Tensor:
        """
        ref가 dict인 경우 등 다양한 포맷을 고려한다면 여기서 분기 처리.
        - dict{"latents": Tensor, ...} → latents만 SOS로 교체
        - Tensor → SOS tensor 생성
        """
        if isinstance(ref, dict):
            if "latents" in ref:
                tensor = ref["latents"]
                sos_tensor = self._make_sos_tensor_like(tensor)
                out = dict(ref)
                out["latents"] = sos_tensor
                return out
            else:
                logging.warning(f"[SOS] Dict ref missing 'latents' key, found keys: {list(ref.keys())}")
                # 첫 번째 텐서 값을 사용하여 SOS 생성
                first_tensor = None
                for value in ref.values():
                    if torch.is_tensor(value):
                        first_tensor = value
                        break
                if first_tensor is not None:
                    sos_tensor = self._make_sos_tensor_like(first_tensor)
                    out = dict(ref)
                    # 'latents' 키가 없으면 첫 번째 키를 사용
                    first_key = next(iter(ref.keys()))
                    out[first_key] = sos_tensor
                    return out
                else:
                    logging.error("[SOS] No tensor found in dict ref")
                    return torch.zeros(1, 128)
        elif torch.is_tensor(ref):
            return self._make_sos_tensor_like(ref)
        else:
            # 알 수 없는 포맷: 기본 SOS 텐서 반환
            logging.warning(f"[SOS] unknown ref format: {type(ref)}, returning default SOS tensor.")
            return torch.zeros(1, 128)

    def _make_sos_tensor_like(self, ref: torch.Tensor) -> torch.Tensor:
        """
        Latent 형식에 맞춰 SOS token 생성
        지원 형식:
        - [Seq, D]: 직접 SOS 생성
        - [B, Seq, D]: 배치 차원 유지하여 SOS 생성
        """
        device = ref.device if ref.is_cuda else "cpu"
        dtype = ref.dtype
        
        # print(f"ref shape : {ref.shape}")

        if ref.dim() == 2:
            # [Seq, D] 형식
            seq_len, d_model = ref.shape
            sos_gen = SOSTokenLatents(d_model=d_model).to(device)
            with torch.no_grad():
                out = sos_gen(seq_len=seq_len, device=device)  # (Seq, D)
            return out.to(dtype)
            
        elif ref.dim() == 3:
            # [B, Seq, D] 형식
            batch_size, seq_len, d_model = ref.shape
            sos_gen = SOSTokenLatents(d_model=d_model).to(device)
            with torch.no_grad():
                # 각 배치에 대해 동일한 SOS 생성
                base_sos = sos_gen(seq_len=seq_len, device=device)  # (Seq, D)
                out = base_sos.unsqueeze(0).expand(batch_size, -1, -1)  # (B, Seq, D)
            return out.to(dtype)
            
        else:
            logging.error(f"[SOS] Unsupported latent tensor shape: {ref.shape}, expected [Seq, D] or [B, Seq, D]")
            # 폴백: 가장 간단한 형태로 SOS 생성
            d_model = ref.shape[-1] if ref.numel() > 0 else 128
            seq_len = ref.shape[-2] if ref.dim() >= 2 else 1
            sos_gen = SOSTokenLatents(d_model=d_model).to(device)
            with torch.no_grad():
                out = sos_gen(seq_len=seq_len, device=device)
            return out.to(dtype)



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = PrecomputedDataset("/home/jeongseon38/datasets/videos/splits/.precomputed")
    # loader = DataLoader(ds, batch_size=4, shuffle=True)
    # for batch in loader:
    #     print(batch)
    it = iter(ds)

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    for batch in loader:
        # print(batch)
        print("-----------------")
        print(batch["latent_conditions"]["latents"].shape)
        print(batch["latent_conditions"]["num_frames"])
        print(batch["latent_conditions"]["height"])
        print(batch["latent_conditions"]["width"])
        
        print("-----------------")
        break
        
        
        