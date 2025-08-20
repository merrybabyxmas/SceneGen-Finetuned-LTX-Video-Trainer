from dataclasses import dataclass

# =========================
# Configs
# =========================

@dataclass
class SBDConfig:
    preproc_batch_size: int = 128
    enc_batch_size: int = 128
    similarity_threshold: float = 0.8
    target_h: int = 224
    target_w: int = 224
    use_gpu: bool = True
    visualize_keyframes: bool = False  # save SOS/EOS JPEGs

@dataclass
class IOConfig:
    splits_root: str = "./splits"           # where shot videos go
    keyframes_root: str = "./keyframes"     # where SOS/EOS JPEGs go (optional)
    split_ext: str = "mp4"                   # mp4 preferred
    default_fps: int = 24

@dataclass
class PipelineConfig:
    sbd: SBDConfig = SBDConfig()
    io: IOConfig = IOConfig()
