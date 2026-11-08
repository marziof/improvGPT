# src/gpt/config.py
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import yaml


@dataclass
class Config:
    device: str = "auto"        # "auto" | "cpu" | "cuda" | "mps"
    max_steps: int = 5000
    eval_interval: int = 250
    eval_batches: int = 20
    ckpt_interval: int = 1000
    run_dir: str = "runs"
    run_name: str | None = None        # auto-filled in load_config
    resume: str | None = None          # path to a checkpoint to resume from
    init_from: str | None = None
    min_lr_ratio: float = 0.1
    lr: float = 1.0e-3
    weight_decay: float = 0.1
    optimizer: str = "adamw"
    warmup_steps: int = 100  
    min_lr_ratio: float = 0.1  
    grad_clip: float = 1.0
    seed: int=42
    overfit: bool = False
    overwrite: bool = False
    tokenizer_sha256: str | None = None
    data_format: str = "stream"          # "stream" (flat token file) or "examples" (masked improv rows)
    copy_embedding: list[int] | None = None      # [dst, src]: initialize token dst's embedding row from token src

    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    context_len: int = 256
    batch_size: int = 64
    mlp_mult: int = 4
    data_dir: str = "data"
    vocab_size: int | None = None      # filled from meta.json if left as None

    amp: str = "none"                  # "none" or "fp16" (fp16 only takes effect on CUDA)
    wandb: bool = False
    wandb_project: str = "swgpt"



    def __post_init__(self):
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert (self.d_model // self.n_head) % 2 == 0, "head_dim must be even for RoPE"
        assert isinstance(self.lr, float), "lr must be a float; write 1.0e-3, not 1e-3, in YAML"
        assert self.optimizer in ("adamw", "sgd"), f"unknown optimizer {self.optimizer}"
        assert self.warmup_steps < self.max_steps, "warmup_steps must be < max_steps"
        assert self.amp in ("none", "fp16"), f"unknown amp mode {self.amp}"
        assert self.device in ("auto", "cpu", "cuda", "mps"), f"unknown device {self.device}"
        assert self.data_format in ("stream", "examples")




ARCH_KEYS = ("n_layer", "n_head", "d_model", "context_len", "vocab_size", "mlp_mult")

def load_config(path, **overrides):
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    d.update(overrides)

    if d.get("init_from"):                       # architecture is dictated by the pretrained checkpoint
        import torch
        ck_cfg = torch.load(d["init_from"], map_location="cpu")["cfg"]
        for k in ARCH_KEYS:
            if k in d and d[k] != ck_cfg[k]:
                raise ValueError(f"{k}={d[k]} conflicts with the checkpoint ({ck_cfg[k]})")
            d[k] = ck_cfg[k]

    cfg = Config(**d)

    meta = json.loads((Path(cfg.data_dir) / "meta.json").read_text())
    if cfg.vocab_size is None:
        cfg.vocab_size = meta["vocab_size"]
    else:                                        # set from the checkpoint above
        assert meta["vocab_size"] == cfg.vocab_size, "data tokenizer vocab != model vocab"
    from gpt.utils import checkpoint_tokenizer_hash         # local import avoids a circular import at module load
    cfg.tokenizer_sha256 = meta.get("tokenizer_sha256")      # hash of the tokenizer behind THIS run's data
    if d.get("init_from"):
        ck_hash = checkpoint_tokenizer_hash(d["init_from"], ck_cfg)
        if ck_hash and cfg.tokenizer_sha256 and ck_hash != cfg.tokenizer_sha256:
            raise ValueError("data folder and init_from checkpoint were tokenized with different tokenizers")
        if not (ck_hash and cfg.tokenizer_sha256):
            print("warning: tokenizer hash missing on the checkpoint or the data folder; only the vocab size was checked")
    if cfg.run_name is None:
        cfg.run_name = f"{datetime.now():%Y-%m-%d_%H%M}_{Path(path).stem}"
    return cfg

