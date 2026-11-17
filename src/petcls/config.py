"""Typed experiment configuration loaded from YAML with CLI overrides."""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data"
    dataset: str = "oxford_pets"  # "oxford_pets" | "fake" (for smoke tests / CI)
    img_size: int = 224
    val_fraction: float = 0.1
    num_workers: int = 4
    fake_size: int = 64  # number of samples per split when dataset == "fake"


@dataclass
class ModelConfig:
    name: str = "resnet50"  # "resnet50" | "resnet18_scratch" | "scratch_cnn"
    pretrained: bool = True
    dropout: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    backbone_lr_mult: float = 0.1  # discriminative LR for pretrained layers
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 1
    freeze_backbone_epochs: int = 1  # train only the head first, then unfreeze
    patience: int = 5  # early stopping on val accuracy
    max_steps_per_epoch: int | None = None  # cap for smoke tests
    amp: bool = True
    seed: int = 42


@dataclass
class TrackingConfig:
    mlflow: bool = True
    mlflow_uri: str = "sqlite:///mlflow.db"
    wandb: bool = False
    wandb_project: str = "pet-classifier-mlops"
    experiment: str = "oxford-pets"


@dataclass
class Config:
    run_name: str = "resnet50"
    output_dir: str = "artifacts"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def flat(self) -> dict[str, Any]:
        """Flattened `section.key` dict, handy for experiment trackers."""
        out: dict[str, Any] = {}
        for k, v in self.to_dict().items():
            if isinstance(v, dict):
                out.update({f"{k}.{kk}": vv for kk, vv in v.items()})
            else:
                out[k] = v
        return out


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "tracking": TrackingConfig,
}


def _coerce(value: str, current: Any) -> Any:
    """Parse a CLI override string using YAML rules (so 1e-3, true, null all work)."""
    parsed = yaml.safe_load(value)
    if isinstance(parsed, str):  # YAML 1.1 reads "3e-4" as a string
        with contextlib.suppress(ValueError):
            parsed = float(parsed)
    if isinstance(current, float) and isinstance(parsed, int):
        return float(parsed)
    return parsed


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply `section.key=value` overrides, e.g. `train.lr=3e-4`."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        target: Any = cfg
        for p in parts[:-1]:
            target = getattr(target, p)
        if not hasattr(target, parts[-1]):
            raise KeyError(f"Unknown config key: {key}")
        setattr(target, parts[-1], _coerce(value, getattr(target, parts[-1])))
    return cfg


def from_dict(d: dict[str, Any]) -> Config:
    cfg = Config()
    for k, v in (d or {}).items():
        if k in _SECTIONS:
            section = getattr(cfg, k)
            for kk, vv in v.items():
                if not hasattr(section, kk):
                    raise KeyError(f"Unknown config key: {k}.{kk}")
                setattr(section, kk, vv)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise KeyError(f"Unknown config key: {k}")
    return cfg


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    d = yaml.safe_load(Path(path).read_text()) if path else {}
    return apply_overrides(from_dict(d), overrides or [])
