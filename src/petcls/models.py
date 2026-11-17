"""Model zoo: transfer-learned ResNet-50 and a from-scratch CNN baseline."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .config import ModelConfig


def _conv_bn(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class ScratchCNN(nn.Module):
    """VGG-style CNN with BatchNorm + global average pooling (~4.7M params).

    Deliberately simple: it is the *baseline* that shows what transfer learning buys you.
    """

    def __init__(
        self,
        num_classes: int = 37,
        dropout: float = 0.3,
        widths: tuple[int, ...] = (32, 64, 128, 256, 512),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c = 3
        for w in widths:
            layers += [_conv_bn(c, w), _conv_bn(w, w), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_model(cfg: ModelConfig, num_classes: int) -> nn.Module:
    if cfg.name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
        m = models.resnet50(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(m.fc.in_features, num_classes))
        return m
    if cfg.name == "resnet18_scratch":
        # Same data pipeline, random init: a *stronger* from-scratch baseline than ScratchCNN.
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(m.fc.in_features, num_classes))
        return m
    if cfg.name == "scratch_cnn":
        return ScratchCNN(num_classes=num_classes, dropout=cfg.dropout)
    raise ValueError(f"Unknown model {cfg.name!r}")


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    if isinstance(model, models.ResNet):
        return list(model.fc.parameters())
    return list(model.parameters())


def backbone_parameters(model: nn.Module) -> list[nn.Parameter]:
    head = {id(p) for p in head_parameters(model)}
    return [p for p in model.parameters() if id(p) not in head]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in backbone_parameters(model):
        p.requires_grad_(trainable)


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block: the usual Grad-CAM target."""
    if isinstance(model, models.ResNet):
        return model.layer4[-1]
    if isinstance(model, ScratchCNN):
        # last conv_bn block before the final max-pool
        return model.features[-2]
    raise ValueError(f"No Grad-CAM target known for {type(model).__name__}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    path,
    model: nn.Module,
    model_cfg: ModelConfig,
    class_names: list[str],
    img_size: int,
    extra: dict | None = None,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_cfg.name,
            "dropout": model_cfg.dropout,
            "class_names": class_names,
            "img_size": img_size,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, map_location: str | torch.device = "cpu"):
    """Returns (model in eval mode, checkpoint metadata)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig(name=ckpt["model_name"], pretrained=False, dropout=ckpt.get("dropout", 0.0))
    model = build_model(cfg, num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
