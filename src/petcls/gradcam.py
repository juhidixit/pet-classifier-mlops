"""Grad-CAM (Selvaraju et al., 2017) implemented with forward/backward hooks, no extra deps."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torch import nn


class GradCAM:
    """Class-discriminative localisation map for a CNN.

    >>> cam = GradCAM(model, gradcam_target_layer(model))
    >>> heatmaps, logits = cam(x)              # x: [N, 3, H, W]; heatmaps: [N, H, W] in [0, 1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._h1 = target_layer.register_forward_hook(self._save_acts)
        self._h2 = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _inp, out):
        self._acts = out.detach()

    def _save_grads(self, _m, _gin, gout):
        self._grads = gout[0].detach()

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor | list[int] | None = None):
        was_training = self.model.training
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(1)
            class_idx = torch.as_tensor(class_idx, device=logits.device).view(-1)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, class_idx.view(-1, 1)).sum().backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial gradients
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        lo, hi = flat.min(1)[0].view(-1, 1, 1), flat.max(1)[0].view(-1, 1, 1)
        cam = (cam - lo) / (hi - lo + 1e-8)
        self.model.train(was_training)
        return cam.detach().cpu(), logits.detach()


def overlay(
    image: np.ndarray | Image.Image, heatmap: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Blend a [H, W] heatmap in [0, 1] over an RGB image (HWC uint8 / float or PIL)."""
    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    if heatmap.shape != img.shape[:2]:
        heatmap = (
            np.asarray(
                Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                    (img.shape[1], img.shape[0]), Image.BILINEAR
                ),
                dtype=np.float32,
            )
            / 255.0
        )
    colored = colormaps["jet"](heatmap)[..., :3]
    out = (1 - alpha) * img + alpha * colored
    return Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))


def attention_inside_box(heatmap: np.ndarray, box_mask: np.ndarray) -> float:
    """Fraction of Grad-CAM energy that falls inside a binary mask (e.g. the pet's head box).

    A simple 'pointing game'-style faithfulness check: high values mean the model looks at the
    animal rather than background shortcuts.
    """
    total = float(heatmap.sum())
    return float((heatmap * box_mask).sum() / total) if total > 0 else 0.0
