"""Single-image inference + Grad-CAM, shared by the API and the demo app."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .data import eval_transform
from .gradcam import GradCAM, overlay
from .models import gradcam_target_layer, load_checkpoint


@dataclass
class Prediction:
    label: str
    confidence: float
    top_k: list[dict]


class Predictor:
    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, meta = load_checkpoint(checkpoint, map_location=self.device)
        self.model.to(self.device)
        self.class_names: list[str] = meta["class_names"]
        self.img_size: int = meta["img_size"]
        self.model_name: str = meta["model_name"]
        self.tf = eval_transform(self.img_size)

    @staticmethod
    def load_image(data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert("RGB")

    def _tensor(self, img: Image.Image) -> torch.Tensor:
        return self.tf(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, img: Image.Image, k: int = 5) -> Prediction:
        probs = self.model(self._tensor(img)).softmax(1)[0].cpu().numpy()
        top = np.argsort(-probs)[:k]
        top_k = [{"label": self.class_names[i], "probability": float(probs[i])} for i in top]
        return Prediction(top_k[0]["label"], top_k[0]["probability"], top_k)

    def explain(self, img: Image.Image, class_name: str | None = None) -> Image.Image:
        """Grad-CAM overlay on the center-cropped image for `class_name` (default: top-1)."""
        x = self._tensor(img)
        idx = None if class_name is None else [self.class_names.index(class_name)]
        with GradCAM(self.model, gradcam_target_layer(self.model)) as cam:
            heat, _ = cam(x, idx)
        crop = self.tf.transforms[1](self.tf.transforms[0](img))  # same resize + center-crop
        return overlay(crop, heat[0].numpy())
