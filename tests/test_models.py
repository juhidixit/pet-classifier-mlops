import pytest
import torch

from petcls.config import ModelConfig
from petcls.gradcam import GradCAM, attention_inside_box, overlay
from petcls.models import (
    backbone_parameters,
    build_model,
    gradcam_target_layer,
    head_parameters,
    load_checkpoint,
    set_backbone_trainable,
)


@pytest.mark.parametrize("name", ["scratch_cnn", "resnet50"])
def test_forward_shape(name):
    m = build_model(ModelConfig(name=name, pretrained=False), num_classes=37).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 96, 96)).shape == (2, 37)


def test_freeze_backbone_only_trains_head():
    m = build_model(ModelConfig(name="resnet50", pretrained=False), 37)
    set_backbone_trainable(m, False)
    assert all(not p.requires_grad for p in backbone_parameters(m))
    assert all(p.requires_grad for p in head_parameters(m))
    set_backbone_trainable(m, True)
    assert all(p.requires_grad for p in m.parameters())


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model(ModelConfig(name="vit_huge"), 37)


def test_checkpoint_roundtrip(tiny_checkpoint):
    model, meta = load_checkpoint(tiny_checkpoint)
    assert meta["img_size"] == 64 and len(meta["class_names"]) == 37
    assert not model.training


@pytest.mark.parametrize("name", ["scratch_cnn", "resnet50"])
def test_gradcam_heatmap(name):
    m = build_model(ModelConfig(name=name, pretrained=False), 37)
    x = torch.randn(3, 3, 64, 64)
    with GradCAM(m, gradcam_target_layer(m)) as cam:
        heat, logits = cam(x)
        heat_t, _ = cam(x, [5, 5, 5])
    assert heat.shape == (3, 64, 64) and logits.shape == (3, 37)
    assert heat.min() >= 0 and heat.max() <= 1 + 1e-6
    assert heat_t.shape == heat.shape
    # hooks are removed on exit
    assert not gradcam_target_layer(m)._forward_hooks


def test_overlay_and_box_attention():
    import numpy as np

    heat = np.zeros((32, 32), dtype=np.float32)
    heat[8:16, 8:16] = 1.0
    img = overlay(np.zeros((64, 64, 3), dtype=np.uint8), heat)
    assert img.size == (64, 64)
    mask = np.zeros_like(heat)
    mask[8:16, 8:16] = 1
    assert attention_inside_box(heat, mask) == pytest.approx(1.0)
    assert attention_inside_box(heat, 1 - mask) == pytest.approx(0.0)
