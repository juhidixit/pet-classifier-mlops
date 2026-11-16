import pytest
import torch

from petcls.config import ModelConfig
from petcls.models import build_model, save_checkpoint

CLASSES = [f"class_{i}" for i in range(37)]


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """A randomly initialised scratch CNN checkpoint (no downloads needed)."""
    torch.manual_seed(0)
    cfg = ModelConfig(name="scratch_cnn", pretrained=False)
    model = build_model(cfg, num_classes=len(CLASSES))
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    save_checkpoint(path, model, cfg, CLASSES, img_size=64)
    return path
