"""Device selection, one training epoch, and batched inference."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def select_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LR multiplier: linear warm-up then cosine decay to 1% of the base LR."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return fn


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    amp: bool = False,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.train()
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss, correct, seen = 0.0, 0, 0
    for step, (x, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    return {"loss": total_loss / max(1, seen), "accuracy": correct / max(1, seen)}


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device, tta: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (softmax probabilities [N, C], labels [N]). Optional horizontal-flip TTA."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).softmax(1)
        if tta:
            p = 0.5 * (p + model(torch.flip(x, dims=[3])).softmax(1))
        probs.append(p.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)
