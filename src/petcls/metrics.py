"""Evaluation metrics: accuracy@k, macro-F1, calibration (ECE) and confusion analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        centers.append((lo + hi) / 2)
        accs.append(correct[mask].mean() if mask.any() else np.nan)
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(accs), np.array(counts)


def most_confused_pairs(cm: np.ndarray, class_names: list[str], top: int = 10):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:top]
    pairs = []
    for f in flat:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] == 0:
            break
        pairs.append({"true": class_names[i], "predicted": class_names[j], "count": int(off[i, j])})
    return pairs


def summarize(probs: np.ndarray, y: np.ndarray, class_names: list[str]) -> dict:
    pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(len(class_names))))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = np.nan_to_num(cm.diagonal() / cm.sum(1))
    order = np.argsort(per_class)
    return {
        "n_samples": int(len(y)),
        "top1_accuracy": topk_accuracy(probs, y, 1),
        "top5_accuracy": topk_accuracy(probs, y, 5),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ece": expected_calibration_error(probs, y),
        "per_class_accuracy": {
            class_names[i]: float(per_class[i]) for i in range(len(class_names))
        },
        "worst_classes": [
            {"class": class_names[i], "accuracy": float(per_class[i])} for i in order[:5]
        ],
        "most_confused_pairs": most_confused_pairs(cm, class_names),
        "confusion_matrix": cm.tolist(),
    }
