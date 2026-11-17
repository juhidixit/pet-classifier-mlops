"""Held-out test evaluation, calibration, error analysis and Grad-CAM galleries.

    python -m petcls.evaluate --checkpoint artifacts/resnet50/model.pt --out reports/resnet50

Produces `metrics.json` plus figures: confusion matrix, reliability diagram, per-class accuracy,
and Grad-CAM galleries of the most confident correct and the most confident *wrong* predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import DataConfig  # noqa: E402
from .data import build_datasets, denormalize, make_loader, species_of  # noqa: E402
from .engine import predict_loader, select_device  # noqa: E402
from .gradcam import GradCAM, overlay  # noqa: E402
from .metrics import (
    apply_temperature,
    bootstrap_ci,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
    summarize,
)  # noqa: E402
from .models import gradcam_target_layer, load_checkpoint  # noqa: E402

log = logging.getLogger("petcls.evaluate")


def species_error_breakdown(probs: np.ndarray, y: np.ndarray, names: list[str]) -> dict:
    """Are mistakes fine-grained (wrong breed, right species) or coarse (cat <-> dog)?"""
    pred = probs.argmax(1)
    wrong = np.where(pred != y)[0]
    cross = sum(species_of(names[y[i]]) != species_of(names[pred[i]]) for i in wrong)
    return {
        "n_errors": int(len(wrong)),
        "cross_species_errors": int(cross),
        "within_species_errors": int(len(wrong) - cross),
        "species_accuracy": float(1 - cross / len(y)) if len(y) else 0.0,
    }


def plot_confusion(cm: np.ndarray, names: list[str], path: Path) -> None:
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised confusion matrix (test set)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(probs: np.ndarray, y: np.ndarray, ece: float, path: Path) -> None:
    centers, accs, counts = reliability_curve(probs, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ok = ~np.isnan(accs)
    ax.bar(
        centers[ok],
        accs[ok],
        width=1 / len(centers),
        alpha=0.7,
        edgecolor="black",
        label="accuracy per confidence bin",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_per_class(per_class: dict[str, float], path: Path) -> None:
    items = sorted(per_class.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 10))
    colors = ["#c0504d" if species_of(k) == "cat" else "#4f81bd" for k, _ in items]
    ax.barh([k for k, _ in items], [v for _, v in items], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test accuracy")
    ax.set_title("Per-breed accuracy (red = cat, blue = dog)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def gradcam_gallery(model, dataset, indices, probs, y, names, device, path: Path, title: str):
    if len(indices) == 0:
        return
    cols = min(4, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.8 * rows), squeeze=False)
    with GradCAM(model, gradcam_target_layer(model)) as cam:
        for ax, idx in zip(axes.flat, indices, strict=False):
            x, _ = dataset[int(idx)]
            heat, _ = cam(x.unsqueeze(0).to(device))
            img = denormalize(x).permute(1, 2, 0).numpy()
            ax.imshow(overlay(img, heat[0].numpy()))
            p = int(probs[idx].argmax())
            ok = p == y[idx]
            ax.set_title(
                f"true: {names[y[idx]]}\npred: {names[p]} ({probs[idx, p]:.0%})",
                fontsize=8,
                color="green" if ok else "red",
            )
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(
    checkpoint: str,
    out: str,
    data_root: str = "data",
    dataset: str = "oxford_pets",
    batch_size: int = 64,
    num_workers: int = 4,
    tta: bool = True,
    device: str | None = None,
    fake_size: int = 64,
) -> dict:
    dev = select_device(device)
    model, ckpt = load_checkpoint(checkpoint, map_location="cpu")
    model.to(dev)
    names = ckpt["class_names"]
    dcfg = DataConfig(
        root=data_root,
        dataset=dataset,
        img_size=ckpt["img_size"],
        num_workers=num_workers,
        fake_size=fake_size,
    )
    bundle = build_datasets(dcfg)
    test = bundle.test
    probs, y = predict_loader(
        model, make_loader(test, batch_size, False, num_workers), dev, tta=tta
    )
    # Calibrate on the validation split only (never the test set).
    vprobs, vy = predict_loader(
        model, make_loader(bundle.val, batch_size, False, num_workers), dev, tta=tta
    )
    temperature = fit_temperature(np.log(np.clip(vprobs, 1e-12, 1)), vy)
    cal_probs = apply_temperature(probs, temperature)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = summarize(probs, y, names)
    report.update(bootstrap_ci(probs, y))
    report["calibration"] = {
        "temperature": temperature,
        "ece_before": report["ece"],
        "ece_after": expected_calibration_error(cal_probs, y),
    }
    report.update(
        {
            "model": ckpt["model_name"],
            "tta": tta,
            "species": species_error_breakdown(probs, y, names),
            "checkpoint_val_acc": ckpt.get("extra", {}).get("val_acc"),
        }
    )

    plot_confusion(np.array(report["confusion_matrix"]), names, out_dir / "confusion_matrix.png")
    plot_reliability(probs, y, report["ece"], out_dir / "reliability.png")
    plot_per_class(report["per_class_accuracy"], out_dir / "per_class_accuracy.png")

    conf, pred = probs.max(1), probs.argmax(1)
    wrong = np.where(pred != y)[0]
    right = np.where(pred == y)[0]
    worst = wrong[np.argsort(-conf[wrong])][:8]  # confidently wrong = most informative
    best = right[np.argsort(-conf[right])][:4]
    gradcam_gallery(
        model,
        test,
        worst,
        probs,
        y,
        names,
        dev,
        out_dir / "gradcam_errors.png",
        "Grad-CAM: most confident mistakes",
    )
    gradcam_gallery(
        model,
        test,
        best,
        probs,
        y,
        names,
        dev,
        out_dir / "gradcam_correct.png",
        "Grad-CAM: most confident correct predictions",
    )
    report["confident_errors"] = [
        {
            "index": int(i),
            "true": names[y[i]],
            "predicted": names[pred[i]],
            "confidence": float(conf[i]),
        }
        for i in worst
    ]

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    np.save(out_dir / "test_probs.npy", probs.astype(np.float16))
    log.info(
        "%s: top1=%.4f top5=%.4f macroF1=%.4f ECE=%.4f",
        ckpt["model_name"],
        report["top1_accuracy"],
        report["top5_accuracy"],
        report["macro_f1"],
        report["ece"],
    )
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--dataset", default="oxford_pets")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = evaluate(
        a.checkpoint,
        a.out,
        a.data_root,
        a.dataset,
        a.batch_size,
        a.num_workers,
        not a.no_tta,
        a.device,
    )
    print(
        json.dumps(
            {
                k: r[k]
                for k in ("model", "top1_accuracy", "top5_accuracy", "macro_f1", "ece", "species")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
