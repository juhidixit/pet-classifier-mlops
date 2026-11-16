"""Training entry point.

    python -m petcls.train --config configs/resnet50.yaml train.lr=3e-4 train.epochs=15

Two-stage transfer learning for pretrained models (head-only warm-up, then full fine-tuning with
a discriminative learning rate), label smoothing, warm-up + cosine schedule, early stopping on a
stratified validation split, and MLflow / W&B tracking. Writes the best checkpoint and a JSON
summary to `<output_dir>/<run_name>/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch import nn

from .config import Config, load_config
from .data import build_datasets, make_loader
from .engine import predict_loader, seed_everything, select_device, train_one_epoch, warmup_cosine
from .metrics import topk_accuracy
from .models import (
    backbone_parameters,
    build_model,
    count_parameters,
    head_parameters,
    save_checkpoint,
    set_backbone_trainable,
)
from .tracking import Tracker

log = logging.getLogger("petcls.train")


def _optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    t = cfg.train
    if cfg.model.name == "resnet50" and cfg.model.pretrained:
        groups = [
            {"params": backbone_parameters(model), "lr": t.lr * t.backbone_lr_mult},
            {"params": head_parameters(model), "lr": t.lr},
        ]
    else:
        groups = [{"params": model.parameters(), "lr": t.lr}]
    return torch.optim.AdamW(groups, weight_decay=t.weight_decay)


def run(cfg: Config, device: str | None = None) -> dict:
    seed_everything(cfg.train.seed)
    dev = select_device(device)
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    data = build_datasets(cfg.data, seed=cfg.train.seed)
    train_dl = make_loader(data.train, cfg.train.batch_size, True, cfg.data.num_workers)
    val_dl = make_loader(data.val, cfg.train.batch_size * 2, False, cfg.data.num_workers)

    model = build_model(cfg.model, len(data.class_names)).to(dev)
    log.info(
        "model=%s params=%.2fM device=%s train=%d val=%d test=%d",
        cfg.model.name,
        count_parameters(model) / 1e6,
        dev,
        len(data.train),
        len(data.val),
        len(data.test),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = _optimizer(model, cfg)
    steps_per_epoch = len(train_dl)
    if cfg.train.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.train.max_steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine(
            steps_per_epoch * cfg.train.epochs, steps_per_epoch * cfg.train.warmup_epochs
        ),
    )

    tracker = Tracker(cfg)
    freeze = cfg.model.name == "resnet50" and cfg.model.pretrained
    best_acc, best_epoch, bad_epochs, history = -1.0, -1, 0, []
    ckpt_path = out_dir / "model.pt"
    t0 = time.time()
    for epoch in range(cfg.train.epochs):
        if freeze:
            set_backbone_trainable(model, epoch >= cfg.train.freeze_backbone_epochs)
        tr = train_one_epoch(
            model,
            train_dl,
            criterion,
            optimizer,
            scheduler,
            dev,
            amp=cfg.train.amp,
            max_steps=cfg.train.max_steps_per_epoch,
        )
        probs, y = predict_loader(model, val_dl, dev)
        val_loss = nn.functional.nll_loss(
            torch.log(torch.from_numpy(probs) + 1e-12), torch.from_numpy(y)
        ).item()
        row = {
            "epoch": epoch + 1,
            "train_loss": tr["loss"],
            "train_acc": tr["accuracy"],
            "val_loss": val_loss,
            "val_acc": topk_accuracy(probs, y, 1),
            "lr": optimizer.param_groups[-1]["lr"],
            "elapsed_s": time.time() - t0,
        }
        history.append(row)
        tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)
        log.info(
            "epoch %d/%d train_loss=%.3f train_acc=%.3f val_loss=%.3f val_acc=%.3f",
            epoch + 1,
            cfg.train.epochs,
            row["train_loss"],
            row["train_acc"],
            row["val_loss"],
            row["val_acc"],
        )
        if row["val_acc"] > best_acc:
            best_acc, best_epoch, bad_epochs = row["val_acc"], epoch + 1, 0
            save_checkpoint(
                ckpt_path,
                model,
                cfg.model,
                data.class_names,
                cfg.data.img_size,
                extra={"val_acc": best_acc, "epoch": best_epoch},
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                log.info("early stopping at epoch %d", epoch + 1)
                break

    summary = {
        "run_name": cfg.run_name,
        "model": cfg.model.name,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_time_s": time.time() - t0,
        "device": str(dev),
        "params": count_parameters(model),
        "checkpoint": str(ckpt_path),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    tracker.set_summary({"best_val_acc": best_acc, "best_epoch": best_epoch})
    tracker.log_artifact(out_dir / "history.json")
    tracker.close()
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=None, help="YAML config file")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("overrides", nargs="*", help="section.key=value overrides")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run(load_config(args.config, args.overrides), device=args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
