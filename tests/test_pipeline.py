"""End-to-end smoke test: train -> checkpoint -> evaluate on synthetic data (no downloads)."""

import json

from petcls.config import load_config
from petcls.evaluate import evaluate
from petcls.train import run


def test_train_and_evaluate(tmp_path):
    cfg = load_config("configs/smoke.yaml", [f"output_dir={tmp_path}"])
    summary = run(cfg, device="cpu")
    ckpt = tmp_path / "smoke" / "model.pt"
    assert ckpt.exists()
    assert summary["epochs_run"] == 2
    history = json.loads((tmp_path / "smoke" / "history.json").read_text())
    assert {"train_loss", "val_acc", "lr"} <= set(history[0])

    report = evaluate(
        str(ckpt),
        str(tmp_path / "eval"),
        dataset="fake",
        num_workers=0,
        device="cpu",
        fake_size=cfg.data.fake_size,
    )
    assert 0.0 <= report["top1_accuracy"] <= 1.0
    for f in ["metrics.json", "confusion_matrix.png", "reliability.png", "per_class_accuracy.png"]:
        assert (tmp_path / "eval" / f).exists()
