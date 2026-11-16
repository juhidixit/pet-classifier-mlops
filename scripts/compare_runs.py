"""Build the README results table and learning-curve figure from finished runs.

python scripts/compare_runs.py --runs resnet50 scratch_cnn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--reports", default="reports")
    a = p.parse_args()

    rows = [
        "| Model | Params | Test top-1 | Test top-5 | Macro-F1 | ECE | Cat/dog acc. | "
        "Epochs | Train time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in a.runs:
        summ = json.loads((Path(a.artifacts) / run / "train_summary.json").read_text())
        hist = json.loads((Path(a.artifacts) / run / "history.json").read_text())
        m = json.loads((Path(a.reports) / run / "metrics.json").read_text())
        rows.append(
            f"| {run} | {summ['params'] / 1e6:.1f}M | **{m['top1_accuracy']:.1%}** | "
            f"{m['top5_accuracy']:.1%} | {m['macro_f1']:.3f} | {m['ece']:.3f} | "
            f"{m['species']['species_accuracy']:.1%} | {summ['epochs_run']} | "
            f"{summ['train_time_s'] / 60:.0f} min ({summ['device']}) |"
        )
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], "--", label=f"{run} train")
        axes[0].plot(ep, [h["val_loss"] for h in hist], label=f"{run} val")
        axes[1].plot(ep, [h["train_acc"] for h in hist], "--", label=f"{run} train")
        axes[1].plot(ep, [h["val_acc"] for h in hist], label=f"{run} val")
    axes[0].set(title="Loss", xlabel="epoch")
    axes[1].set(title="Accuracy", xlabel="epoch", ylim=(0, 1))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(a.reports).mkdir(exist_ok=True)
    fig.savefig(Path(a.reports) / "learning_curves.png", dpi=130)
    table = "\n".join(rows)
    (Path(a.reports) / "results_table.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
