"""Thin experiment-tracking facade over MLflow and Weights & Biases.

Both backends are optional: if a library is missing or disabled the tracker becomes a no-op,
so training never crashes because of tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._mlflow = None
        self._wandb = None
        t = cfg.tracking
        if t.mlflow:
            try:
                import mlflow

                mlflow.set_tracking_uri(t.mlflow_uri)
                mlflow.set_experiment(t.experiment)
                mlflow.start_run(run_name=cfg.run_name)
                mlflow.log_params(cfg.flat())
                self._mlflow = mlflow
            except Exception as e:  # pragma: no cover - depends on environment
                log.warning("MLflow disabled: %s", e)
        if t.wandb:
            try:
                import wandb

                # When launched by `wandb agent`, sweep params arrive via wandb.config.
                self._wandb = wandb.init(
                    project=t.wandb_project, name=cfg.run_name, config=cfg.to_dict(), reinit=True
                )
            except Exception as e:  # pragma: no cover
                log.warning("W&B disabled: %s", e)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)
        if self._wandb:
            self._wandb.log(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow:
            self._mlflow.log_artifact(str(path))
        if self._wandb:
            self._wandb.save(str(path), policy="now")

    def set_summary(self, summary: dict[str, Any]) -> None:
        if self._mlflow:
            self._mlflow.log_metrics(
                {k: v for k, v in summary.items() if isinstance(v, (int, float))}
            )
        if self._wandb:
            self._wandb.summary.update(summary)

    def close(self) -> None:
        if self._mlflow:
            self._mlflow.end_run()
        if self._wandb:
            self._wandb.finish()
