"""FastAPI inference service.

Endpoints
---------
GET  /health        liveness + model metadata
POST /predict       multipart image -> top-k breeds
POST /explain       multipart image -> Grad-CAM overlay (PNG)
GET  /metrics       Prometheus text format: request counts and latency
"""

from __future__ import annotations

import io
import os
import time
from collections import defaultdict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, Response
from PIL import UnidentifiedImageError

from petcls import __version__
from petcls.predict import Predictor

MAX_BYTES = 10 * 1024 * 1024
app = FastAPI(title="Pet Breed Classifier", version=__version__)
_predictor: Predictor | None = None
_counts: dict[str, int] = defaultdict(int)
_latency_sum: dict[str, float] = defaultdict(float)


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        path = os.environ.get("MODEL_PATH", "artifacts/model.pt")
        if not os.path.exists(path):
            raise HTTPException(503, f"Model checkpoint not found at {path}")
        _predictor = Predictor(path, device=os.environ.get("DEVICE", "cpu"))
    return _predictor


def set_predictor(p: Predictor | None) -> None:
    """Dependency-injection hook (used by tests)."""
    global _predictor
    _predictor = p


async def _read_image(file: UploadFile):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 10 MB")
    try:
        return Predictor.load_image(data)
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(415, "File is not a readable image") from e


def _record(route: str, t0: float) -> None:
    _counts[route] += 1
    _latency_sum[route] += time.perf_counter() - t0


@app.get("/health")
def health():
    p = get_predictor()
    return {
        "status": "ok",
        "model": p.model_name,
        "num_classes": len(p.class_names),
        "version": __version__,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), k: int = Query(5, ge=1, le=37)):
    t0 = time.perf_counter()
    img = await _read_image(file)
    pred = get_predictor().predict(img, k=k)
    _record("predict", t0)
    return {
        "label": pred.label,
        "confidence": pred.confidence,
        "top_k": pred.top_k,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.post("/explain")
async def explain(file: UploadFile = File(...), class_name: str | None = None):
    t0 = time.perf_counter()
    img = await _read_image(file)
    p = get_predictor()
    if class_name is not None and class_name not in p.class_names:
        raise HTTPException(422, f"Unknown class {class_name!r}")
    buf = io.BytesIO()
    p.explain(img, class_name).save(buf, format="PNG")
    _record("explain", t0)
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = ["# TYPE petcls_requests_total counter", "# TYPE petcls_latency_seconds_sum counter"]
    for route, n in _counts.items():
        lines.append(f'petcls_requests_total{{route="{route}"}} {n}')
        lines.append(f'petcls_latency_seconds_sum{{route="{route}"}} {_latency_sum[route]:.4f}')
    return "\n".join(lines) + "\n"
