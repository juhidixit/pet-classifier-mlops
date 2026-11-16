import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from petcls.predict import Predictor
from serve import app as app_module


@pytest.fixture()
def client(tiny_checkpoint):
    app_module.set_predictor(Predictor(str(tiny_checkpoint)))
    yield TestClient(app_module.app)
    app_module.set_predictor(None)


def _jpeg(w=120, h=90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["num_classes"] == 37


def test_predict_topk(client):
    r = client.post("/predict?k=3", files={"file": ("pet.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["top_k"]) == 3
    probs = [p["probability"] for p in body["top_k"]]
    assert probs == sorted(probs, reverse=True)
    assert body["label"] == body["top_k"][0]["label"]


def test_explain_returns_png(client):
    r = client.post("/explain", files={"file": ("pet.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (64, 64)


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 415


def test_rejects_unknown_class(client):
    r = client.post(
        "/explain?class_name=Dragon", files={"file": ("pet.jpg", _jpeg(), "image/jpeg")}
    )
    assert r.status_code == 422


def test_metrics_counts_requests(client):
    client.post("/predict", files={"file": ("pet.jpg", _jpeg(), "image/jpeg")})
    assert 'petcls_requests_total{route="predict"}' in client.get("/metrics").text
