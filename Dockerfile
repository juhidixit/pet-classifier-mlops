# syntax=docker/dockerfile:1.7
# CPU inference image for the FastAPI service (~900 MB with CPU-only torch).
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/model.pt DEVICE=cpu OMP_NUM_THREADS=2

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY serve ./serve
RUN pip install --no-deps .

# The model is copied in at build time (`make docker` puts the best checkpoint here).
COPY artifacts/model.pt /app/artifacts/model.pt

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
