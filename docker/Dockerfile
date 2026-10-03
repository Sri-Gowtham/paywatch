# Build from the repo root:  docker build -f docker/Dockerfile -t paywatch-api .
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

FROM python:3.11-slim
# libgomp1: OpenMP runtime required by the xgboost wheel
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/api ./src/api
COPY models/compact ./models/compact
ENV PAYWATCH_MODELS_DIR=/app/models/compact \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# ONE worker: user-behaviour and entity-history state live in process memory
CMD ["uvicorn", "--factory", "src.api.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
