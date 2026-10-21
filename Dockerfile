# PrintGuard AI server image (CPU, torch-free ONNX runtime path).
# Model weights are mounted at runtime (./weights), never baked into the image.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY printguard ./printguard
COPY server ./server
COPY dashboard ./dashboard
COPY configs ./configs

RUN useradd --create-home --uid 1000 printguard && mkdir -p /app/data /app/weights && chown -R printguard /app
USER printguard
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "-m", "server", "--config", "configs/server.docker.yaml"]
