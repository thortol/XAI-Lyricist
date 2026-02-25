FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/usr/local/share/nltk_data \
    XAI_DEVICE=cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    espeak-ng \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt /app/requirements-api.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements-api.txt

RUN python - <<'PY'
import os
import nltk

path = os.environ["NLTK_DATA"]
os.makedirs(path, exist_ok=True)
for pkg in ("punkt", "punkt_tab"):
    nltk.download(pkg, download_dir=path, quiet=True)
PY

COPY api /app/api
COPY models /app/models
COPY utils /app/utils
COPY configs /app/configs
COPY binary /app/binary
COPY bestM2LCkpt.pt /app/bestM2LCkpt.pt

EXPOSE 10000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
