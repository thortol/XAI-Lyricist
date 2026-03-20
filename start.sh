#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
NLTK_DATA_DIR="${NLTK_DATA_DIR:-$ROOT_DIR/.nltk_data}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python executable '$PYTHON_BIN' not found." >&2
  exit 1
fi

if [ ! -f "requirements-api.txt" ]; then
  echo "Error: requirements-api.txt is missing." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[start.sh] Creating virtualenv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[start.sh] Upgrading pip/setuptools/wheel"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo "[start.sh] Installing backend dependencies"
"$PYTHON_BIN" -m pip install -r requirements-api.txt

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NLTK_DATA="$NLTK_DATA_DIR${NLTK_DATA:+:$NLTK_DATA}"

echo "[start.sh] Ensuring NLTK data is available at $NLTK_DATA_DIR"
"$PYTHON_BIN" - <<PY || true
import os
import ssl
import nltk

# Fix for "unable to get local issuer certificate" on Mac
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

os.makedirs(os.environ["NLTK_DATA"], exist_ok=True)
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.download(pkg, download_dir=os.environ["NLTK_DATA"], quiet=True)
        print(f"[start.sh] Success: {pkg} downloaded.")
    except Exception as e:
        print(f"[start.sh] Warning: nltk.download({pkg!r}) failed: {e}")
PY

echo "[start.sh] Starting API at http://$HOST:$PORT"
exec "$PYTHON_BIN" -m uvicorn api.main:app --host "$HOST" --port "$PORT" --reload
