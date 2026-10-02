#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != "Linux" ]; then echo 'Use Linux or WSL. This worker must remain separate from the Windows engine.'; exit 1; fi
PYTHON_BIN="${EDGELENS_WORKER_PYTHON:-python3.11}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Verified worker requires Python 3.11"'
"$PYTHON_BIN" -m venv .venv-tflite
.venv-tflite/bin/python -m pip install -r backend/requirements-tflite-linux-tested.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv-tflite/bin/python -m pip check
.venv-tflite/bin/python -c 'import torch,litert_torch,ai_edge_litert; print("Worker imports passed; run the conversion smoke test before using your own model.")'
