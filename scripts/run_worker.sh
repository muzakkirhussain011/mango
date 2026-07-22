#!/usr/bin/env bash
# Reliable LOCAL FedGMA R&D worker (macOS Apple Silicon / Linux / any machine with the repo).
# Runs forever: pulls latest code + job queue, runs queued experiments, pushes results back to
# GitHub for the Claude loop to read. No Colab, no disconnects.
#
# One-time: create a GitHub fine-grained PAT (repo muzakkirhussain011/mango, Contents: read+write).
#
# Usage (from the repo root or anywhere):
#   export GH_TOKEN=<your token>          # so results can be pushed back (never share this)
#   bash scripts/run_worker.sh            # CPU — most reliable for these tiny models
#   WORKER_DEVICE=mps bash scripts/run_worker.sh   # Apple GPU (faster; auto-falls back if unsupported)
#
# Stop with Ctrl-C. Re-run anytime; it skips already-finished jobs.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${GH_TOKEN:-}" ]; then
  echo "[run_worker] WARNING: GH_TOKEN is not set — the worker will run but CANNOT push results."
  echo "[run_worker]   Set it first:  export GH_TOKEN=your_token"
fi

: "${WORKER_DEVICE:=cpu}"     # default CPU for max reliability; override with WORKER_DEVICE=mps
export WORKER_DEVICE

PY="${PYTHON:-python3}"
echo "[run_worker] installing deps (once)…"
"$PY" -m pip install -e . -q
echo "[run_worker] starting worker on device=$WORKER_DEVICE (Ctrl-C to stop)…"
exec "$PY" scripts/colab_worker.py
