#!/bin/bash
# =============================================================================
# VelesDB Benchmarks — Unified Setup
# =============================================================================
# All engines run in Docker. This script only sets up the Python bench env.
#
# Usage:
#   bash setup.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
SIFT_HDF5="/tmp/sift-128-euclidean.hdf5"

echo "=== VelesDB Benchmarks — Setup ==="
echo ""

# -----------------------------------------------
# 1. Python venv
# -----------------------------------------------
echo "[1/3] Setting up Python venv..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR" 2>/dev/null || python3 -m venv --without-pip "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

if ! command -v pip &>/dev/null; then
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py 2>&1 | tail -3
fi

pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q
echo "  Venv ready at $VENV_DIR"

# -----------------------------------------------
# 2. SIFT1M dataset (for vector benchmark)
# -----------------------------------------------
echo "[2/3] Checking SIFT1M dataset..."
if [ -f "$SIFT_HDF5" ]; then
    echo "  Already at $SIFT_HDF5 ($(ls -lh $SIFT_HDF5 | awk '{print $5}'))"
else
    echo "  Downloading SIFT1M (~501MB)..."
    wget -q --show-progress -O "$SIFT_HDF5" \
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    echo "  Downloaded: $(ls -lh $SIFT_HDF5 | awk '{print $5}')"
fi

# -----------------------------------------------
# 3. Build & start Docker containers
# -----------------------------------------------
echo "[3/3] Building and starting Docker containers..."
cd "$SCRIPT_DIR"
docker compose build velesdb
docker compose up -d
echo "  Waiting for all services to be healthy..."
sleep 5

# Check health
for svc in velesdb clickhouse qdrant memgraph; do
    container="bench-$svc"
    if docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null | grep -q healthy; then
        echo "  ✓ $svc healthy"
    else
        echo "  ⏳ $svc starting (check: docker compose logs $svc)"
    fi
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 bench_vector.py"
echo "  python3 bench_graph.py"
echo "  python3 bench_hybrid.py"
echo "  python3 bench_clickbench.py"
echo "  python3 bench_full_audit.py"
