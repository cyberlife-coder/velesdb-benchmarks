#!/bin/bash
# =============================================================================
# VelesDB vs Qdrant — ANN Vector Search Benchmark (WSL2)
# =============================================================================
set -euo pipefail
trap 'pkill -f "qdrant" 2>/dev/null || true' EXIT

QDRANT_BIN="$HOME/qdrant-bench/qdrant"
VENV_DIR="/tmp/bench-venv"
BENCH_DIR="/mnt/d/Projets-dev/Benchs/velesdb_vs"
SIFT_HDF5="/tmp/sift-128-euclidean.hdf5"
QDRANT_DATA="$HOME/qdrant-bench/storage"

# 0. Check prerequisites
if [ ! -f "$QDRANT_BIN" ]; then
    echo "ERROR: Qdrant not found at $QDRANT_BIN"
    echo "Run: bash wsl_setup_phase2_3.sh"
    exit 1
fi
if [ ! -f "$SIFT_HDF5" ]; then
    echo "ERROR: SIFT1M not found at $SIFT_HDF5"
    echo "Run: bash wsl_setup_phase2_3.sh"
    exit 1
fi

# 1. Start Qdrant server
echo "=== Starting Qdrant server ==="
pkill -f "qdrant" 2>/dev/null || true
sleep 1

mkdir -p "$QDRANT_DATA"
cd "$HOME/qdrant-bench"
QDRANT__STORAGE__STORAGE_PATH="$QDRANT_DATA" "$QDRANT_BIN" &
QDRANT_PID=$!

echo "  Waiting for Qdrant..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:6333/healthz >/dev/null 2>&1; then
        echo "  Qdrant ready! (PID $QDRANT_PID)"
        break
    fi
    sleep 1
done

# 2. Run benchmark
echo ""
echo "=== Running Vector Benchmark ==="
source "$VENV_DIR/bin/activate"
python3 "$BENCH_DIR/bench_vector.py" --sift "$SIFT_HDF5" "$@"

# 3. Cleanup (handled by trap)
echo ""
echo "  Done."
