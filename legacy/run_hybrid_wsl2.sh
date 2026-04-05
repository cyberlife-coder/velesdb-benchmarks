#!/bin/bash
# =============================================================================
# VelesDB Hybrid vs CH+Qdrant+Memgraph — Multi-Paradigm Benchmark (WSL2)
# All 3 engines must be running simultaneously.
# =============================================================================
set -euo pipefail
trap 'pkill -f "clickhouse-server" 2>/dev/null; pkill -f "qdrant" 2>/dev/null; true' EXIT

CH_BIN="$HOME/clickhouse-bench/clickhouse"
QDRANT_BIN="$HOME/qdrant-bench/qdrant"
QDRANT_DATA="$HOME/qdrant-bench/storage"
VENV_DIR="/tmp/bench-venv"
BENCH_DIR="/mnt/d/Projets-dev/Benchs/velesdb_vs"

# 1. Start ClickHouse
echo "=== Starting ClickHouse ==="
pkill -f "clickhouse-server" 2>/dev/null || true
sleep 1
mkdir -p "$HOME/ch-bench-data/tmp" "$HOME/ch-bench-data/log"
"$CH_BIN" server \
    -L "$HOME/ch-bench-data/log/clickhouse.log" \
    -E "$HOME/ch-bench-data/log/clickhouse-err.log" \
    -- --path="$HOME/ch-bench-data/" --http_port=8123 --tcp_port=9000 \
       --listen_host=127.0.0.1 --user_files_path=/tmp/ \
       --tmp_path="$HOME/ch-bench-data/tmp/" &
CH_PID=$!
echo "  Waiting..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8123/ping >/dev/null 2>&1; then
        echo "  ClickHouse ready! (PID $CH_PID)"
        break
    fi
    sleep 1
done

# 2. Start Qdrant
echo ""
echo "=== Starting Qdrant ==="
pkill -f "qdrant" 2>/dev/null || true
sleep 1
mkdir -p "$QDRANT_DATA"
cd "$HOME/qdrant-bench"
QDRANT__STORAGE__STORAGE_PATH="$QDRANT_DATA" "$QDRANT_BIN" &
QDRANT_PID=$!
echo "  Waiting..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:6333/healthz >/dev/null 2>&1; then
        echo "  Qdrant ready! (PID $QDRANT_PID)"
        break
    fi
    sleep 1
done

# 3. Ensure Memgraph is running
echo ""
echo "=== Ensuring Memgraph ==="
if sudo systemctl is-active --quiet memgraph 2>/dev/null; then
    echo "  Already running"
else
    sudo systemctl start memgraph
    sleep 2
    echo "  Started"
fi

# 4. Run benchmark
echo ""
echo "=== Running Hybrid Benchmark ==="
source "$VENV_DIR/bin/activate"
python3 "$BENCH_DIR/bench_hybrid.py" "$@"

# 5. Cleanup (handled by trap)
echo ""
echo "  Done."
