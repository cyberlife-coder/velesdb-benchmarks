#!/bin/bash
# =============================================================================
# Run the fair benchmark in WSL2
# Both engines on the same OS, same Python, same localhost
# =============================================================================
set -e

CH_BIN="$HOME/clickhouse-bench/clickhouse"
VENV_DIR="/tmp/bench-venv"
BENCH_DIR="/mnt/d/Projets-dev/Benchs/velesdb_vs"

# 1. Start ClickHouse server in background
echo "=== Starting ClickHouse server ==="
mkdir -p /tmp/ch-data /tmp/ch-log

# Kill any existing clickhouse-server
pkill -f "clickhouse-server" 2>/dev/null || true
sleep 1

"$CH_BIN" server \
    -L /tmp/ch-log/clickhouse.log \
    -E /tmp/ch-log/clickhouse-err.log \
    -- --path=/tmp/ch-data/ --http_port=8123 --tcp_port=9000 --listen_host=127.0.0.1 &
CH_PID=$!
echo "  ClickHouse PID: $CH_PID"

# Wait for server to be ready
echo "  Waiting for server..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8123/ping >/dev/null 2>&1; then
        echo "  Server ready!"
        break
    fi
    sleep 1
done

# Verify
CH_VER=$(curl -s "http://127.0.0.1:8123/?query=SELECT+version()")
echo "  ClickHouse version: $CH_VER"

# 2. Run benchmark
echo ""
echo "=== Running Benchmark ==="
source "$VENV_DIR/bin/activate"
python3 "$BENCH_DIR/bench_multicolumn.py" "$@"

# 3. Cleanup
echo ""
echo "=== Stopping ClickHouse ==="
kill $CH_PID 2>/dev/null || true
wait $CH_PID 2>/dev/null || true
echo "  Done."
