#!/bin/bash
# =============================================================================
# VelesDB vs Memgraph — Graph Traversal Benchmark (WSL2)
# =============================================================================
set -euo pipefail

VENV_DIR="/tmp/bench-venv"
BENCH_DIR="/mnt/d/Projets-dev/Benchs/velesdb_vs"

# 1. Ensure Memgraph is running
echo "=== Starting Memgraph ==="
if sudo systemctl is-active --quiet memgraph 2>/dev/null; then
    echo "  Already running"
else
    sudo systemctl start memgraph
    echo "  Started"
fi

# Wait for Memgraph to be ready
echo "  Waiting for Memgraph..."
for i in $(seq 1 15); do
    if echo "RETURN 1;" | mgconsole --host 127.0.0.1 --port 7687 >/dev/null 2>&1; then
        echo "  Memgraph ready!"
        break
    fi
    # Alternative check via bolt
    if python3 -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://127.0.0.1:7687', auth=('', ''))
d.verify_connectivity()
d.close()
print('ok')
" 2>/dev/null | grep -q ok; then
        echo "  Memgraph ready! (via bolt)"
        break
    fi
    sleep 1
done

# 2. Run benchmark
echo ""
echo "=== Running Graph Benchmark ==="
source "$VENV_DIR/bin/activate"
python3 "$BENCH_DIR/bench_graph.py" "$@"

echo ""
echo "  Done."
