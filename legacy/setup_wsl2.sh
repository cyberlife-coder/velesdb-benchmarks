#!/bin/bash
# =============================================================================
# VelesDB vs ClickHouse — WSL2 Fair Benchmark Setup
# =============================================================================
# Run: wsl -d Ubuntu -- bash /mnt/d/Projets-dev/Benchs/velesdb_vs/setup_wsl2.sh
# =============================================================================
set -e

echo "=== WSL2 Fair Benchmark Setup ==="

# 1. System packages
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv curl build-essential pkg-config libssl-dev cmake > /dev/null 2>&1
echo "  Done."

# 2. Install Rust
echo "[2/5] Installing Rust..."
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "  Rust already installed: $(rustc --version)"
fi

# 3. Install ClickHouse server
echo "[3/5] Installing ClickHouse..."
if ! command -v clickhouse-server &>/dev/null; then
    sudo apt-get install -y -qq apt-transport-https ca-certificates curl gnupg > /dev/null 2>&1
    curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | sudo gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg 2>/dev/null || true
    ARCH=$(dpkg --print-architecture)
    echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] https://packages.clickhouse.com/deb stable main" | sudo tee /etc/apt/sources.list.d/clickhouse.list > /dev/null
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clickhouse-server clickhouse-client > /dev/null 2>&1
    echo "  ClickHouse installed: $(clickhouse-client --version 2>/dev/null || echo 'installed')"
else
    echo "  ClickHouse already installed: $(clickhouse-client --version)"
fi

# 4. Python venv + deps
echo "[4/5] Setting up Python venv..."
BENCH_DIR="/mnt/d/Projets-dev/Benchs/velesdb_vs"
VENV_DIR="$BENCH_DIR/.venv-wsl"

python3 -m venv "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR" --without-pip
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q 2>/dev/null
pip install clickhouse-connect numpy maturin -q

# 5. Build velesdb-python
echo "[5/5] Building velesdb-python for Linux..."
VELESDB_PYTHON="/mnt/d/Projets-dev/velesDB/velesdb-core/crates/velesdb-python"
if [ -d "$VELESDB_PYTHON" ]; then
    source "$HOME/.cargo/env" 2>/dev/null || true
    cd "$VELESDB_PYTHON"
    maturin develop --release
    echo "  VelesDB Python built successfully."
else
    echo "  ERROR: velesdb-python not found at $VELESDB_PYTHON"
    exit 1
fi

echo ""
echo "=== Setup complete! ==="
echo "Run the benchmark with:"
echo "  wsl -d Ubuntu -- bash -c 'source /mnt/d/Projets-dev/Benchs/velesdb_vs/.venv-wsl/bin/activate && python3 /mnt/d/Projets-dev/Benchs/velesdb_vs/bench_multicolumn.py'"
