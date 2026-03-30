#!/bin/bash
# =============================================================================
# VelesDB vs ClickHouse — WSL2 Setup (NO SUDO NEEDED)
# =============================================================================
# Uses the already-downloaded clickhouse binary + rustup (user-level)
# Run: wsl -d Ubuntu -- bash /mnt/d/Projets-dev/Benchs/velesdb_vs/setup_wsl2_nosudo.sh
# =============================================================================
set -e

BENCH_DIR="/mnt/d/Projets-dev/Benchs/velesdb_vs"
VELESDB_PYTHON="/mnt/d/Projets-dev/velesDB/velesdb-core/crates/velesdb-python"
VENV_DIR="$BENCH_DIR/.venv-wsl"
CH_DIR="$HOME/clickhouse-bench"

echo "=== WSL2 Fair Benchmark Setup (no sudo) ==="

# 1. ClickHouse binary (already downloaded, just organize)
echo "[1/4] Setting up ClickHouse binary..."
mkdir -p "$CH_DIR"
if [ -f "$HOME/clickhouse" ]; then
    mv "$HOME/clickhouse" "$CH_DIR/clickhouse" 2>/dev/null || true
fi
if [ -f "$CH_DIR/clickhouse" ]; then
    chmod +x "$CH_DIR/clickhouse"
    echo "  ClickHouse binary ready: $($CH_DIR/clickhouse local --version 2>/dev/null | head -1)"
else
    echo "  Downloading ClickHouse..."
    curl -sSL https://clickhouse.com/ | bash -s -- --dir="$CH_DIR"
    chmod +x "$CH_DIR/clickhouse"
fi

# 2. Install Rust (user-level, no sudo)
echo "[2/4] Installing Rust..."
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
    echo "  Rust installed: $(rustc --version)"
else
    source "$HOME/.cargo/env" 2>/dev/null || true
    echo "  Rust already installed: $(rustc --version)"
fi

# 3. Python venv + deps
echo "[3/4] Setting up Python environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install clickhouse-connect numpy maturin -q
echo "  Python venv ready at $VENV_DIR"

# 4. Build velesdb-python for Linux
echo "[4/4] Building velesdb-python..."
if [ -d "$VELESDB_PYTHON" ]; then
    cd "$VELESDB_PYTHON"
    source "$HOME/.cargo/env"
    source "$VENV_DIR/bin/activate"
    maturin develop --release 2>&1
    echo "  velesdb-python built."
else
    echo "  ERROR: $VELESDB_PYTHON not found"
    exit 1
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run the benchmark:"
echo "  1. Start ClickHouse server (in one terminal):"
echo "     wsl -d Ubuntu -- $CH_DIR/clickhouse server --config-file=/dev/null"
echo ""
echo "  2. Run benchmark (in another terminal):"
echo "     wsl -d Ubuntu -- bash -c 'source $VENV_DIR/bin/activate && python3 $BENCH_DIR/bench_multicolumn.py'"
