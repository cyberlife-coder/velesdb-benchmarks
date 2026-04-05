#!/bin/bash
set -e

VENV_DIR="/tmp/bench-venv"
rm -rf "$VENV_DIR"

# Create venv without pip (Ubuntu 24.04 PEP 668 issue)
python3 -m venv --without-pip "$VENV_DIR"

# Bootstrap pip inside the venv
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
"$VENV_DIR/bin/python3" /tmp/get-pip.py 2>&1 | tail -3

echo "--- pip version ---"
"$VENV_DIR/bin/pip" --version

echo "--- Installing deps ---"
"$VENV_DIR/bin/pip" install clickhouse-connect numpy maturin 2>&1 | tail -5

echo "--- Verifying ---"
"$VENV_DIR/bin/python3" -c "import clickhouse_connect; import numpy; print('deps OK')"

echo "--- Building velesdb-python ---"
source "$HOME/.cargo/env" 2>/dev/null || true
VELESDB_ROOT="/mnt/d/Projets-dev/velesDB/velesdb-core"
VELESDB_PY="$VELESDB_ROOT/crates/velesdb-python"

# Temporarily rename Windows .venv so maturin doesn't find it
if [ -d "$VELESDB_ROOT/.venv" ]; then
    mv "$VELESDB_ROOT/.venv" "$VELESDB_ROOT/.venv-win-backup"
    echo "  (Renamed .venv to .venv-win-backup)"
fi

cd "$VELESDB_PY"
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
"$VENV_DIR/bin/maturin" develop --release 2>&1 | tail -15
BUILD_RC=$?

# Restore Windows .venv
if [ -d "$VELESDB_ROOT/.venv-win-backup" ]; then
    mv "$VELESDB_ROOT/.venv-win-backup" "$VELESDB_ROOT/.venv"
    echo "  (Restored .venv)"
fi

if [ $BUILD_RC -ne 0 ]; then
    echo "BUILD FAILED"
    exit 1
fi

echo "--- Verifying velesdb ---"
"$VENV_DIR/bin/python3" -c "import velesdb; print(f'velesdb {velesdb.__version__} OK')"

echo ""
echo "=== DONE ==="
echo "Venv at: $VENV_DIR"
