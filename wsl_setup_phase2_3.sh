#!/bin/bash
# =============================================================================
# Setup Qdrant + Memgraph + Python deps for Phases 2-4 (WSL2)
# =============================================================================
set -euo pipefail

VENV_DIR="/tmp/bench-venv"
QDRANT_DIR="$HOME/qdrant-bench"
SIFT_HDF5="/tmp/sift-128-euclidean.hdf5"

echo "=== Phase 2-4 Setup ==="

# -----------------------------------------------
# 1. Qdrant binary (native Linux, no Docker)
# -----------------------------------------------
echo ""
echo "--- Qdrant ---"
if [ -f "$QDRANT_DIR/qdrant" ]; then
    echo "  Already installed at $QDRANT_DIR/qdrant"
else
    mkdir -p "$QDRANT_DIR"
    echo "  Downloading Qdrant v1.17.1..."
    wget -q --show-progress -O /tmp/qdrant.tar.gz \
        "https://github.com/qdrant/qdrant/releases/download/v1.17.1/qdrant-x86_64-unknown-linux-gnu.tar.gz"
    tar -xzf /tmp/qdrant.tar.gz -C "$QDRANT_DIR"
    chmod +x "$QDRANT_DIR/qdrant"
    rm /tmp/qdrant.tar.gz
    echo "  Installed: $QDRANT_DIR/qdrant"
fi

# -----------------------------------------------
# 2. SIFT1M dataset (ANN benchmarks standard)
# -----------------------------------------------
echo ""
echo "--- SIFT1M Dataset ---"
if [ -f "$SIFT_HDF5" ]; then
    echo "  Already at $SIFT_HDF5 ($(ls -lh $SIFT_HDF5 | awk '{print $5}'))"
else
    echo "  Downloading SIFT1M (128-dim, 1M vectors, ~501MB)..."
    wget -q --show-progress -O "$SIFT_HDF5" \
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    echo "  Downloaded: $(ls -lh $SIFT_HDF5 | awk '{print $5}')"
fi

# -----------------------------------------------
# 3. Memgraph (native .deb package)
# -----------------------------------------------
echo ""
echo "--- Memgraph ---"
if command -v memgraph &>/dev/null || dpkg -l memgraph 2>/dev/null | grep -q "^ii"; then
    echo "  Already installed"
else
    echo "  Downloading Memgraph..."
    UBUNTU_VER=$(lsb_release -rs)
    if [[ "$UBUNTU_VER" == "24.04" ]]; then
        MG_URL="https://download.memgraph.com/memgraph/v3.1.1/ubuntu-24.04/memgraph_3.1.1-1_amd64.deb"
    else
        MG_URL="https://download.memgraph.com/memgraph/v3.1.1/ubuntu-22.04/memgraph_3.1.1-1_amd64.deb"
    fi
    wget -q --show-progress -O /tmp/memgraph.deb "$MG_URL"
    echo "  Installing (requires sudo)..."
    sudo dpkg -i /tmp/memgraph.deb || sudo apt-get install -f -y
    rm /tmp/memgraph.deb
    echo "  Installed"
fi

# -----------------------------------------------
# 4. Python deps (into existing venv)
# -----------------------------------------------
echo ""
echo "--- Python deps ---"
if [ ! -d "$VENV_DIR" ]; then
    echo "  ERROR: venv not found at $VENV_DIR"
    echo "  Run wsl_setup_venv.sh first"
    exit 1
fi

source "$VENV_DIR/bin/activate"
pip install h5py qdrant-client neo4j 2>&1 | tail -5

echo ""
echo "--- Verifying ---"
python3 -c "
import h5py, qdrant_client, neo4j, velesdb
print(f'h5py OK, qdrant-client OK, neo4j OK, velesdb {velesdb.__version__} OK')
"

echo ""
echo "=== Setup complete ==="
echo "  Qdrant:    $QDRANT_DIR/qdrant"
echo "  SIFT1M:    $SIFT_HDF5"
echo "  Memgraph:  $(dpkg -l memgraph 2>/dev/null | grep memgraph | awk '{print $3}' || echo 'check install')"
echo "  Venv:      $VENV_DIR"
