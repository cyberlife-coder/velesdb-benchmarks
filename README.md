# VelesDB Benchmarks — VelesDB vs Specialist Databases

Fair benchmark suite comparing [VelesDB](https://github.com/cyberlife-coder/VelesDB) (multi-model: vector + graph + columnar) against specialist databases on their home turf.

## Test Environment

| Parameter | Value |
|-----------|-------|
| **Date** | April 5, 2026 |
| **CPU** | Intel Core i9-14900KF (24 cores, 32 threads, AVX2) |
| **RAM** | 64 GB DDR5 |
| **OS** | Windows 11 Pro + WSL2 Ubuntu 24.04 |
| **Storage** | NVMe SSD |
| **VelesDB** | 1.12.0 (develop branch, all perf optimizations merged) |
| **Qdrant** | 1.17.1 (Docker) |
| **Memgraph** | 3.9.0 (Docker) |
| **ClickHouse** | 26.3.2.3 (Docker) |
| **Python** | 3.12 (WSL2 venv) |
| **Rust** | 1.94.0 |

## Fairness Guarantees

- Same dataset loaded into all engines
- Same WSL2 environment, same Python process
- Same LIMIT on both sides (equal result volume)
- Result count verified (both engines return same number of results)
- Warmup rounds before measurement
- p50/p99 latency reported

---

## Results (VelesDB 1.12.0 — April 5, 2026)

### Vector Search (SIFT1M, 1M × 128D, Euclidean) — vs Qdrant

| Metric | VelesDB | Qdrant | Ratio |
|--------|---------|--------|-------|
| **kNN@10 p50** | **411 µs** | 10.3 ms | VelesDB **25.1x faster** |
| **kNN@100 p50** | **1.3 ms** | 8.8 ms | VelesDB **6.6x faster** |
| **Insert 1M** | 22.1K vec/s | ~15.5K vec/s | VelesDB **1.4x faster** |

#### Recall

| Mode | Recall@10 | Recall@100 | Latency p50 |
|------|-----------|------------|-------------|
| VelesDB Fast | 0.992 | 0.994 | 2.0 ms |
| VelesDB Balanced | 0.992 | 0.994 | 2.1 ms |
| VelesDB Accurate | 0.992 | 0.994 | 2.2 ms |
| Qdrant (default) | 0.998 | 0.996 | 10.3 ms |

### Graph Traversal (5K nodes, 55K edges) — vs Memgraph

| Query | VelesDB | Memgraph | Ratio | Results |
|-------|---------|----------|-------|---------|
| **BFS 1-hop** | **2 µs** | 423 µs | VelesDB **189x faster** | 10 = 10 |
| **BFS 2-hop** | **22 µs** | 3.2 ms | VelesDB **146x faster** | 110 = 110 |
| **BFS 3-hop** | **72 µs** | 2.1 ms | VelesDB **30x faster** | 200 = 200 |
| **Multi-hop** | **27 µs** | 499 µs | VelesDB **18x faster** | 10 = 10 |
| **Edge loading** | 1.03M edges/s | — | — | — |

### Columnar Queries (1M rows, 24 columns) — vs ClickHouse

*Measured on 1M-row real ClickBench dataset (Yandex.Metrica hits), 100 rounds, 20 warmup.*

| Query | VelesDB | ClickHouse | Ratio |
|-------|---------|------------|-------|
| **Q20 UserID point lookup** | **17 µs** | 2.89 ms | VelesDB **173x faster** |
| **Q21 URL LIKE '%google%'** | 35.8 ms | **8.9 ms** | ClickHouse **4.0x faster** |
| **Q24 URL pattern + 6 cols** | 21.3 ms | **6.2 ms** | ClickHouse **3.5x faster** |
| **Q37 CounterID=62 AND 3 predicates** | **32 µs** | 5.79 ms | VelesDB **181x faster** |
| **Q38 CounterID=62 AND flags** | **42 µs** | 5.28 ms | VelesDB **125x faster** |
| **Q39 CounterID=62 AND links** | **46 µs** | 5.28 ms | VelesDB **115x faster** |
| **Q41 CounterID=62 AND traffic** | **32 µs** | 5.50 ms | VelesDB **170x faster** |
| **AdvEngineID!=0 search** | **34 µs** | 6.78 ms | VelesDB **200x faster** |
| **IsMobile=1 filter** | **22 µs** | 5.88 ms | VelesDB **270x faster** |

### Improvement vs v1.11.0

| Metric | v1.11.0 | v1.12.0 | Change |
|--------|---------|---------|--------|
| vs Qdrant kNN@10 | 19.7x faster | **25.1x faster** | +27% improved |
| vs Qdrant kNN@100 | 3.6x faster | **6.6x faster** | +83% improved |
| vs Qdrant insert | 1.2x faster | **1.4x faster** | +17% improved |
| vs Memgraph BFS 1-hop | 189x faster | **189x faster** | Stable |
| vs Memgraph BFS 2-hop | 97x faster | **146x faster** | +50% improved |
| vs ClickHouse Q37 | Parity (est.) | **181x faster** | Real measurement (1M rows) |
| vs ClickHouse Q20 | Parity (est.) | **173x faster** | Real measurement (1M rows) |
| vs ClickHouse IsMobile | VelesDB faster (est.) | **270x faster** | Real measurement (1M rows) |

### Historical Improvement (v1.10.0 → v1.12.0)

| Metric | v1.10.0 | v1.12.0 | Change |
|--------|---------|---------|--------|
| vs Qdrant search | 17.7x faster | **25.1x faster** | Improved |
| vs Qdrant insert | **23x slower** | **1.4x faster** | **Reversed** |
| vs Memgraph BFS 1-hop | **100x slower** | **189x faster** | **Reversed** |
| vs Memgraph BFS 3-hop | **25,000x slower** | **30x faster** | **Reversed** |
| vs ClickHouse Q37 | **345x slower** | **181x faster** | **Reversed** |

---

## Quick Start

```bash
# Setup
bash wsl_setup_venv.sh

# Start competitors
docker run -d --name bench-memgraph -p 7687:7687 memgraph/memgraph:latest
docker run -d --name bench-qdrant -p 16333:6333 qdrant/qdrant:latest
docker run -d --name bench-clickhouse -p 8123:8123 clickhouse/clickhouse-server:latest

# Run benchmarks
source /tmp/bench-venv/bin/activate
python3 bench_full_audit.py          # Vector + Graph (~5 min)
python3 bench_graph_quick.py         # Graph only (~30s)
python3 bench_vector.py --qdrant-port 16333  # Vector only (~3 min)
python3 bench_clickbench.py --skip-ch-import  # Columnar (~15 min)
```

## License

MIT
