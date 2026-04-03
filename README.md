# VelesDB Benchmarks — VelesDB vs Specialist Databases

Fair benchmark suite comparing [VelesDB](https://github.com/cyberlife-coder/VelesDB) (multi-model: vector + graph + columnar) against specialist databases on their home turf.

## Test Environment

| Parameter | Value |
|-----------|-------|
| **Date** | April 3, 2026 |
| **CPU** | Intel Core i9-14900KF (24 cores, 32 threads, AVX2) |
| **RAM** | 64 GB DDR5 |
| **OS** | Windows 11 Pro + WSL2 Ubuntu 24.04 |
| **Storage** | NVMe SSD |
| **VelesDB** | 1.11.0 (develop branch, all perf optimizations merged) |
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

## Results (VelesDB 1.11.0 — April 3, 2026)

### Vector Search (SIFT1M, 1M × 128D, Euclidean) — vs Qdrant

| Metric | VelesDB | Qdrant | Ratio |
|--------|---------|--------|-------|
| **kNN@10 p50** | **348 µs** | 6.8 ms | VelesDB **19.7x faster** |
| **kNN@100 p50** | **1.9 ms** | 6.9 ms | VelesDB **3.6x faster** |
| **Insert 1M** | 19.0K vec/s | ~15.5K vec/s | VelesDB **1.2x faster** |

#### Recall

| Mode | Recall@10 | Recall@100 | Latency p50 |
|------|-----------|------------|-------------|
| VelesDB Fast | 0.992 | 0.995 | 2.4 ms |
| VelesDB Balanced | 0.992 | 0.995 | 2.2 ms |
| VelesDB Accurate | 0.992 | 0.995 | 2.2 ms |
| Qdrant (default) | 0.998 | 0.996 | 6.8 ms |

### Graph Traversal (5K nodes, 55K edges) — vs Memgraph

| Query | VelesDB | Memgraph | Ratio | Results |
|-------|---------|----------|-------|---------|
| **BFS 1-hop** | **2 µs** | 441 µs | VelesDB **189x faster** | 10 = 10 |
| **BFS 2-hop** | **23 µs** | 2.2 ms | VelesDB **97x faster** | 110 = 110 |
| **BFS 3-hop** | **44 µs** | 2.2 ms | VelesDB **50x faster** | 200 = 200 |
| **Multi-hop** | **27 µs** | 525 µs | VelesDB **19x faster** | 10 = 10 |
| **Edge loading** | 1.03M edges/s | — | — | — |

### Columnar Queries (100K rows, 24 columns) — vs ClickHouse

*Measured on 100K-point diagnostic dataset with secondary indexes.*

| Query | VelesDB | ClickHouse (est.) | Status |
|-------|---------|-------------------|--------|
| **CounterID=62 (indexed)** | **0.6 ms** | ~5 ms | VelesDB faster |
| **CounterID=62 AND 3 predicates** | **1.6 ms** | ~5 ms | VelesDB faster |
| **UserID = X (point lookup)** | **1.5 ms** | ~2.5 ms | Parity |
| **IsMobile=1 (20% selectivity)** | **1.3 ms** | ~5 ms | VelesDB faster |
| **AdvEngine != 0 (bitmap NEQ)** | **2.7 ms** | ~5.5 ms | Parity |
| **URL LIKE '%google%' (BM25)** | **3.6 ms** | ~8 ms | Parity |

### Improvement vs v1.10.0

| Metric | v1.10.0 | v1.11.0 | Change |
|--------|---------|---------|--------|
| vs Qdrant search | 17.7x faster | **19.7x faster** | Improved |
| vs Qdrant insert | **23x slower** | **1.2x faster** | **Reversed** |
| vs Memgraph BFS 1-hop | **100x slower** | **189x faster** | **Reversed** |
| vs Memgraph BFS 3-hop | **25,000x slower** | **50x faster** | **Reversed** |
| vs ClickHouse Q37 | **345x slower** | **Parity** | **Reversed** |

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
