# VelesDB Benchmarks — VelesDB vs Specialist Databases

Fair benchmark suite comparing [VelesDB](https://github.com/cyberlife-coder/VelesDB) (multi-model: vector + graph + columnar) against specialist databases on their home turf.

## Fairness Guarantees

- **All engines run in Docker** — same isolation, same overhead
- All accessed via HTTP/network from the same Python process
- Same dataset loaded into all engines
- Same LIMIT on both sides (equal result volume)
- Warmup rounds before measurement
- p50/p99 latency reported

## Test Environment

| Parameter | Value |
|-----------|-------|
| **CPU** | Intel Core i9-14900KF (24 cores, 32 threads, AVX2) |
| **RAM** | 64 GB DDR5 |
| **OS** | Windows 11 Pro + WSL2 Ubuntu 24.04 |
| **Storage** | NVMe SSD |
| **Runtime** | All engines in Docker containers |

### Engine Versions (pinned in docker-compose.yml)

| Engine | Image |
|--------|-------|
| **VelesDB** | Built from source (velesdb-core/Dockerfile) |
| **ClickHouse** | `clickhouse/clickhouse-server:24.12-alpine` |
| **Qdrant** | `qdrant/qdrant:v1.13.2` |
| **Memgraph** | `memgraph/memgraph:2.21.1` |

## Quick Start

```bash
# 1. Setup (Python venv + Docker build + start all engines)
bash setup.sh

# 2. Activate venv
source .venv/bin/activate

# 3. Run benchmarks
python3 bench_vector.py          # Vector search vs Qdrant (~5 min)
python3 bench_graph.py           # Graph traversal vs Memgraph (~3 min)
python3 bench_multicolumn.py     # Columnar queries vs ClickHouse (~2 min)
python3 bench_clickbench.py      # ClickBench adapted vs ClickHouse (~15 min)
python3 bench_hybrid.py          # Hybrid multi-paradigm (~5 min)
python3 bench_full_audit.py      # Quick audit (vector + graph)

# JSON output for CI/automation
python3 bench_vector.py --json > results/vector.json
```

### Manual Docker Management

```bash
# Start all engines
docker compose up -d

# Check health
docker compose ps

# Rebuild VelesDB after code changes
docker compose build velesdb
docker compose up -d velesdb

# View logs
docker compose logs velesdb
docker compose logs clickhouse

# Stop all
docker compose down

# Clean volumes (reset all data)
docker compose down -v
```

## Benchmarks

| Benchmark | VelesDB vs | What it measures |
|-----------|-----------|------------------|
| `bench_vector.py` | Qdrant | ANN search (SIFT1M), recall@k, QPS |
| `bench_graph.py` | Memgraph | BFS/DFS traversal, pattern matching |
| `bench_multicolumn.py` | ClickHouse | Multi-predicate filters, projections |
| `bench_clickbench.py` | ClickHouse | Real ClickBench queries (1M rows) |
| `bench_hybrid.py` | CH+Qdrant+igraph | Multi-paradigm hybrid queries |
| `bench_full_audit.py` | All | Quick audit across all paradigms |

## License

MIT
