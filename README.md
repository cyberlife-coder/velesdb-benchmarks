# VelesDB Benchmarks — VelesDB vs Specialist Databases

Fair benchmark suite comparing [VelesDB](https://github.com/cyberlife-coder/VelesDB) (multi-model: vector + graph + columnar) against specialist databases on their home turf.

## Phases

| Phase | File | VelesDB vs | Workload |
|-------|------|------------|----------|
| 1 | `bench_clickbench.py` | **ClickHouse** | ClickBench 1M rows, multi-predicate dashboard queries |
| 2 | `bench_vector.py` | **Qdrant** | SIFT1M (1M x 128-dim), kNN recall@k + latency |
| 3 | `bench_graph.py` | **Memgraph** | 50K-node social network, BFS/DFS/MATCH traversal |
| 4 | `bench_hybrid.py` | **CH + Qdrant + igraph** | Multi-paradigm queries (vector+graph+columnar) |

## Fairness Guarantees

- Same dataset loaded into all engines
- Same WSL2 environment, same Python process
- Same LIMIT on both sides (equal result volume)
- Warmup rounds before measurement
- p50/p99/mean/min latency reported

## Results Summary (VelesDB 1.10.0)

### Phase 1 — Columnar (ClickBench)
- ClickHouse **22-345x faster** for dashboard queries with rare predicates
- Root cause: HNSW post-filtering at low selectivity ([#487](https://github.com/cyberlife-coder/VelesDB/issues/487))

### Phase 2 — Vector (SIFT1M)
- VelesDB **17.7x faster** at kNN@10 (138 us vs 2.44 ms)
- VelesDB **6.4x faster** at kNN@100 (443 us vs 2.85 ms)
- Qdrant **23x faster** at insertion ([#488](https://github.com/cyberlife-coder/VelesDB/issues/488))

### Phase 3 — Graph (Social Network)
- Memgraph **100-25000x faster** at graph traversal
- Root cause: Vec allocations + no adjacency index ([#491](https://github.com/cyberlife-coder/VelesDB/issues/491))

### Phase 4 — Hybrid (Multi-Paradigm)
- VelesDB **5x faster** on Q1 (vector + graph 1-hop) — unified engine advantage
- Combined stack **1.9-9.4x faster** on Q2-Q5 (filtered queries, deep graph)

## Related Issues

| Issue | Description |
|-------|-------------|
| [#486](https://github.com/cyberlife-coder/VelesDB/issues/486) | VelesQL uint64 parse error |
| [#487](https://github.com/cyberlife-coder/VelesDB/issues/487) | HNSW post-filtering 200-345x slowdown |
| [#488](https://github.com/cyberlife-coder/VelesDB/issues/488) | Insertion 23x slower than Qdrant at 1M scale |
| [#489](https://github.com/cyberlife-coder/VelesDB/issues/489) | match_query projected always empty |
| [#490](https://github.com/cyberlife-coder/VelesDB/issues/490) | traverse_bfs API inconsistencies |
| [#491](https://github.com/cyberlife-coder/VelesDB/issues/491) | Graph traversal 100-25000x slower |
| [#492](https://github.com/cyberlife-coder/VelesDB/issues/492) | VelesQL IN operator missing |

## Quick Start (WSL2)

### Prerequisites

- WSL2 with Ubuntu 22.04 or 24.04
- Rust toolchain (`rustup`)
- Python 3.10+
- VelesDB source at `/mnt/d/Projets-dev/velesDB/velesdb-core/`

### Setup

```bash
# 1. Create venv + build velesdb-python
bash wsl_setup_venv.sh

# 2. Install Qdrant + SIFT1M + Memgraph + Python deps
bash wsl_setup_phase2_3.sh
```

### Run Benchmarks

```bash
# Phase 1: VelesDB vs ClickHouse
bash run_clickbench_wsl2.sh --rounds 50 --warmup 10

# Phase 2: VelesDB vs Qdrant
bash run_vector_wsl2.sh --rounds 30 --warmup 5

# Phase 3: VelesDB vs Memgraph
bash run_graph_wsl2.sh --rounds 50 --warmup 10

# Phase 4: Hybrid (all engines must be running)
bash run_hybrid_wsl2.sh --rounds 30 --warmup 5
```

All benchmarks support `--json` for machine-readable output.

## Environment

- **OS**: Windows 11 Pro + WSL2 Ubuntu
- **CPU**: Intel i9-14900KF
- **RAM**: 64 GB DDR5
- **Storage**: NVMe SSD
- **VelesDB**: 1.10.0
- **ClickHouse**: 26.4.1.392
- **Qdrant**: 1.17.1
- **Memgraph**: 3.1.1

## License

MIT
