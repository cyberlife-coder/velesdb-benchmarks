#!/usr/bin/env python3
"""
VelesDB vs Qdrant — ANN Vector Search Benchmark
=================================================

Uses SIFT1M from ann-benchmarks.com (1M vectors, 128-dim, euclidean).
Standard ANN benchmark with ground truth for recall measurement.

Metrics:
  - Recall@k (vs exact ground truth)
  - Latency: p50, p99, mean, min
  - QPS (queries per second)

Fairness:
  - Same dataset, same queries, same machine
  - Same WSL2 environment, same Python process
  - Both engines use HNSW index
"""

import argparse
import json
import os
import shutil
import statistics
import sys
import time

import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py not installed (pip install h5py)")
    sys.exit(1)

try:
    import velesdb
except ImportError:
    print("ERROR: velesdb not installed")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("ERROR: qdrant-client not installed (pip install qdrant-client)")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_ROUNDS = 10
MEASURE_ROUNDS = 50
BATCH_SIZE = 10000
SIFT_PATH = "/tmp/sift-128-euclidean.hdf5"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} \u00b5s"
    return f"{seconds * 1_000:.2f} ms"


def recall_at_k(predicted_ids: list, ground_truth_ids: list, k: int) -> float:
    """Compute recall@k: fraction of true k-NN found in predicted results."""
    true_set = set(ground_truth_ids[:k])
    pred_set = set(predicted_ids[:k])
    if not true_set:
        return 0.0
    return len(true_set & pred_set) / len(true_set)


def measure_search(func, warmup=None, rounds=None) -> dict:
    if warmup is None:
        warmup = WARMUP_ROUNDS
    if rounds is None:
        rounds = MEASURE_ROUNDS
    for _ in range(warmup):
        func()
    times = []
    result = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    total_time = sum(times)
    return {
        "mean": statistics.mean(times),
        "median": percentile(times, 50),
        "p99": percentile(times, 99),
        "min": min(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "qps": rounds / total_time,
        "rounds": rounds,
        "_result": result,
    }


# ---------------------------------------------------------------------------
# Load SIFT1M dataset
# ---------------------------------------------------------------------------


def load_sift(path: str, n_queries: int = 1000) -> dict:
    """Load SIFT1M from HDF5 (ann-benchmarks format)."""
    print(f"  Loading SIFT1M from {path}...")
    with h5py.File(path, "r") as f:
        train = np.array(f["train"])      # (1M, 128) base vectors
        test = np.array(f["test"])        # (10K, 128) query vectors
        neighbors = np.array(f["neighbors"])  # (10K, 100) ground truth
        distances = np.array(f["distances"])  # (10K, 100) ground truth

    # Use subset of queries for benchmark speed
    test = test[:n_queries]
    neighbors = neighbors[:n_queries]
    distances = distances[:n_queries]

    print(f"  Base vectors:  {train.shape[0]:,} x {train.shape[1]}D")
    print(f"  Query vectors: {test.shape[0]:,}")
    print(f"  Ground truth:  top-{neighbors.shape[1]} per query")
    return {
        "train": train,
        "test": test,
        "neighbors": neighbors,
        "distances": distances,
    }


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------


def setup_qdrant(client: QdrantClient, vectors: np.ndarray) -> float:
    """Insert vectors into Qdrant. Returns load time."""
    dim = vectors.shape[1]
    client.recreate_collection(
        collection_name="sift",
        vectors_config=VectorParams(size=dim, distance=Distance.EUCLID),
    )

    total = len(vectors)
    t0 = time.perf_counter()
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = vectors[start:end]
        points = [
            PointStruct(id=int(start + i), vector=batch[i].tolist())
            for i in range(len(batch))
        ]
        client.upsert(collection_name="sift", points=points)
        if (end % 100000 == 0) or end >= total:
            print(f"    Qdrant: {end:,}/{total:,}")
    load_time = time.perf_counter() - t0

    # Wait for indexing
    info = client.get_collection("sift")
    while info.status != "green":
        time.sleep(0.5)
        info = client.get_collection("sift")

    return load_time


def setup_velesdb(vectors: np.ndarray, db_path: str):
    """Insert vectors into VelesDB. Returns (collection, load_time)."""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    dim = vectors.shape[1]
    db = velesdb.Database(db_path)
    collection = db.create_collection("sift", dimension=dim, metric="euclidean")

    total = len(vectors)
    t0 = time.perf_counter()
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = vectors[start:end]
        points = [
            {"id": int(start + i), "vector": batch[i].tolist()}
            for i in range(len(batch))
        ]
        collection.upsert(points)
        if (end % 100000 == 0) or end >= total:
            print(f"    VelesDB: {end:,}/{total:,}")
    load_time = time.perf_counter() - t0

    return collection, load_time


# ---------------------------------------------------------------------------
# Benchmark queries
# ---------------------------------------------------------------------------


def define_queries(qdrant_client, veles_col, dataset, top_k_values):
    """Define search queries for various top_k values."""
    queries = []

    for k in top_k_values:
        # Pick a single representative query for latency measurement
        q_vec = dataset["test"][0].tolist()
        gt = dataset["neighbors"][0].tolist()

        def make_qdrant_fn(vec, topk):
            def fn():
                res = qdrant_client.query_points(
                    collection_name="sift",
                    query=vec,
                    limit=topk,
                )
                return [p.id for p in res.points]
            return fn

        def make_veles_fn(vec, topk):
            def fn():
                res = veles_col._inner.search(vector=vec, top_k=topk)
                return [r["id"] for r in res]
            return fn

        queries.append({
            "name": f"kNN_top{k}",
            "description": f"Single query, top-{k} nearest neighbors",
            "top_k": k,
            "qdrant_fn": make_qdrant_fn(q_vec, k),
            "veles_fn": make_veles_fn(q_vec, k),
            "ground_truth": gt,
        })

    return queries


def measure_recall_batch(search_fn, dataset, top_k, n_queries=None):
    """Measure average recall@k across multiple queries."""
    test = dataset["test"]
    neighbors = dataset["neighbors"]
    if n_queries:
        test = test[:n_queries]
        neighbors = neighbors[:n_queries]

    recalls = []
    for i in range(len(test)):
        ids = search_fn(test[i].tolist(), top_k)
        r = recall_at_k(ids, neighbors[i].tolist(), top_k)
        recalls.append(r)
    return statistics.mean(recalls)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[dict], machine: dict, n_vectors: int):
    print("\n" + "=" * 78)
    print("  VelesDB vs Qdrant \u2014 ANN Vector Search Benchmark")
    print("=" * 78)
    print(f"  Dataset:  SIFT1M ({n_vectors:,} vectors, 128-dim, euclidean)")
    print(f"  OS:       {machine.get('os', 'WSL2')}")
    print(f"  VelesDB:  {machine.get('velesdb', '?')}")
    print(f"  Qdrant:   {machine.get('qdrant', '?')}")
    print(f"  Rounds:   {MEASURE_ROUNDS} (warmup: {WARMUP_ROUNDS})")
    print(f"  Fairness: Same dataset, same queries, same process")
    print("=" * 78)

    for r in results:
        qd = r["qdrant"]
        vl = r["velesdb"]

        ratio = qd["median"] / vl["median"] if vl["median"] > 0 else 0
        if ratio > 1:
            winner = f"VelesDB {ratio:.1f}x faster"
        elif ratio > 0:
            winner = f"Qdrant {1/ratio:.1f}x faster"
        else:
            winner = "N/A"

        print(f"\n  {r['name']}: {r['description']}")
        print(f"    {'Engine':<12} {'Median':>10} {'P99':>10} {'QPS':>10} "
              f"{'Recall':>10}")
        print(f"    {'─' * 56}")
        print(f"    {'Qdrant':<12} {fmt_time(qd['median']):>10} "
              f"{fmt_time(qd['p99']):>10} {qd['qps']:>10.0f} "
              f"{r['qdrant_recall']:>9.3f}")
        print(f"    {'VelesDB':<12} {fmt_time(vl['median']):>10} "
              f"{fmt_time(vl['p99']):>10} {vl['qps']:>10.0f} "
              f"{r['velesdb_recall']:>9.3f}")
        print(f"    \u2192 {winner}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global MEASURE_ROUNDS, WARMUP_ROUNDS

    parser = argparse.ArgumentParser(
        description="VelesDB vs Qdrant \u2014 ANN Vector Benchmark")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sift", default=SIFT_PATH)
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--top-k", type=str, default="10,100",
                        help="Comma-separated top-k values (default: 10,100)")
    parser.add_argument("--n-queries", type=int, default=200,
                        help="Number of queries for recall measurement")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    MEASURE_ROUNDS = args.rounds
    WARMUP_ROUNDS = args.warmup
    top_k_values = [int(x.strip()) for x in args.top_k.split(",")]

    print("VelesDB vs Qdrant \u2014 ANN Vector Search Benchmark")
    print("=" * 52)

    machine = {
        "os": "WSL2",
        "velesdb": velesdb.__version__,
    }

    # Load dataset
    dataset = load_sift(args.sift, n_queries=args.n_queries)
    n_vectors = dataset["train"].shape[0]

    # Connect to Qdrant
    print("\n  Connecting to Qdrant...")
    try:
        qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
        qdrant_info = qdrant.get_collections()
        machine["qdrant"] = f"v{args.qdrant_port}"
        print(f"  Qdrant OK (port {args.qdrant_port})")
    except Exception as e:
        print(f"  ERROR connecting to Qdrant: {e}")
        sys.exit(1)

    # Load vectors into engines
    print(f"\n  Loading {n_vectors:,} vectors into Qdrant...")
    qdrant_load = setup_qdrant(qdrant, dataset["train"])
    print(f"  Qdrant: {qdrant_load:.2f}s")

    db_path = "/tmp/velesdb_vector"
    print(f"\n  Loading {n_vectors:,} vectors into VelesDB...")
    veles_col, velesdb_load = setup_velesdb(dataset["train"], db_path)
    print(f"  VelesDB: {velesdb_load:.2f}s")

    # Define queries
    queries = define_queries(qdrant, veles_col, dataset, top_k_values)

    # Run benchmark
    print(f"\n  Running {MEASURE_ROUNDS} rounds per query "
          f"(warmup: {WARMUP_ROUNDS})...")

    all_results = []
    for q in queries:
        k = q["top_k"]
        print(f"    {q['name']}...")

        # Latency measurement (single query, many rounds)
        qdrant_m = measure_search(q["qdrant_fn"])
        veles_m = measure_search(q["veles_fn"])

        # Recall measurement (many queries, single pass)
        print(f"      Recall@{k} ({args.n_queries} queries)...")

        def qdrant_search(vec, topk=k):
            res = qdrant.query_points(
                collection_name="sift", query=vec, limit=topk)
            return [p.id for p in res.points]

        def veles_search(vec, topk=k):
            res = veles_col._inner.search(vector=vec, top_k=topk)
            return [r["id"] for r in res]

        qdrant_recall = measure_recall_batch(
            qdrant_search, dataset, k, args.n_queries)
        veles_recall = measure_recall_batch(
            veles_search, dataset, k, args.n_queries)

        all_results.append({
            "name": q["name"],
            "description": q["description"],
            "top_k": k,
            "qdrant": {x: v for x, v in qdrant_m.items() if x != "_result"},
            "velesdb": {x: v for x, v in veles_m.items() if x != "_result"},
            "qdrant_recall": qdrant_recall,
            "velesdb_recall": veles_recall,
        })

    # Output
    if args.json:
        output = {
            "benchmark": "ann-vector",
            "machine": machine,
            "dataset": {"name": "SIFT1M", "vectors": n_vectors,
                        "dimension": 128, "metric": "euclidean"},
            "config": {"rounds": MEASURE_ROUNDS, "warmup": WARMUP_ROUNDS,
                       "n_queries_recall": args.n_queries},
            "load_times": {"qdrant": qdrant_load, "velesdb": velesdb_load},
            "results": all_results,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_results(all_results, machine, n_vectors)

    # Cleanup
    del veles_col
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    print("\n  Done.")


if __name__ == "__main__":
    main()
