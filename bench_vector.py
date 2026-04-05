#!/usr/bin/env python3
"""
VelesDB vs Qdrant — ANN Vector Search Benchmark
=================================================

Uses SIFT1M from ann-benchmarks.com (1M vectors, 128-dim, euclidean).
Standard ANN benchmark with ground truth for recall measurement.

Fairness:
  - Both engines run in Docker containers
  - Both accessed via HTTP from the same Python process
  - Same dataset, same queries, same machine
"""

import argparse
import json
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
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("ERROR: qdrant-client not installed (pip install qdrant-client)")
    sys.exit(1)

from velesdb_client import VelesDBClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_ROUNDS = 10
MEASURE_ROUNDS = 50
BATCH_SIZE = 1000  # Smaller batches for HTTP transport
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
        return f"{seconds * 1_000_000:.0f} µs"
    return f"{seconds * 1_000:.2f} ms"


def recall_at_k(predicted_ids: list, ground_truth_ids: list, k: int) -> float:
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
    print(f"  Loading SIFT1M from {path}...")
    with h5py.File(path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])
        neighbors = np.array(f["neighbors"])
        distances = np.array(f["distances"])

    test = test[:n_queries]
    neighbors = neighbors[:n_queries]
    distances = distances[:n_queries]

    print(f"  Base vectors:  {train.shape[0]:,} x {train.shape[1]}D")
    print(f"  Query vectors: {test.shape[0]:,}")
    print(f"  Ground truth:  top-{neighbors.shape[1]} per query")
    return {"train": train, "test": test, "neighbors": neighbors, "distances": distances}


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------


def setup_qdrant(client: QdrantClient, vectors: np.ndarray) -> float:
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

    info = client.get_collection("sift")
    while info.status != "green":
        time.sleep(0.5)
        info = client.get_collection("sift")
    return load_time


def setup_velesdb(client: VelesDBClient, vectors: np.ndarray) -> float:
    dim = vectors.shape[1]
    client.delete_collection("sift")
    client.create_collection("sift", dimension=dim, metric="euclidean")

    total = len(vectors)
    t0 = time.perf_counter()
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = vectors[start:end]
        points = [
            {"id": int(start + i), "vector": batch[i].tolist()}
            for i in range(len(batch))
        ]
        client.upsert_points("sift", points)
        if (end % 100000 == 0) or end >= total:
            print(f"    VelesDB: {end:,}/{total:,}")
    load_time = time.perf_counter() - t0
    return load_time


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def measure_recall_batch(search_fn, dataset, top_k, n_queries=None):
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


def print_results(results: list[dict], machine: dict, n_vectors: int):
    print("\n" + "=" * 78)
    print("  VelesDB vs Qdrant — ANN Vector Search Benchmark")
    print("=" * 78)
    print(f"  Dataset:  SIFT1M ({n_vectors:,} vectors, 128-dim, euclidean)")
    print(f"  Runtime:  All engines in Docker, accessed via HTTP")
    print(f"  VelesDB:  {machine.get('velesdb', '?')}")
    print(f"  Qdrant:   {machine.get('qdrant', '?')}")
    print(f"  Rounds:   {MEASURE_ROUNDS} (warmup: {WARMUP_ROUNDS})")
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
        print(f"    {'Engine':<12} {'Median':>10} {'P99':>10} {'QPS':>10} {'Recall':>10}")
        print(f"    {'─' * 56}")
        print(f"    {'Qdrant':<12} {fmt_time(qd['median']):>10} "
              f"{fmt_time(qd['p99']):>10} {qd['qps']:>10.0f} "
              f"{r['qdrant_recall']:>9.3f}")
        print(f"    {'VelesDB':<12} {fmt_time(vl['median']):>10} "
              f"{fmt_time(vl['p99']):>10} {vl['qps']:>10.0f} "
              f"{r['velesdb_recall']:>9.3f}")
        print(f"    → {winner}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global MEASURE_ROUNDS, WARMUP_ROUNDS

    parser = argparse.ArgumentParser(description="VelesDB vs Qdrant — ANN Vector Benchmark")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sift", default=SIFT_PATH)
    parser.add_argument("--velesdb-host", default="localhost")
    parser.add_argument("--velesdb-port", type=int, default=8080)
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--top-k", type=str, default="10,100")
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    MEASURE_ROUNDS = args.rounds
    WARMUP_ROUNDS = args.warmup
    top_k_values = [int(x.strip()) for x in args.top_k.split(",")]

    print("VelesDB vs Qdrant — ANN Vector Search Benchmark")
    print("=" * 52)

    # Connect to VelesDB
    print("\n  Connecting to VelesDB...")
    veles = VelesDBClient(host=args.velesdb_host, port=args.velesdb_port)
    try:
        info = veles.health()
        machine = {"velesdb": info.get("version", "?")}
        print(f"  VelesDB {machine['velesdb']} OK")
    except Exception as e:
        print(f"  ERROR connecting to VelesDB: {e}")
        sys.exit(1)

    # Connect to Qdrant
    print("  Connecting to Qdrant...")
    try:
        qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
        qdrant.get_collections()
        machine["qdrant"] = f"port {args.qdrant_port}"
        print(f"  Qdrant OK ({machine['qdrant']})")
    except Exception as e:
        print(f"  ERROR connecting to Qdrant: {e}")
        sys.exit(1)

    # Load dataset
    dataset = load_sift(args.sift, n_queries=args.n_queries)
    n_vectors = dataset["train"].shape[0]

    # Load vectors
    print(f"\n  Loading {n_vectors:,} vectors into Qdrant...")
    qdrant_load = setup_qdrant(qdrant, dataset["train"])
    print(f"  Qdrant: {qdrant_load:.2f}s")

    print(f"\n  Loading {n_vectors:,} vectors into VelesDB...")
    velesdb_load = setup_velesdb(veles, dataset["train"])
    print(f"  VelesDB: {velesdb_load:.2f}s")

    # Run benchmarks
    print(f"\n  Running {MEASURE_ROUNDS} rounds per query (warmup: {WARMUP_ROUNDS})...")

    all_results = []
    for k in top_k_values:
        q_vec = dataset["test"][0].tolist()
        name = f"kNN_top{k}"
        print(f"    {name}...")

        def make_qdrant_fn(vec, topk):
            def fn():
                res = qdrant.query_points(collection_name="sift", query=vec, limit=topk)
                return [p.id for p in res.points]
            return fn

        def make_veles_fn(vec, topk):
            def fn():
                res = veles.search("sift", vector=vec, top_k=topk)
                return [r["id"] for r in res]
            return fn

        qdrant_m = measure_search(make_qdrant_fn(q_vec, k))
        veles_m = measure_search(make_veles_fn(q_vec, k))

        # Recall
        print(f"      Recall@{k} ({args.n_queries} queries)...")

        def qdrant_search(vec, topk=k):
            res = qdrant.query_points(collection_name="sift", query=vec, limit=topk)
            return [p.id for p in res.points]

        def veles_search(vec, topk=k):
            res = veles.search("sift", vector=vec, top_k=topk)
            return [r["id"] for r in res]

        qdrant_recall = measure_recall_batch(qdrant_search, dataset, k, args.n_queries)
        veles_recall = measure_recall_batch(veles_search, dataset, k, args.n_queries)

        all_results.append({
            "name": name,
            "description": f"Single query, top-{k} nearest neighbors",
            "top_k": k,
            "qdrant": {x: v for x, v in qdrant_m.items() if x != "_result"},
            "velesdb": {x: v for x, v in veles_m.items() if x != "_result"},
            "qdrant_recall": qdrant_recall,
            "velesdb_recall": veles_recall,
        })

    if args.json:
        output = {
            "benchmark": "ann-vector",
            "machine": machine,
            "dataset": {"name": "SIFT1M", "vectors": n_vectors, "dimension": 128, "metric": "euclidean"},
            "config": {"rounds": MEASURE_ROUNDS, "warmup": WARMUP_ROUNDS, "n_queries_recall": args.n_queries},
            "load_times": {"qdrant": qdrant_load, "velesdb": velesdb_load},
            "results": all_results,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_results(all_results, machine, n_vectors)

    # Cleanup
    veles.delete_collection("sift")
    print("\n  Done.")


if __name__ == "__main__":
    main()
