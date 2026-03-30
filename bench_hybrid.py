#!/usr/bin/env python3
"""
VelesDB Hybrid vs ClickHouse + Qdrant + igraph Combined
========================================================

The ultimate benchmark: hybrid queries that need ALL 3 paradigms.
VelesDB handles vector + graph + columnar natively.
The "combined" approach orchestrates 3 separate engines.

Dataset: 100K persons with:
  - Embeddings (128-dim, interest profiles)
  - Graph relations (KNOWS)
  - Columnar metadata (age, city, income, sector)

Queries (all require 2+ paradigms):
  Q1: "Find people similar to X who are friends of Y"       → vector search + graph traversal
  Q2: "Find people similar to X with age > 30 and income > 80K"
      → vector search + columnar filter
  Q3: "Find friends-of-friends of X who work in Tech sector"
      → graph traversal + columnar filter
  Q4: "Find people similar to X, friends of Y, age > 30"
      → vector + graph + columnar (triple hybrid)
  Q5: "Find people similar to X AND within 2 hops of Y"
      → vector + graph (2-hop)

Fairness:
  - Same dataset across all engines
  - Same WSL2 environment, same Python process
  - VelesDB: single engine, one process
  - Combined: 3 engines, orchestration overhead measured
"""

import argparse
import json
import math
import os
import random
import shutil
import statistics
import sys
import time

import numpy as np

try:
    import igraph as ig
except ImportError:
    print("ERROR: igraph not installed (pip install igraph)")
    sys.exit(1)

try:
    import velesdb
except ImportError:
    print("ERROR: velesdb not installed")
    sys.exit(1)

try:
    import clickhouse_connect
except ImportError:
    print("ERROR: clickhouse-connect not installed")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter,
        FieldCondition, Range, MatchValue,
    )
except ImportError:
    print("ERROR: qdrant-client not installed")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_ROUNDS = 10
MEASURE_ROUNDS = 50
N_PERSONS = 100000
DIMENSION = 128
BATCH_SIZE = 5000
TOP_K = 50

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


def measure(func, warmup=None, rounds=None) -> dict:
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
    return {
        "mean": statistics.mean(times),
        "median": percentile(times, 50),
        "p99": percentile(times, 99),
        "min": min(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "rounds": rounds,
        "_result": result,
    }


def pseudo_embedding(i: int, dim: int = DIMENSION) -> list:
    return [math.sin(i * 0.01 + j * 0.01) for j in range(dim)]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(seed=42):
    """Generate persons with embeddings + metadata + graph relations."""
    rng = random.Random(seed)

    print(f"  Generating {N_PERSONS:,} persons with embeddings + graph + metadata...")

    sectors = ["Tech", "Finance", "Health", "Retail", "Energy"]
    cities = ["Paris", "London", "Berlin", "Tokyo", "NYC", "SF", "Sydney", "Toronto"]

    persons = []
    for i in range(N_PERSONS):
        persons.append({
            "id": i,
            "name": f"Person_{i}",
            "age": rng.randint(18, 80),
            "income": rng.randint(20000, 200000),
            "city": rng.choice(cities),
            "company_id": rng.randint(0, 499),
            "sector": rng.choice(sectors),
            "vector": pseudo_embedding(i, DIMENSION),
        })

    # KNOWS edges (power-law)
    edges = []
    edge_id = 0
    n_edges = N_PERSONS * 10
    for _ in range(n_edges):
        a = int(rng.paretovariate(1.5)) % N_PERSONS
        b = int(rng.paretovariate(1.5)) % N_PERSONS
        if a != b:
            edges.append({"id": edge_id, "source": a, "target": b,
                          "label": "KNOWS"})
            edge_id += 1

    print(f"  Generated: {len(persons):,} persons, {len(edges):,} edges")
    return {"persons": persons, "edges": edges}


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------


def setup_all_engines(dataset, ch_client, qdrant, veles_db_path):
    """Load data into all engines. Returns (veles_col, veles_graph, veles_db, ig_knows, times)."""
    persons = dataset["persons"]
    edges = dataset["edges"]
    times = {}

    # --- ClickHouse (columnar metadata) ---
    print("\n  Loading into ClickHouse (metadata)...")
    t0 = time.perf_counter()
    ch_client.command("DROP TABLE IF EXISTS persons")
    ch_client.command("""
        CREATE TABLE persons (
            id UInt32, name String, age UInt8, income UInt32,
            city String, company_id UInt16, sector String
        ) ENGINE = MergeTree() ORDER BY id
    """)
    rows = [[p["id"], p["name"], p["age"], p["income"],
             p["city"], p["company_id"], p["sector"]] for p in persons]
    ch_client.insert("persons", rows,
                     column_names=["id", "name", "age", "income",
                                   "city", "company_id", "sector"])
    times["clickhouse"] = time.perf_counter() - t0
    print(f"    ClickHouse: {times['clickhouse']:.2f}s")

    # --- Qdrant (vectors) ---
    print("  Loading into Qdrant (vectors)...")
    t0 = time.perf_counter()
    try:
        qdrant.delete_collection("persons")
    except Exception:
        pass
    qdrant.create_collection(
        collection_name="persons",
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
    )
    for start in range(0, len(persons), BATCH_SIZE):
        batch = persons[start:start + BATCH_SIZE]
        points = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload={"age": p["age"], "income": p["income"],
                         "city": p["city"], "sector": p["sector"]},
            )
            for p in batch
        ]
        qdrant.upsert(collection_name="persons", points=points)
        if (start + BATCH_SIZE) % 20000 == 0 or start + BATCH_SIZE >= len(persons):
            print(f"    Qdrant: {min(start + BATCH_SIZE, len(persons)):,}/{len(persons):,}")
    while qdrant.get_collection("persons").status.value != "green":
        time.sleep(0.2)
    times["qdrant"] = time.perf_counter() - t0
    print(f"    Qdrant: {times['qdrant']:.2f}s")

    # --- igraph (graph) ---
    print("  Building igraph (KNOWS graph)...")
    t0 = time.perf_counter()
    ig_knows = ig.Graph(n=N_PERSONS, directed=True)
    knows_edges = [(e["source"], e["target"]) for e in edges if e["label"] == "KNOWS"]
    ig_knows.add_edges(knows_edges)
    # Store sector for fast lookup in combined Q3
    ig_sectors = [p["sector"] for p in persons]
    ig_ages = [p["age"] for p in persons]
    ig_incomes = [p["income"] for p in persons]
    times["igraph"] = time.perf_counter() - t0
    print(f"    igraph: {times['igraph']:.2f}s")

    # --- VelesDB (all-in-one) ---
    print("  Loading into VelesDB (vector + graph + payload)...")
    if os.path.exists(veles_db_path):
        shutil.rmtree(veles_db_path)
    db = velesdb.Database(veles_db_path)

    t0 = time.perf_counter()
    # Vector collection with full payload (includes sector, age, income)
    col = db.create_collection("persons", dimension=DIMENSION, metric="cosine")
    for start in range(0, len(persons), BATCH_SIZE):
        batch = persons[start:start + BATCH_SIZE]
        points = [
            {"id": p["id"], "vector": p["vector"],
             "payload": {"name": p["name"], "age": p["age"],
                         "income": p["income"], "city": p["city"],
                         "sector": p["sector"]}}
            for p in batch
        ]
        col.upsert(points)
        if (start + BATCH_SIZE) % 20000 == 0 or start + BATCH_SIZE >= len(persons):
            print(f"    VelesDB vectors: {min(start + BATCH_SIZE, len(persons)):,}/{len(persons):,}")

    # Graph collection (sector stored in node payload for Q3)
    graph = db.create_graph_collection("social")
    for p in persons:
        graph.store_node_payload(p["id"], {
            "_labels": ["Person"],
            "name": p["name"],
            "sector": p["sector"],
            "age": p["age"],
            "income": p["income"],
        })
    for e in edges:
        graph.add_edge(e)
    graph.flush()

    times["velesdb"] = time.perf_counter() - t0
    print(f"    VelesDB: {times['velesdb']:.2f}s")

    return col, graph, db, ig_knows, ig_sectors, ig_ages, ig_incomes, times


# ---------------------------------------------------------------------------
# Hybrid queries
# ---------------------------------------------------------------------------


def define_hybrid_queries(ch_client, qdrant, ig_knows, ig_sectors, ig_ages,
                           ig_incomes, veles_col, veles_graph, veles_db):
    """Define hybrid queries needing 2+ paradigms."""
    queries = []
    query_vec = pseudo_embedding(42, DIMENSION)
    # Use a typical (non-hub) source person
    outdegrees = ig_knows.outdegree()
    target_deg = 20
    source_person = min(range(N_PERSONS), key=lambda v: abs(outdegrees[v] - target_deg))
    print(f"  Source person: {source_person} (KNOWS out-degree: {outdegrees[source_person]})")

    # =========================================================
    # Q1: Vector + Graph — "Similar to X AND friends of Y"
    # =========================================================
    def combined_q1():
        # Step 1: 1-hop friends (igraph — C, in-process)
        friend_ids = set(ig_knows.neighbors(source_person, mode="out"))
        # Step 2: vector search (Qdrant — network call)
        res = qdrant.query_points(
            collection_name="persons", query=query_vec, limit=TOP_K * 10)
        similar_ids = [p.id for p in res.points]
        # Step 3: intersect in Python
        return [pid for pid in similar_ids if pid in friend_ids][:TOP_K]

    def velesdb_q1():
        # Step 1: 1-hop friends (VelesDB graph — in-process)
        friends = veles_graph.traverse_bfs(source_person, max_depth=1,
                                            rel_types=["KNOWS"])
        friend_ids = set(f["target_id"] for f in friends)
        # Step 2: vector search (VelesDB — in-process, same data)
        res = veles_col._inner.search(vector=query_vec, top_k=TOP_K * 10)
        # Step 3: intersect
        return [r["id"] for r in res if r["id"] in friend_ids][:TOP_K]

    queries.append(("Q1_vector_graph",
                     f"Similar to X AND friends of Y \u2192 top {TOP_K}",
                     combined_q1, velesdb_q1))

    # =========================================================
    # Q2: Vector + Columnar — "Similar to X WHERE age > 30 AND income > 80K"
    # =========================================================
    def combined_q2():
        # Qdrant with payload filter (network call with filter)
        res = qdrant.query_points(
            collection_name="persons",
            query=query_vec,
            query_filter=Filter(must=[
                FieldCondition(key="age", range=Range(gt=30)),
                FieldCondition(key="income", range=Range(gt=80000)),
            ]),
            limit=TOP_K,
        )
        return [p.id for p in res.points]

    def velesdb_q2():
        # VelesQL: vector + range filters in single call
        res = veles_db.execute_query(
            f"SELECT id FROM persons WHERE vector NEAR $v "
            f"AND age > 30 AND income > 80000 LIMIT {TOP_K}",
            {"v": query_vec},
        )
        return [r.get("payload", {}).get("id") for r in res]

    queries.append(("Q2_vector_columnar",
                     f"Similar to X WHERE age>30 AND income>80K \u2192 top {TOP_K}",
                     combined_q2, velesdb_q2))

    # =========================================================
    # Q3: Graph + Columnar — "Friends-of-friends in Tech sector"
    # =========================================================
    def combined_q3():
        # Step 1: 2-hop friends (igraph BFS — C, in-process)
        friend_ids = ig_knows.neighborhood(source_person, order=2,
                                            mode="out", mindist=1)[:5000]
        if not friend_ids:
            return []
        # Step 2: filter by sector (ClickHouse — network call with SQL)
        id_list = ",".join(str(x) for x in friend_ids)
        res = ch_client.query(
            f"SELECT id FROM persons WHERE id IN ({id_list}) "
            f"AND sector = 'Tech' LIMIT {TOP_K}"
        )
        return [r[0] for r in res.result_rows]

    def velesdb_q3():
        # Step 1: 2-hop friends (VelesDB graph — in-process)
        friends = veles_graph.traverse_bfs(source_person, max_depth=2,
                                            rel_types=["KNOWS"])
        friend_ids = list(set(f["target_id"] for f in friends))
        # Step 2: filter by sector using graph node payloads (no network call)
        results = []
        for fid in friend_ids[:5000]:
            payload = veles_graph.get_node_payload(fid)
            if payload and payload.get("sector") == "Tech":
                results.append(fid)
                if len(results) >= TOP_K:
                    break
        return results

    queries.append(("Q3_graph_columnar",
                     f"Friends-of-friends in Tech sector \u2192 top {TOP_K}",
                     combined_q3, velesdb_q3))

    # =========================================================
    # Q4: Triple Hybrid — Vector + Graph + Columnar
    # "Similar to X, friends of Y, age > 30, income > 50K"
    # =========================================================
    def combined_q4():
        # Step 1: friends (igraph — C, in-process)
        friend_ids = set(ig_knows.neighbors(source_person, mode="out"))
        # Step 2: vector + filter (Qdrant — network call with filter)
        res = qdrant.query_points(
            collection_name="persons",
            query=query_vec,
            query_filter=Filter(must=[
                FieldCondition(key="age", range=Range(gt=30)),
                FieldCondition(key="income", range=Range(gt=50000)),
            ]),
            limit=TOP_K * 10,
        )
        similar = [p.id for p in res.points]
        # Step 3: intersect (Python)
        return [pid for pid in similar if pid in friend_ids][:TOP_K]

    def velesdb_q4():
        # Step 1: friends (VelesDB graph — in-process)
        friends = veles_graph.traverse_bfs(source_person, max_depth=1,
                                            rel_types=["KNOWS"])
        friend_ids = set(f["target_id"] for f in friends)
        # Step 2: vector + filter (VelesQL — single in-process call)
        res = veles_db.execute_query(
            f"SELECT id FROM persons WHERE vector NEAR $v "
            f"AND age > 30 AND income > 50000 LIMIT {TOP_K * 10}",
            {"v": query_vec},
        )
        similar = [r.get("payload", {}).get("id") for r in res]
        # Step 3: intersect
        return [pid for pid in similar if pid in friend_ids][:TOP_K]

    queries.append(("Q4_triple_hybrid",
                     f"Similar + friends + age>30 + income>50K \u2192 top {TOP_K}",
                     combined_q4, velesdb_q4))

    # =========================================================
    # Q5: Vector + Graph (2-hop) — "Similar to X AND within 2 hops of Y"
    # =========================================================
    def combined_q5():
        # 2-hop BFS (igraph — C, in-process)
        network_ids = set(ig_knows.neighborhood(source_person, order=2,
                                                  mode="out", mindist=1))
        # Vector search (Qdrant — network call)
        res = qdrant.query_points(
            collection_name="persons", query=query_vec, limit=TOP_K * 20)
        return [p.id for p in res.points if p.id in network_ids][:TOP_K]

    def velesdb_q5():
        # 2-hop BFS (VelesDB graph — in-process)
        friends = veles_graph.traverse_bfs(source_person, max_depth=2,
                                            rel_types=["KNOWS"])
        network_ids = set(f["target_id"] for f in friends)
        # Vector search (VelesDB — in-process)
        res = veles_col._inner.search(vector=query_vec, top_k=TOP_K * 20)
        return [r["id"] for r in res if r["id"] in network_ids][:TOP_K]

    queries.append(("Q5_vector_graph2hop",
                     f"Similar to X AND within 2 hops \u2192 top {TOP_K}",
                     combined_q5, velesdb_q5))

    return queries


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[dict], machine: dict):
    print("\n" + "=" * 78)
    print("  VelesDB (unified) vs CH+Qdrant+igraph (combined)")
    print("  Hybrid Multi-Paradigm Benchmark")
    print("=" * 78)
    print(f"  Dataset:  {N_PERSONS:,} persons (vector + graph + metadata)")
    print(f"  OS:       {machine.get('os', 'WSL2')}")
    print(f"  VelesDB:  {machine.get('velesdb', '?')}")
    print(f"  CH:       {machine.get('clickhouse', '?')}")
    print(f"  Qdrant:   port {machine.get('qdrant_port', '?')}")
    print(f"  igraph:   {machine.get('igraph', '?')}")
    print(f"  Rounds:   {MEASURE_ROUNDS} (warmup: {WARMUP_ROUNDS})")
    print(f"  Fairness: Same data everywhere, same process")
    print("=" * 78)

    for r in results:
        cb = r["combined"]
        vl = r["velesdb"]

        ratio = cb["median"] / vl["median"] if vl["median"] > 0 else 0
        if ratio > 1:
            winner = f"VelesDB {ratio:.1f}x faster"
        elif ratio > 0:
            winner = f"Combined {1/ratio:.1f}x faster"
        else:
            winner = "N/A"

        print(f"\n  {r['name']}: {r['description']}")
        print(f"    {'Approach':<24} {'Median':>10} {'P99':>10} "
              f"{'Mean':>10} {'Min':>10}")
        print(f"    {'─' * 56}")
        print(f"    {'CH+Qdrant+igraph':<24} "
              f"{fmt_time(cb['median']):>10} {fmt_time(cb['p99']):>10} "
              f"{fmt_time(cb['mean']):>10} {fmt_time(cb['min']):>10}")
        print(f"    {'VelesDB (unified)':<24} "
              f"{fmt_time(vl['median']):>10} {fmt_time(vl['p99']):>10} "
              f"{fmt_time(vl['mean']):>10} {fmt_time(vl['min']):>10}")
        print(f"    \u2192 {winner}")
        print(f"    Results: Combined={r['combined_count']}, "
              f"VelesDB={r['veles_count']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global MEASURE_ROUNDS, WARMUP_ROUNDS, TOP_K

    parser = argparse.ArgumentParser(
        description="VelesDB Hybrid vs 3 Engines Combined")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--ch-host", default="localhost")
    parser.add_argument("--ch-port", type=int, default=8123)
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    MEASURE_ROUNDS = args.rounds
    WARMUP_ROUNDS = args.warmup
    TOP_K = args.top_k

    print("VelesDB Hybrid vs CH+Qdrant+igraph Combined")
    print("=" * 50)

    machine = {
        "os": "WSL2",
        "velesdb": velesdb.__version__,
        "qdrant_port": args.qdrant_port,
        "igraph": ig.__version__,
    }

    # Connect to external engines
    print("\n  Connecting to engines...")
    try:
        ch_client = clickhouse_connect.get_client(
            host=args.ch_host, port=args.ch_port)
        machine["clickhouse"] = str(ch_client.command("SELECT version()"))
        print(f"  ClickHouse {machine['clickhouse']}")
    except Exception as e:
        print(f"  ERROR ClickHouse: {e}")
        sys.exit(1)

    try:
        qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
        qdrant.get_collections()
        print(f"  Qdrant OK (port {args.qdrant_port})")
    except Exception as e:
        print(f"  ERROR Qdrant: {e}")
        sys.exit(1)

    # Generate and load data
    dataset = generate_dataset()
    veles_db_path = "/tmp/velesdb_hybrid"
    (veles_col, veles_graph, veles_db,
     ig_knows, ig_sectors, ig_ages, ig_incomes,
     load_times) = setup_all_engines(dataset, ch_client, qdrant, veles_db_path)
    del dataset

    # Define and run queries
    print(f"\n  Running {MEASURE_ROUNDS} rounds per query "
          f"(warmup: {WARMUP_ROUNDS})...")
    queries = define_hybrid_queries(
        ch_client, qdrant, ig_knows, ig_sectors, ig_ages, ig_incomes,
        veles_col, veles_graph, veles_db)

    all_results = []
    for name, desc, combined_fn, veles_fn in queries:
        print(f"    {name}...")
        combined_m = measure(combined_fn)
        veles_m = measure(veles_fn)

        all_results.append({
            "name": name,
            "description": desc,
            "combined": {k: v for k, v in combined_m.items() if k != "_result"},
            "velesdb": {k: v for k, v in veles_m.items() if k != "_result"},
            "combined_count": len(combined_m["_result"]) if combined_m["_result"] else 0,
            "veles_count": len(veles_m["_result"]) if veles_m["_result"] else 0,
        })

    # Output
    if args.json:
        output = {
            "benchmark": "hybrid-multi-paradigm",
            "machine": machine,
            "dataset": {"persons": N_PERSONS, "dimension": DIMENSION},
            "config": {"rounds": MEASURE_ROUNDS, "warmup": WARMUP_ROUNDS,
                       "top_k": TOP_K},
            "load_times": load_times,
            "results": all_results,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_results(all_results, machine)

    # Cleanup
    del veles_col, veles_graph, veles_db
    if os.path.exists(veles_db_path):
        shutil.rmtree(veles_db_path, ignore_errors=True)

    print("\n  Done.")


if __name__ == "__main__":
    main()
