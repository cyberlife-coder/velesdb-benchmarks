#!/usr/bin/env python3
"""
VelesDB Hybrid vs ClickHouse + Qdrant + igraph Combined
========================================================

The ultimate benchmark: hybrid queries needing ALL 3 paradigms.
All engines run in Docker, accessed via network.
"""

import argparse
import json
import math
import random
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
    import clickhouse_connect
except ImportError:
    print("ERROR: clickhouse-connect not installed")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter,
        FieldCondition, Range,
    )
except ImportError:
    print("ERROR: qdrant-client not installed")
    sys.exit(1)

from velesdb_client import VelesDBClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_ROUNDS = 10
MEASURE_ROUNDS = 50
N_PERSONS = 100000
DIMENSION = 128
BATCH_SIZE = 1000
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
        return f"{seconds * 1_000_000:.0f} µs"
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
    rng = random.Random(seed)
    print(f"  Generating {N_PERSONS:,} persons...")

    sectors = ["Tech", "Finance", "Health", "Retail", "Energy"]
    cities = ["Paris", "London", "Berlin", "Tokyo", "NYC", "SF", "Sydney", "Toronto"]

    persons = []
    for i in range(N_PERSONS):
        persons.append({
            "id": i, "name": f"Person_{i}",
            "age": rng.randint(18, 80), "income": rng.randint(20000, 200000),
            "city": rng.choice(cities), "company_id": rng.randint(0, 499),
            "sector": rng.choice(sectors),
            "vector": pseudo_embedding(i, DIMENSION),
        })

    edges = []
    edge_id = 0
    for _ in range(N_PERSONS * 10):
        a = int(rng.paretovariate(1.5)) % N_PERSONS
        b = int(rng.paretovariate(1.5)) % N_PERSONS
        if a != b:
            edges.append({"id": edge_id, "source": a, "target": b, "label": "KNOWS"})
            edge_id += 1

    print(f"  Generated: {len(persons):,} persons, {len(edges):,} edges")
    return {"persons": persons, "edges": edges}


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------


def setup_all_engines(dataset, ch_client, qdrant, veles: VelesDBClient):
    persons = dataset["persons"]
    edges = dataset["edges"]
    times = {}

    # ClickHouse
    print("\n  Loading into ClickHouse...")
    t0 = time.perf_counter()
    ch_client.command("DROP TABLE IF EXISTS persons")
    ch_client.command("""
        CREATE TABLE persons (
            id UInt32, name String, age UInt8, income UInt32,
            city String, company_id UInt16, sector String
        ) ENGINE = MergeTree() ORDER BY id
    """)
    rows = [[p["id"], p["name"], p["age"], p["income"], p["city"], p["company_id"], p["sector"]] for p in persons]
    ch_client.insert("persons", rows, column_names=["id", "name", "age", "income", "city", "company_id", "sector"])
    times["clickhouse"] = time.perf_counter() - t0

    # Qdrant
    print("  Loading into Qdrant...")
    t0 = time.perf_counter()
    try:
        qdrant.delete_collection("persons")
    except Exception:
        pass
    qdrant.create_collection("persons", VectorParams(size=DIMENSION, distance=Distance.COSINE))
    for start in range(0, len(persons), BATCH_SIZE):
        batch = persons[start:start + BATCH_SIZE]
        points = [PointStruct(id=p["id"], vector=p["vector"],
                              payload={"age": p["age"], "income": p["income"],
                                       "city": p["city"], "sector": p["sector"]})
                  for p in batch]
        qdrant.upsert("persons", points)
    while qdrant.get_collection("persons").status.value != "green":
        time.sleep(0.2)
    times["qdrant"] = time.perf_counter() - t0

    # igraph (in-process, same as before — this is the "combined" approach)
    print("  Building igraph...")
    t0 = time.perf_counter()
    ig_knows = ig.Graph(n=N_PERSONS, directed=True)
    ig_knows.add_edges([(e["source"], e["target"]) for e in edges])
    times["igraph"] = time.perf_counter() - t0

    # VelesDB (vector + graph via HTTP)
    print("  Loading into VelesDB...")
    t0 = time.perf_counter()
    veles.delete_collection("persons")
    veles.create_collection("persons", dimension=DIMENSION, metric="cosine")
    for start in range(0, len(persons), BATCH_SIZE):
        batch = persons[start:start + BATCH_SIZE]
        points = [{"id": p["id"], "vector": p["vector"],
                    "payload": {"name": p["name"], "age": p["age"], "income": p["income"],
                                "city": p["city"], "sector": p["sector"]}}
                  for p in batch]
        veles.upsert_points("persons", points)
        if (start + BATCH_SIZE) % 20000 == 0 or start + BATCH_SIZE >= len(persons):
            print(f"    VelesDB vectors: {min(start + BATCH_SIZE, len(persons)):,}/{len(persons):,}")

    # Graph edges
    for e in edges:
        veles.add_edge("persons", e)
    # Store node payloads for graph queries
    for p in persons:
        veles.store_node_payload("persons", p["id"], {
            "_labels": ["Person"], "sector": p["sector"], "age": p["age"], "income": p["income"],
        })
    times["velesdb"] = time.perf_counter() - t0

    return ig_knows, times


# ---------------------------------------------------------------------------
# Hybrid queries
# ---------------------------------------------------------------------------


def define_hybrid_queries(ch_client, qdrant, ig_knows, veles: VelesDBClient):
    queries = []
    query_vec = pseudo_embedding(42, DIMENSION)
    outdegrees = ig_knows.outdegree()
    source_person = min(range(N_PERSONS), key=lambda v: abs(outdegrees[v] - 20))
    print(f"  Source person: {source_person} (degree: {outdegrees[source_person]})")

    # Q1: Vector + Graph
    def combined_q1():
        friend_ids = set(ig_knows.neighbors(source_person, mode="out"))
        res = qdrant.query_points("persons", query=query_vec, limit=TOP_K * 10)
        return [p.id for p in res.points if p.id in friend_ids][:TOP_K]

    def velesdb_q1():
        friends = veles.traverse_bfs("persons", source_person, max_depth=1, rel_types=["KNOWS"])
        friend_ids = set(f["target_id"] for f in friends)
        res = veles.search("persons", vector=query_vec, top_k=TOP_K * 10)
        return [r["id"] for r in res if r["id"] in friend_ids][:TOP_K]

    queries.append(("Q1_vector_graph", f"Similar + friends → top {TOP_K}", combined_q1, velesdb_q1))

    # Q2: Vector + Columnar
    def combined_q2():
        res = qdrant.query_points("persons", query=query_vec,
                                   query_filter=Filter(must=[
                                       FieldCondition(key="age", range=Range(gt=30)),
                                       FieldCondition(key="income", range=Range(gt=80000)),
                                   ]), limit=TOP_K)
        return [p.id for p in res.points]

    def velesdb_q2():
        res = veles.execute_query(
            f"SELECT id FROM persons WHERE vector NEAR $v AND age > 30 AND income > 80000 LIMIT {TOP_K}",
            {"v": query_vec})
        return [r.get("payload", {}).get("id") for r in res]

    queries.append(("Q2_vector_columnar", f"Similar + age>30 + income>80K → top {TOP_K}", combined_q2, velesdb_q2))

    # Q3: Graph + Columnar
    def combined_q3():
        friend_ids = ig_knows.neighborhood(source_person, order=2, mode="out", mindist=1)[:5000]
        if not friend_ids:
            return []
        id_list = ",".join(str(x) for x in friend_ids)
        res = ch_client.query(f"SELECT id FROM persons WHERE id IN ({id_list}) AND sector = 'Tech' LIMIT {TOP_K}")
        return [r[0] for r in res.result_rows]

    def velesdb_q3():
        friends = veles.traverse_bfs("persons", source_person, max_depth=2, rel_types=["KNOWS"])
        friend_ids = list(set(f["target_id"] for f in friends))
        results = []
        for fid in friend_ids[:5000]:
            payload = veles.get_node_payload("persons", fid)
            if payload and payload.get("sector") == "Tech":
                results.append(fid)
                if len(results) >= TOP_K:
                    break
        return results

    queries.append(("Q3_graph_columnar", f"Friends-of-friends in Tech → top {TOP_K}", combined_q3, velesdb_q3))

    return queries


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[dict], machine: dict):
    print("\n" + "=" * 78)
    print("  VelesDB (unified) vs CH+Qdrant+igraph (combined)")
    print("=" * 78)
    print(f"  Dataset:  {N_PERSONS:,} persons (vector + graph + metadata)")
    print(f"  Runtime:  All engines in Docker, accessed via network")
    print(f"  Rounds:   {MEASURE_ROUNDS} (warmup: {WARMUP_ROUNDS})")
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
        print(f"    {'Approach':<24} {'Median':>10} {'P99':>10} {'Mean':>10}")
        print(f"    {'─' * 44}")
        print(f"    {'CH+Qdrant+igraph':<24} {fmt_time(cb['median']):>10} {fmt_time(cb['p99']):>10} {fmt_time(cb['mean']):>10}")
        print(f"    {'VelesDB (unified)':<24} {fmt_time(vl['median']):>10} {fmt_time(vl['p99']):>10} {fmt_time(vl['mean']):>10}")
        print(f"    → {winner}")
        print(f"    Results: Combined={r['combined_count']}, VelesDB={r['veles_count']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global MEASURE_ROUNDS, WARMUP_ROUNDS, TOP_K

    parser = argparse.ArgumentParser(description="VelesDB Hybrid vs 3 Engines Combined")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--velesdb-host", default="localhost")
    parser.add_argument("--velesdb-port", type=int, default=8080)
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

    # Connect
    veles = VelesDBClient(host=args.velesdb_host, port=args.velesdb_port)
    try:
        info = veles.health()
        machine = {"velesdb": info.get("version", "?")}
        print(f"  VelesDB {machine['velesdb']} OK")
    except Exception as e:
        print(f"  ERROR VelesDB: {e}")
        sys.exit(1)

    try:
        ch_client = clickhouse_connect.get_client(host=args.ch_host, port=args.ch_port)
        machine["clickhouse"] = str(ch_client.command("SELECT version()"))
    except Exception as e:
        print(f"  ERROR ClickHouse: {e}")
        sys.exit(1)

    try:
        qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
        qdrant.get_collections()
    except Exception as e:
        print(f"  ERROR Qdrant: {e}")
        sys.exit(1)

    dataset = generate_dataset()
    ig_knows, load_times = setup_all_engines(dataset, ch_client, qdrant, veles)
    del dataset

    print(f"\n  Running {MEASURE_ROUNDS} rounds per query (warmup: {WARMUP_ROUNDS})...")
    queries = define_hybrid_queries(ch_client, qdrant, ig_knows, veles)

    all_results = []
    for name, desc, combined_fn, veles_fn in queries:
        print(f"    {name}...")
        combined_m = measure(combined_fn)
        veles_m = measure(veles_fn)
        all_results.append({
            "name": name, "description": desc,
            "combined": {k: v for k, v in combined_m.items() if k != "_result"},
            "velesdb": {k: v for k, v in veles_m.items() if k != "_result"},
            "combined_count": len(combined_m["_result"]) if combined_m["_result"] else 0,
            "veles_count": len(veles_m["_result"]) if veles_m["_result"] else 0,
        })

    if args.json:
        output = {"benchmark": "hybrid", "machine": machine, "load_times": load_times, "results": all_results}
        print(json.dumps(output, indent=2, default=str))
    else:
        print_results(all_results, machine)

    veles.delete_collection("persons")
    print("\n  Done.")


if __name__ == "__main__":
    main()
