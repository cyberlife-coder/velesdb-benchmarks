#!/usr/bin/env python3
"""
VelesDB vs ClickHouse — Multi-column Benchmark
================================================

Compares multi-column filter + projection performance.
Both engines run in Docker, accessed via HTTP.
"""

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time

try:
    import clickhouse_connect
except ImportError:
    print("ERROR: clickhouse-connect not installed")
    sys.exit(1)

from velesdb_client import VelesDBClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
DIMENSION = 128
TOP_K = 100
WARMUP_ROUNDS = 10
MEASURE_ROUNDS = 100
DEFAULT_DATASETS = [10_000, 100_000]
BATCH_SIZE = 1000

CATEGORIES = ["tech", "science", "business", "sports", "health", "music", "art", "food"]
REGIONS = ["eu-west", "eu-east", "us-west", "us-east", "asia-pac",
           "latam", "africa", "mena", "oceania", "nordic"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pseudo_embedding(i: int, dim: int = DIMENSION) -> list:
    return [(math.sin(i * 0.01 + j * 0.01)) for j in range(dim)]


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
        "result_sample": result,
    }


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_dataset(n: int) -> list[dict]:
    random.seed(SEED)
    rows = []
    for i in range(n):
        rows.append({
            "id": i,
            "category": CATEGORIES[i % len(CATEGORIES)],
            "price": round(random.uniform(1.0, 999.99), 2),
            "stock": random.randint(0, 200),
            "rating": round(random.uniform(1.0, 5.0), 1),
            "region": REGIONS[i % len(REGIONS)],
            "title": f"Product {i}",
        })
    return rows


# ---------------------------------------------------------------------------
# ClickHouse setup
# ---------------------------------------------------------------------------


def setup_clickhouse(client, rows: list[dict], table: str = "products"):
    client.command(f"DROP TABLE IF EXISTS {table}")
    client.command(f"""
        CREATE TABLE {table} (
            id UInt64, category LowCardinality(String), price Float64,
            stock UInt32, rating Float32, region LowCardinality(String), title String
        ) ENGINE = MergeTree() ORDER BY id
    """)
    col_names = ["id", "category", "price", "stock", "rating", "region", "title"]
    for start in range(0, len(rows), 5000):
        batch = rows[start:start + 5000]
        data = [[r[c] for c in col_names] for r in batch]
        client.insert(table, data, column_names=col_names)
    return int(client.command(f"SELECT count() FROM {table}"))


# ---------------------------------------------------------------------------
# VelesDB setup
# ---------------------------------------------------------------------------


def setup_velesdb(client: VelesDBClient, rows: list[dict], col_name: str = "products"):
    client.delete_collection(col_name)
    client.create_collection(col_name, dimension=DIMENSION, metric="cosine")

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        points = []
        for r in batch:
            points.append({
                "id": r["id"],
                "vector": pseudo_embedding(r["id"], DIMENSION),
                "payload": {
                    "category": r["category"], "price": r["price"],
                    "stock": r["stock"], "rating": r["rating"],
                    "region": r["region"], "title": r["title"],
                },
            })
        client.upsert_points(col_name, points)


# ---------------------------------------------------------------------------
# Benchmark operations
# ---------------------------------------------------------------------------


def run_benchmarks(n: int, ch_client, veles: VelesDBClient, col_name: str) -> dict:
    query_vec = pseudo_embedding(9999, DIMENSION)
    results = {"dataset_size": n, "operations": {}}

    # OP1: 3-predicate filter + 4-col projection
    op_name = "filter_3pred_project_4col"
    print(f"    {op_name}...")

    def ch_op1():
        return ch_client.query(
            "SELECT category, price, title, rating FROM products "
            "WHERE category = 'tech' AND price > 100 AND stock < 50"
        ).result_rows

    def veles_op1():
        res = veles.search(col_name, vector=query_vec, top_k=TOP_K, filter={
            "condition": {"type": "and", "conditions": [
                {"type": "eq", "field": "category", "value": "tech"},
                {"type": "gt", "field": "price", "value": 100},
                {"type": "lt", "field": "stock", "value": 50},
            ]}
        })
        return [(r["payload"]["category"], r["payload"]["price"],
                 r["payload"]["title"], r["payload"]["rating"]) for r in res]

    ch_m = measure(ch_op1)
    veles_m = measure(veles_op1)
    results["operations"][op_name] = {
        "description": "WHERE category='tech' AND price>100 AND stock<50 → 4 cols",
        "clickhouse": {k: v for k, v in ch_m.items() if k != "result_sample"},
        "velesdb": {k: v for k, v in veles_m.items() if k != "result_sample"},
        "ch_result_count": len(ch_m["result_sample"]),
        "veles_result_count": len(veles_m["result_sample"]),
    }

    # OP2: Nested AND/OR filter
    op_name = "filter_nested_and_or"
    print(f"    {op_name}...")

    def ch_op2():
        return ch_client.query(
            "SELECT id, category, price, rating, region FROM products "
            "WHERE category IN ('tech', 'science') AND price >= 50 AND price <= 500 AND rating >= 3.5"
        ).result_rows

    def veles_op2():
        res = veles.search(col_name, vector=query_vec, top_k=TOP_K, filter={
            "condition": {"type": "and", "conditions": [
                {"type": "in", "field": "category", "values": ["tech", "science"]},
                {"type": "gte", "field": "price", "value": 50},
                {"type": "lte", "field": "price", "value": 500},
                {"type": "gte", "field": "rating", "value": 3.5},
            ]}
        })
        return [(r["id"], r["payload"]["category"], r["payload"]["price"],
                 r["payload"]["rating"], r["payload"]["region"]) for r in res]

    ch_m = measure(ch_op2)
    veles_m = measure(veles_op2)
    results["operations"][op_name] = {
        "description": "category IN (tech,science) AND price 50..500 AND rating>=3.5",
        "clickhouse": {k: v for k, v in ch_m.items() if k != "result_sample"},
        "velesdb": {k: v for k, v in veles_m.items() if k != "result_sample"},
        "ch_result_count": len(ch_m["result_sample"]),
        "veles_result_count": len(veles_m["result_sample"]),
    }

    # OP3: Single predicate + all columns
    op_name = "filter_1pred_project_all"
    print(f"    {op_name}...")

    def ch_op3():
        return ch_client.query(
            "SELECT * FROM products WHERE region = 'eu-west' LIMIT 100"
        ).result_rows

    def veles_op3():
        res = veles.search(col_name, vector=query_vec, top_k=TOP_K, filter={
            "condition": {"type": "eq", "field": "region", "value": "eu-west"}
        })
        return [(r["id"], r["payload"]["category"], r["payload"]["price"],
                 r["payload"]["stock"], r["payload"]["rating"],
                 r["payload"]["region"], r["payload"]["title"]) for r in res]

    ch_m = measure(ch_op3)
    veles_m = measure(veles_op3)
    results["operations"][op_name] = {
        "description": "WHERE region='eu-west' LIMIT 100 → all columns",
        "clickhouse": {k: v for k, v in ch_m.items() if k != "result_sample"},
        "velesdb": {k: v for k, v in veles_m.items() if k != "result_sample"},
        "ch_result_count": len(ch_m["result_sample"]),
        "veles_result_count": len(veles_m["result_sample"]),
    }

    # OP4: Vector search + payload
    op_name = "vector_search_payload"
    print(f"    {op_name}...")

    def ch_op4():
        return ch_client.query(
            "SELECT id, category, price, rating FROM products ORDER BY price DESC LIMIT 100"
        ).result_rows

    def veles_op4():
        res = veles.search(col_name, vector=query_vec, top_k=TOP_K)
        return [(r["id"], r["payload"]["category"], r["payload"]["price"],
                 r["payload"]["rating"]) for r in res]

    ch_m = measure(ch_op4)
    veles_m = measure(veles_op4)
    results["operations"][op_name] = {
        "description": "Top-100 retrieval + 4-col projection",
        "clickhouse": {k: v for k, v in ch_m.items() if k != "result_sample"},
        "velesdb": {k: v for k, v in veles_m.items() if k != "result_sample"},
        "note": "Different ranking strategies",
    }

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(all_results: list[dict], machine: dict):
    print("\n" + "=" * 78)
    print("  VelesDB vs ClickHouse — Multi-column Benchmark")
    print("=" * 78)
    print(f"  Runtime:  All engines in Docker, accessed via HTTP")
    print(f"  VelesDB:  {machine.get('velesdb', '?')}")
    print(f"  CH:       {machine.get('clickhouse', '?')}")
    print(f"  Rounds:   {MEASURE_ROUNDS} (warmup: {WARMUP_ROUNDS})")
    print("=" * 78)

    for res in all_results:
        n = res["dataset_size"]
        print(f"\n{'─' * 78}")
        print(f"  Dataset: {n:,} rows")

        for op_name, op_data in res["operations"].items():
            ch = op_data["clickhouse"]
            vl = op_data["velesdb"]
            if ch["median"] > 0 and vl["median"] > 0:
                ratio = ch["median"] / vl["median"]
                winner = f"VelesDB {ratio:.1f}x faster" if ratio > 1 else f"ClickHouse {1/ratio:.1f}x faster"
            else:
                winner = "N/A"

            print(f"\n  ▸ {op_name}: {op_data['description']}")
            print(f"    {'Engine':<14} {'Median':>12} {'P99':>12} {'Mean':>12}")
            print(f"    {'─' * 50}")
            print(f"    {'ClickHouse':<14} {fmt_time(ch['median']):>12} {fmt_time(ch['p99']):>12} {fmt_time(ch['mean']):>12}")
            print(f"    {'VelesDB':<14} {fmt_time(vl['median']):>12} {fmt_time(vl['p99']):>12} {fmt_time(vl['mean']):>12}")
            print(f"    → {winner}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global MEASURE_ROUNDS, WARMUP_ROUNDS

    parser = argparse.ArgumentParser(description="VelesDB vs ClickHouse multi-column benchmark")
    parser.add_argument("--datasets", type=int, nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--velesdb-host", default="localhost")
    parser.add_argument("--velesdb-port", type=int, default=8080)
    parser.add_argument("--ch-host", default="localhost")
    parser.add_argument("--ch-port", type=int, default=8123)
    args = parser.parse_args()

    MEASURE_ROUNDS = args.rounds
    WARMUP_ROUNDS = args.warmup

    print("VelesDB vs ClickHouse — Multi-column Benchmark")
    print("=" * 50)

    # Connect to VelesDB
    veles = VelesDBClient(host=args.velesdb_host, port=args.velesdb_port)
    try:
        info = veles.health()
        machine = {"velesdb": info.get("version", "?")}
        print(f"  VelesDB {machine['velesdb']} OK")
    except Exception as e:
        print(f"  ERROR VelesDB: {e}")
        sys.exit(1)

    # Connect to ClickHouse
    try:
        ch_client = clickhouse_connect.get_client(host=args.ch_host, port=args.ch_port)
        machine["clickhouse"] = str(ch_client.command("SELECT version()"))
        print(f"  ClickHouse {machine['clickhouse']} OK")
    except Exception as e:
        print(f"  ERROR ClickHouse: {e}")
        sys.exit(1)

    all_results = []
    for n in args.datasets:
        print(f"\n  Dataset: {n:,} rows")
        rows = generate_dataset(n)

        print("  Loading into ClickHouse...")
        t0 = time.perf_counter()
        setup_clickhouse(ch_client, rows)
        ch_load = time.perf_counter() - t0
        print(f"  ClickHouse: {ch_load:.2f}s")

        col_name = f"products_{n}"
        print("  Loading into VelesDB...")
        t0 = time.perf_counter()
        setup_velesdb(veles, rows, col_name)
        veles_load = time.perf_counter() - t0
        print(f"  VelesDB: {veles_load:.2f}s")

        results = run_benchmarks(n, ch_client, veles, col_name)
        results["load_times"] = {"clickhouse": ch_load, "velesdb": veles_load}
        all_results.append(results)

        veles.delete_collection(col_name)

    if args.json:
        print(json.dumps({"machine": machine, "results": all_results}, indent=2, default=str))
    else:
        print_results(all_results, machine)

    print("\n  Done.")


if __name__ == "__main__":
    main()
