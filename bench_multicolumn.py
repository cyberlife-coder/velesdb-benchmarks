#!/usr/bin/env python3
"""
VelesDB vs ClickHouse — Multi-column Benchmark
================================================

Compares multi-column filter + projection performance between:
  - VelesDB (vector DB with structured payload filtering)
  - ClickHouse (columnar OLAP engine)

Both engines are tested from Python with identical timing methodology.

Prerequisites:
    docker compose up -d                     # Start ClickHouse
    pip install -r requirements.txt          # Install deps
    python bench_multicolumn.py              # Run benchmark

    # Custom dataset sizes:
    python bench_multicolumn.py --datasets 10000 100000 1000000

    # JSON output:
    python bench_multicolumn.py --json
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Dependencies check
# ---------------------------------------------------------------------------

try:
    import velesdb
except ImportError:
    print("ERROR: velesdb not installed. pip install velesdb")
    sys.exit(1)

try:
    import clickhouse_connect
except ImportError:
    print("ERROR: clickhouse-connect not installed. pip install clickhouse-connect")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("WARNING: numpy not installed, using pure Python vectors (slower generation)")
    np = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
DIMENSION = 128
TOP_K = 100
WARMUP_ROUNDS = 10
MEASURE_ROUNDS = 100
DEFAULT_DATASETS = [10_000, 100_000]
BATCH_SIZE = 5000

CATEGORIES = ["tech", "science", "business", "sports", "health", "music", "art", "food"]
REGIONS = ["eu-west", "eu-east", "us-west", "us-east", "asia-pac",
           "latam", "africa", "mena", "oceania", "nordic"]

CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pseudo_embedding(i: int, dim: int = DIMENSION) -> list:
    return [(math.sin(i * 0.01 + j * 0.01)) for j in range(dim)]


def get_machine_info() -> dict:
    info = {
        "cpu": "unknown",
        "ram": "unknown",
        "os": platform.platform(),
        "python": platform.python_version(),
        "velesdb": velesdb.__version__,
        "date": time.strftime("%Y-%m-%d"),
    }
    if platform.system() == "Windows":
        try:
            r = subprocess.check_output(
                ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True, timeout=10,
            ).strip()
            if r:
                info["cpu"] = r
        except Exception:
            pass
        try:
            r = subprocess.check_output(
                ["powershell", "-Command",
                 "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)"],
                text=True, timeout=10,
            ).strip()
            if r:
                info["ram"] = f"{r} GB"
        except Exception:
            pass
    return info


def percentile(data, p):
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def fmt_us(seconds: float) -> str:
    return f"{seconds * 1_000_000:.0f} µs"


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1_000:.2f} ms"


def fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return fmt_us(seconds)
    return fmt_ms(seconds)


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
        "max": max(times),
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
            id       UInt64,
            category LowCardinality(String),
            price    Float64,
            stock    UInt32,
            rating   Float32,
            region   LowCardinality(String),
            title    String
        ) ENGINE = MergeTree()
        ORDER BY id
    """)

    # Batch insert
    col_names = ["id", "category", "price", "stock", "rating", "region", "title"]
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        data = [[r[c] for c in col_names] for r in batch]
        client.insert(table, data, column_names=col_names)

    count = client.command(f"SELECT count() FROM {table}")
    return int(count)


# ---------------------------------------------------------------------------
# VelesDB setup
# ---------------------------------------------------------------------------


def setup_velesdb(rows: list[dict], db_path: str) -> "velesdb.Collection":
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = velesdb.Database(db_path)
    collection = db.create_collection("products", dimension=DIMENSION, metric="cosine")

    # Batch insert with embeddings
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        points = []
        for r in batch:
            points.append({
                "id": r["id"],
                "vector": pseudo_embedding(r["id"], DIMENSION),
                "payload": {
                    "category": r["category"],
                    "price": r["price"],
                    "stock": r["stock"],
                    "rating": r["rating"],
                    "region": r["region"],
                    "title": r["title"],
                },
            })
        collection.upsert(points)

    return collection


# ---------------------------------------------------------------------------
# Benchmark operations
# ---------------------------------------------------------------------------


def run_benchmarks(n: int, ch_client, veles_col) -> dict:
    query_vec = pseudo_embedding(9999, DIMENSION)
    results = {"dataset_size": n, "operations": {}}

    # =====================================================================
    # OP1: Multi-predicate filter + multi-column projection
    #   "tech" category, price > 100, stock < 50
    #   Project: category, price, title, rating
    # =====================================================================
    op_name = "filter_3pred_project_4col"
    print(f"    {op_name}...")

    def ch_op1():
        return ch_client.query(
            "SELECT category, price, title, rating "
            "FROM products "
            "WHERE category = 'tech' AND price > 100 AND stock < 50"
        ).result_rows

    def veles_op1():
        res = veles_col.search(
            vector=query_vec,
            top_k=TOP_K,
            filter={
                "condition": {
                    "type": "and",
                    "conditions": [
                        {"type": "eq", "field": "category", "value": "tech"},
                        {"type": "gt", "field": "price", "value": 100},
                        {"type": "lt", "field": "stock", "value": 50},
                    ],
                }
            },
        )
        # Project 4 columns from payload
        return [
            (r["payload"]["category"], r["payload"]["price"],
             r["payload"]["title"], r["payload"]["rating"])
            for r in res
        ]

    ch_m = measure(ch_op1)
    veles_m = measure(veles_op1)
    results["operations"][op_name] = {
        "description": "WHERE category='tech' AND price>100 AND stock<50 → project 4 cols",
        "clickhouse": {k: v for k, v in ch_m.items() if k != "result_sample"},
        "velesdb": {k: v for k, v in veles_m.items() if k != "result_sample"},
        "ch_result_count": len(ch_m["result_sample"]),
        "veles_result_count": len(veles_m["result_sample"]),
    }

    # =====================================================================
    # OP2: Complex nested filter (AND + OR + range)
    #   (category IN [tech, science]) AND (price BETWEEN 50..500) AND (rating >= 3.5)
    # =====================================================================
    op_name = "filter_nested_and_or"
    print(f"    {op_name}...")

    def ch_op2():
        return ch_client.query(
            "SELECT id, category, price, rating, region "
            "FROM products "
            "WHERE category IN ('tech', 'science') "
            "  AND price >= 50 AND price <= 500 "
            "  AND rating >= 3.5"
        ).result_rows

    def veles_op2():
        res = veles_col.search(
            vector=query_vec,
            top_k=TOP_K,
            filter={
                "condition": {
                    "type": "and",
                    "conditions": [
                        {"type": "in", "field": "category", "values": ["tech", "science"]},
                        {"type": "gte", "field": "price", "value": 50},
                        {"type": "lte", "field": "price", "value": 500},
                        {"type": "gte", "field": "rating", "value": 3.5},
                    ],
                }
            },
        )
        return [
            (r["id"], r["payload"]["category"], r["payload"]["price"],
             r["payload"]["rating"], r["payload"]["region"])
            for r in res
        ]

    ch_m = measure(ch_op2)
    veles_m = measure(veles_op2)
    results["operations"][op_name] = {
        "description": "category IN (tech,science) AND price 50..500 AND rating>=3.5 → 5 cols",
        "clickhouse": {k: v for k, v in ch_m.items() if k != "result_sample"},
        "velesdb": {k: v for k, v in veles_m.items() if k != "result_sample"},
        "ch_result_count": len(ch_m["result_sample"]),
        "veles_result_count": len(veles_m["result_sample"]),
    }

    # =====================================================================
    # OP3: Single predicate + full row projection (all columns)
    # =====================================================================
    op_name = "filter_1pred_project_all"
    print(f"    {op_name}...")

    def ch_op3():
        return ch_client.query(
            "SELECT * FROM products WHERE region = 'eu-west' LIMIT 100"
        ).result_rows

    def veles_op3():
        res = veles_col.search(
            vector=query_vec,
            top_k=TOP_K,
            filter={
                "condition": {"type": "eq", "field": "region", "value": "eu-west"}
            },
        )
        return [
            (r["id"], r["payload"]["category"], r["payload"]["price"],
             r["payload"]["stock"], r["payload"]["rating"],
             r["payload"]["region"], r["payload"]["title"])
            for r in res
        ]

    ch_m = measure(ch_op3)
    veles_m = measure(veles_op3)
    results["operations"][op_name] = {
        "description": "WHERE region='eu-west' LIMIT 100 → all columns",
        "clickhouse": {k: v for k, v in ch_m.items() if k != "result_sample"},
        "velesdb": {k: v for k, v in veles_m.items() if k != "result_sample"},
        "ch_result_count": len(ch_m["result_sample"]),
        "veles_result_count": len(veles_m["result_sample"]),
    }

    # =====================================================================
    # OP4: Aggregation — COUNT + AVG grouped by category
    # =====================================================================
    op_name = "aggregation_group_by"
    print(f"    {op_name}...")

    def ch_op4():
        return ch_client.query(
            "SELECT category, count() AS cnt, avg(price) AS avg_price "
            "FROM products GROUP BY category ORDER BY cnt DESC"
        ).result_rows

    # VelesDB: manual aggregation over search results (no native GROUP BY via Python)
    def veles_op4():
        # Fetch large result set, aggregate in Python
        res = veles_col.search(vector=query_vec, top_k=TOP_K)
        groups = {}
        for r in res:
            cat = r["payload"]["category"]
            price = r["payload"]["price"]
            if cat not in groups:
                groups[cat] = {"cnt": 0, "sum_price": 0.0}
            groups[cat]["cnt"] += 1
            groups[cat]["sum_price"] += price
        return [
            (cat, g["cnt"], g["sum_price"] / g["cnt"] if g["cnt"] else 0)
            for cat, g in sorted(groups.items(), key=lambda x: -x[1]["cnt"])
        ]

    ch_m = measure(ch_op4)
    veles_m = measure(veles_op4)
    results["operations"][op_name] = {
        "description": "GROUP BY category → count, avg(price) [CH: full table / VelesDB: top-K]",
        "clickhouse": {k: v for k, v in ch_m.items() if k != "result_sample"},
        "velesdb": {k: v for k, v in veles_m.items() if k != "result_sample"},
        "note": "Not directly comparable: CH aggregates full table, VelesDB aggregates top-K results",
    }

    # =====================================================================
    # OP5: Pure vector search + payload access (VelesDB advantage)
    # =====================================================================
    op_name = "vector_search_payload"
    print(f"    {op_name}...")

    def ch_op5():
        # ClickHouse has no vector search — simulate with ORDER BY + LIMIT
        return ch_client.query(
            "SELECT id, category, price, rating FROM products ORDER BY price DESC LIMIT 100"
        ).result_rows

    def veles_op5():
        res = veles_col.search(vector=query_vec, top_k=TOP_K)
        return [
            (r["id"], r["payload"]["category"], r["payload"]["price"], r["payload"]["rating"])
            for r in res
        ]

    ch_m = measure(ch_op5)
    veles_m = measure(veles_op5)
    results["operations"][op_name] = {
        "description": "Top-100 retrieval + 4-col projection [VelesDB: vector HNSW / CH: ORDER BY]",
        "clickhouse": {k: v for k, v in ch_m.items() if k != "result_sample"},
        "velesdb": {k: v for k, v in veles_m.items() if k != "result_sample"},
        "note": "Different ranking strategies — shows retrieval + projection overhead",
    }

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(all_results: list[dict], machine: dict):
    print("\n" + "=" * 78)
    print("  VelesDB vs ClickHouse — Multi-column Benchmark Results")
    print("=" * 78)
    print(f"  CPU:     {machine['cpu']}")
    print(f"  RAM:     {machine['ram']}")
    print(f"  OS:      {machine['os']}")
    print(f"  VelesDB: {machine['velesdb']}")
    print(f"  Date:    {machine['date']}")
    print(f"  Rounds:  {MEASURE_ROUNDS} (warmup: {WARMUP_ROUNDS})")
    print("=" * 78)

    for res in all_results:
        n = res["dataset_size"]
        print(f"\n{'─' * 78}")
        print(f"  Dataset: {n:,} rows")
        print(f"{'─' * 78}")

        for op_name, op_data in res["operations"].items():
            print(f"\n  ▸ {op_name}")
            print(f"    {op_data['description']}")

            ch = op_data["clickhouse"]
            vl = op_data["velesdb"]

            # Ratio
            if ch["median"] > 0 and vl["median"] > 0:
                ratio = ch["median"] / vl["median"]
                if ratio > 1:
                    winner = f"VelesDB {ratio:.1f}x faster"
                else:
                    winner = f"ClickHouse {1/ratio:.1f}x faster"
            else:
                winner = "N/A"

            header = f"    {'Engine':<14} {'Median':>12} {'P99':>12} {'Mean':>12} {'Min':>12}"
            print(header)
            print(f"    {'─' * 62}")
            print(f"    {'ClickHouse':<14} {fmt_time(ch['median']):>12} {fmt_time(ch['p99']):>12} "
                  f"{fmt_time(ch['mean']):>12} {fmt_time(ch['min']):>12}")
            print(f"    {'VelesDB':<14} {fmt_time(vl['median']):>12} {fmt_time(vl['p99']):>12} "
                  f"{fmt_time(vl['mean']):>12} {fmt_time(vl['min']):>12}")
            print(f"    → {winner}")

            if "note" in op_data:
                print(f"    ⚠ {op_data['note']}")

            if "ch_result_count" in op_data:
                print(f"    Results: CH={op_data['ch_result_count']}, "
                      f"VelesDB={op_data['veles_result_count']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global MEASURE_ROUNDS, WARMUP_ROUNDS

    parser = argparse.ArgumentParser(description="VelesDB vs ClickHouse multi-column benchmark")
    parser.add_argument("--datasets", type=int, nargs="+", default=DEFAULT_DATASETS,
                        help="Dataset sizes to benchmark (default: 10000 100000)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--rounds", type=int, default=100, help="Measurement rounds (default: 100)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup rounds (default: 10)")
    parser.add_argument("--ch-host", default=CLICKHOUSE_HOST, help="ClickHouse host")
    parser.add_argument("--ch-port", type=int, default=CLICKHOUSE_PORT, help="ClickHouse HTTP port")
    args = parser.parse_args()

    MEASURE_ROUNDS = args.rounds
    WARMUP_ROUNDS = args.warmup

    print("VelesDB vs ClickHouse — Multi-column Benchmark")
    print("=" * 50)

    # Machine info
    machine = get_machine_info()
    print(f"  CPU: {machine['cpu']}")
    print(f"  RAM: {machine['ram']}")

    # Connect to ClickHouse
    print("\n  Connecting to ClickHouse...")
    try:
        ch_client = clickhouse_connect.get_client(
            host=args.ch_host, port=args.ch_port, database="default"
        )
        ch_version = ch_client.command("SELECT version()")
        print(f"  ClickHouse version: {ch_version}")
        machine["clickhouse"] = str(ch_version)
    except Exception as e:
        print(f"\n  ERROR: Cannot connect to ClickHouse at {args.ch_host}:{args.ch_port}")
        print(f"  {e}")
        print(f"\n  Make sure ClickHouse is running:")
        print(f"    docker compose up -d")
        sys.exit(1)

    all_results = []

    for n in args.datasets:
        print(f"\n{'=' * 50}")
        print(f"  Dataset: {n:,} rows")
        print(f"{'=' * 50}")

        # Generate data
        print("  Generating dataset...")
        rows = generate_dataset(n)

        # Setup ClickHouse
        print("  Loading into ClickHouse...")
        t0 = time.perf_counter()
        ch_count = setup_clickhouse(ch_client, rows)
        ch_load_time = time.perf_counter() - t0
        print(f"  ClickHouse: {ch_count:,} rows loaded in {ch_load_time:.2f}s")

        # Setup VelesDB
        db_path = os.path.join(os.environ.get("TEMP", "/tmp"), f"velesdb_bench_vs_{n}")
        print("  Loading into VelesDB...")
        t0 = time.perf_counter()
        veles_col = setup_velesdb(rows, db_path)
        veles_load_time = time.perf_counter() - t0
        print(f"  VelesDB: {n:,} points loaded in {veles_load_time:.2f}s")

        # Run benchmarks
        print("\n  Running benchmarks...")
        results = run_benchmarks(n, ch_client, veles_col)
        results["load_times"] = {
            "clickhouse": ch_load_time,
            "velesdb": veles_load_time,
        }
        all_results.append(results)

        # Cleanup VelesDB
        del veles_col
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)

    # Output
    if args.json:
        output = {
            "machine": machine,
            "config": {
                "dimension": DIMENSION,
                "top_k": TOP_K,
                "warmup_rounds": WARMUP_ROUNDS,
                "measure_rounds": MEASURE_ROUNDS,
            },
            "results": all_results,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_results(all_results, machine)

    print("\n  Done.")


if __name__ == "__main__":
    main()
