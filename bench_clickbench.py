#!/usr/bin/env python3
"""
VelesDB vs ClickHouse — ClickBench-Adapted Benchmark
=====================================================

Uses REAL ClickBench data (Yandex.Metrica web analytics, 1M rows)
with queries adapted from the official 43 ClickBench queries.

Only queries that both engines can handle are compared:
  - Point lookups (Q20)
  - Text/pattern filters (Q21)
  - Multi-predicate dashboard queries (Q37-Q43)

Fairness guarantees:
  - Same dataset (ClickBench hits 1M rows)
  - Same LIMIT on both engines (equal result volume)
  - Same WSL2 environment, same Python process
  - Warmup + multiple rounds with p50/p99

Prerequisites:
    # In WSL2:
    # 1. Download ClickBench 1M subset (if not done):
    #    ~/clickhouse-bench/clickhouse local --query "
    #      SELECT * FROM url('https://datasets.clickhouse.com/hits_compatible/hits.parquet', Parquet)
    #      LIMIT 1000000
    #      INTO OUTFILE '/tmp/hits_1m.parquet' FORMAT Parquet"
    #
    # 2. Run: bash run_clickbench_wsl2.sh
"""

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_ROUNDS = 20
MEASURE_ROUNDS = 100
DIMENSION = 128
BATCH_SIZE = 5000
TOP_K = 100
PARQUET_PATH = "/var/lib/clickhouse/user_files/hits_1m.parquet"

# Columns to keep in VelesDB payload (used by our adapted queries)
PAYLOAD_COLUMNS = [
    "WatchID", "CounterID", "EventDate", "EventTime",
    "UserID", "RegionID", "URL", "Title", "Referer",
    "SearchPhrase", "SearchEngineID", "AdvEngineID",
    "TraficSourceID", "IsRefresh", "DontCountHits",
    "IsLink", "IsDownload", "URLHash", "RefererHash",
    "WindowClientWidth", "WindowClientHeight",
    "ResolutionWidth", "IsMobile", "OS",
]

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
        "_result": result,
    }


# ---------------------------------------------------------------------------
# Load ClickBench data
# ---------------------------------------------------------------------------


def load_clickbench_rows(ch_client) -> list[dict]:
    """Load rows from ClickHouse (already imported from parquet)."""
    print("  Reading rows from ClickHouse for VelesDB import...")
    cols = ", ".join(PAYLOAD_COLUMNS)
    result = ch_client.query(f"SELECT {cols} FROM hits")
    rows = []
    for row in result.result_rows:
        d = {}
        for i, col in enumerate(PAYLOAD_COLUMNS):
            val = row[i]
            # Convert to JSON-compatible types
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            d[col] = val
        rows.append(d)
    return rows


def setup_clickhouse(ch_client):
    """Import ClickBench parquet into ClickHouse server table."""
    ch_client.command("DROP TABLE IF EXISTS hits")
    ch_client.command("""
        CREATE TABLE hits (
            WatchID UInt64, JavaEnable UInt8, Title String, GoodEvent Int16,
            EventTime UInt32, EventDate UInt16, CounterID UInt32,
            ClientIP UInt32, RegionID UInt32, UserID UInt64,
            CounterClass Int8, OS UInt8, UserAgent UInt8,
            URL String, Referer String, IsRefresh UInt8,
            RefererCategoryID UInt16, RefererRegionID UInt32,
            URLCategoryID UInt16, URLRegionID UInt32,
            ResolutionWidth UInt16, ResolutionHeight UInt16,
            ResolutionDepth UInt8, FlashMajor UInt8, FlashMinor UInt8,
            FlashMinor2 String, NetMajor UInt8, NetMinor UInt8,
            UserAgentMajor UInt16, UserAgentMinor String,
            CookieEnable UInt8, JavascriptEnable UInt8,
            IsMobile UInt8, MobilePhone UInt8, MobilePhoneModel String,
            Params String, IPNetworkID UInt32, TraficSourceID Int8,
            SearchEngineID UInt16, SearchPhrase String,
            AdvEngineID UInt8, IsArtificial UInt8, WindowClientWidth UInt16,
            WindowClientHeight UInt16, ClientTimeZone Int16,
            ClientEventTime UInt32, SilverlightVersion1 UInt8,
            SilverlightVersion2 UInt8, SilverlightVersion3 UInt32,
            SilverlightVersion4 UInt16, PageCharset String,
            CodeVersion UInt32, IsLink UInt8, IsDownload UInt8,
            IsNotBounce UInt8, FUniqID UInt64, OriginalURL String,
            HID UInt32, IsOldCounter UInt8, IsEvent UInt8,
            IsParameter UInt8, DontCountHits UInt8, WithHash UInt8,
            HitColor String, LocalEventTime UInt32, Age UInt8,
            Sex UInt8, Income UInt8, Interests UInt16,
            Robotness UInt8, RemoteIP UInt32, WindowName Int32,
            OpenerName Int32, HistoryLength Int16, BrowserLanguage String,
            BrowserCountry String, SocialNetwork String,
            SocialAction String, HTTPError UInt16, SendTiming UInt32,
            DNSTiming UInt32, ConnectTiming UInt32,
            ResponseStartTiming UInt32, ResponseEndTiming UInt32,
            FetchTiming UInt32, SocialSourceNetworkID UInt8,
            SocialSourcePage String, ParamPrice Int64,
            ParamOrderID String, ParamCurrency String,
            ParamCurrencyID UInt16, OpenstatServiceName String,
            OpenstatCampaignID String, OpenstatAdID String,
            OpenstatSourceID String, UTMSource String,
            UTMMedium String, UTMCampaign String, UTMContent String,
            UTMTerm String, FromTag String, HasGCLID UInt8,
            RefererHash UInt64, URLHash UInt64, CLID UInt32
        ) ENGINE = MergeTree()
        ORDER BY (CounterID, EventDate, UserID, EventTime, WatchID)
    """)

    # Import from parquet file
    ch_client.command(f"""
        INSERT INTO hits
        SELECT * FROM file('{PARQUET_PATH}', Parquet)
    """)
    count = int(ch_client.command("SELECT count() FROM hits"))
    return count


def setup_velesdb(rows: list[dict], db_path: str):
    """Insert ClickBench rows into VelesDB with synthetic embeddings."""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = velesdb.Database(db_path)
    collection = db.create_collection("hits", dimension=DIMENSION, metric="cosine")

    # Create secondary indexes on frequently filtered columns.
    # This enables the bitmap pre-filter path (Issue #487) which
    # dramatically improves filtered search performance.
    for col_name in ["CounterID", "UserID", "IsMobile", "IsRefresh",
                     "DontCountHits", "IsLink", "IsDownload",
                     "AdvEngineID", "TraficSourceID", "SearchEngineID"]:
        try:
            collection.create_index(col_name)
        except Exception:
            pass  # Some columns may not support indexing

    total = len(rows)
    batch_size = BATCH_SIZE
    import numpy as np
    for start in range(0, total, batch_size):
        batch = rows[start:start + batch_size]
        # Try fast numpy+JSON path first (10-50x faster than dict path)
        try:
            import json as _json
            n_batch = len(batch)
            vectors_np = np.array(
                [pseudo_embedding(start + i, DIMENSION) for i in range(n_batch)],
                dtype=np.float32,
            )
            ids_list = list(range(start, start + n_batch))
            json_payloads = [_json.dumps(row) for row in batch]
            collection._inner.upsert_bulk_numpy_json(
                vectors_np, ids_list, json_payloads
            )
        except (AttributeError, TypeError):
            # Fallback to dict path
            points = []
            for i, row in enumerate(batch):
                idx = start + i
                points.append({
                    "id": idx,
                    "vector": pseudo_embedding(idx, DIMENSION),
                    "payload": row,
                })
            collection.upsert(points)
        if (start + batch_size) % 50000 == 0 or start + batch_size >= total:
            print(f"    {min(start + batch_size, total):,}/{total:,} inserted")

    return collection, db


# ---------------------------------------------------------------------------
# Adapted ClickBench Queries
# Both engines get the SAME LIMIT for equal volume.
# ---------------------------------------------------------------------------


def define_queries(ch_client, veles_col, veles_db):
    """
    Returns a list of (name, description, ch_func, veles_func, original_clickbench_q).
    All queries have matched LIMIT for equal result volume.
    """
    query_vec = pseudo_embedding(42, DIMENSION)
    queries = []

    # --- Q20: Point lookup ---
    # Original: SELECT UserID FROM hits WHERE UserID = 435090932899640449
    # We pick a UserID that exists in our 1M subset
    sample_uid = int(ch_client.command(
        "SELECT UserID FROM hits WHERE UserID != 0 LIMIT 1"
    ))

    def ch_q20():
        return ch_client.query(
            f"SELECT UserID FROM hits WHERE UserID = {sample_uid} LIMIT {TOP_K}"
        ).result_rows

    def veles_q20():
        res = veles_db.execute_query(
            f"SELECT UserID FROM hits WHERE UserID = {sample_uid} LIMIT {TOP_K}",
            {},
        )
        return [(r.get("payload", {}).get("UserID"),) for r in res]

    queries.append(("Q20_point_lookup", f"WHERE UserID = X → LIMIT {TOP_K}",
                     ch_q20, veles_q20, "Q20"))

    # --- Q21: Text/pattern filter ---
    # Original: SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%'
    # Adapted: return matching rows instead of COUNT
    def ch_q21():
        return ch_client.query(
            f"SELECT WatchID, URL FROM hits WHERE URL LIKE '%google%' LIMIT {TOP_K}"
        ).result_rows

    def veles_q21():
        res = veles_db.execute_query(
            f"SELECT WatchID, URL FROM hits WHERE URL LIKE '%google%' LIMIT {TOP_K}",
            {},
        )
        return [(r.get("payload", {}).get("WatchID"), r.get("payload", {}).get("URL")) for r in res]

    queries.append(("Q21_text_filter", f"WHERE URL LIKE '%google%' → LIMIT {TOP_K}",
                     ch_q21, veles_q21, "Q21"))

    # --- Q24: Full row + pattern filter ---
    # Original: SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10
    # Keep LIMIT 10 as original ClickBench spec (this is a "top 10" query)
    def ch_q24():
        return ch_client.query(
            "SELECT WatchID, CounterID, UserID, URL, Title, EventTime "
            "FROM hits WHERE URL LIKE '%google%' LIMIT 10"
        ).result_rows

    def veles_q24():
        res = veles_db.execute_query(
            "SELECT WatchID, CounterID, UserID, URL, Title, EventTime "
            "FROM hits WHERE URL LIKE '%google%' LIMIT 10",
            {},
        )
        return [
            (r.get("payload", {}).get("WatchID"), r.get("payload", {}).get("CounterID"),
             r.get("payload", {}).get("UserID"), r.get("payload", {}).get("URL"),
             r.get("payload", {}).get("Title"), r.get("payload", {}).get("EventTime"))
            for r in res
        ]

    queries.append(("Q24_pattern_projection", "WHERE URL LIKE '%google%' → 6 cols LIMIT 10",
                     ch_q24, veles_q24, "Q24"))

    # --- Q37: Dashboard multi-predicate (the key comparison!) ---
    # Original: SELECT URL, COUNT(*) AS PageViews FROM hits
    #   WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
    #   AND DontCountHits = 0 AND IsRefresh = 0 AND URL <> ''
    #   GROUP BY URL ORDER BY PageViews DESC LIMIT 10
    # Adapted: skip GROUP BY (not VelesDB's strength), keep the 6-predicate filter
    def ch_q37():
        return ch_client.query(
            f"SELECT URL, Title, CounterID FROM hits "
            f"WHERE CounterID = 62 "
            f"AND DontCountHits = 0 AND IsRefresh = 0 "
            f"AND URL != '' "
            f"LIMIT {TOP_K}"
        ).result_rows

    def veles_q37():
        res = veles_db.execute_query(
            f"SELECT URL, Title, CounterID FROM hits WHERE "
            f"CounterID = 62 AND DontCountHits = 0 AND IsRefresh = 0 AND URL != '' LIMIT {TOP_K}",
            {},
        )
        return [(r.get("payload", {}).get("URL"), r.get("payload", {}).get("Title"), r.get("payload", {}).get("CounterID")) for r in res]

    queries.append(("Q37_dashboard_4pred", f"CounterID=62 AND DontCountHits=0 AND IsRefresh=0 AND URL!='' → 3 cols LIMIT {TOP_K}",
                     ch_q37, veles_q37, "Q37"))

    # --- Q38: Dashboard with Title ---
    def ch_q38():
        return ch_client.query(
            f"SELECT Title, CounterID FROM hits "
            f"WHERE CounterID = 62 "
            f"AND DontCountHits = 0 AND IsRefresh = 0 "
            f"AND Title != '' "
            f"LIMIT {TOP_K}"
        ).result_rows

    def veles_q38():
        res = veles_db.execute_query(
            f"SELECT Title, CounterID FROM hits WHERE "
            f"CounterID = 62 AND DontCountHits = 0 AND IsRefresh = 0 AND Title != '' LIMIT {TOP_K}",
            {},
        )
        return [(r.get("payload", {}).get("Title"), r.get("payload", {}).get("CounterID")) for r in res]

    queries.append(("Q38_dashboard_title", f"CounterID=62 AND flags AND Title!='' → 2 cols LIMIT {TOP_K}",
                     ch_q38, veles_q38, "Q38"))

    # --- Q39: Dashboard with IsLink filter ---
    def ch_q39():
        return ch_client.query(
            f"SELECT URL, CounterID FROM hits "
            f"WHERE CounterID = 62 "
            f"AND IsRefresh = 0 AND IsLink != 0 AND IsDownload = 0 "
            f"LIMIT {TOP_K}"
        ).result_rows

    def veles_q39():
        res = veles_db.execute_query(
            f"SELECT URL, CounterID FROM hits WHERE "
            f"CounterID = 62 AND IsRefresh = 0 AND IsLink != 0 AND IsDownload = 0 LIMIT {TOP_K}",
            {},
        )
        return [(r.get("payload", {}).get("URL"), r.get("payload", {}).get("CounterID")) for r in res]

    queries.append(("Q39_dashboard_links", f"CounterID=62 AND IsRefresh=0 AND IsLink!=0 AND IsDownload=0 → LIMIT {TOP_K}",
                     ch_q39, veles_q39, "Q39"))

    # --- Q41: Dashboard with TraficSourceID IN + RefererHash ---
    def ch_q41():
        return ch_client.query(
            f"SELECT URLHash, CounterID FROM hits "
            f"WHERE CounterID = 62 "
            f"AND IsRefresh = 0 "
            f"AND TraficSourceID IN (-1, 6) "
            f"LIMIT {TOP_K}"
        ).result_rows

    def veles_q41():
        res = veles_db.execute_query(
            f"SELECT URLHash, CounterID FROM hits WHERE "
            f"CounterID = 62 AND IsRefresh = 0 AND (TraficSourceID = -1 OR TraficSourceID = 6) LIMIT {TOP_K}",
            {},
        )
        return [(r.get("payload", {}).get("URLHash"), r.get("payload", {}).get("CounterID")) for r in res]

    queries.append(("Q41_dashboard_traffic", f"CounterID=62 AND IsRefresh=0 AND TraficSource IN(-1,6) → LIMIT {TOP_K}",
                     ch_q41, veles_q41, "Q41"))

    # --- Extra: Wide filter (AdvEngine + Search) — high selectivity ---
    def ch_adv():
        return ch_client.query(
            f"SELECT WatchID, SearchPhrase, AdvEngineID, URL FROM hits "
            f"WHERE AdvEngineID != 0 AND SearchPhrase != '' "
            f"LIMIT {TOP_K}"
        ).result_rows

    def veles_adv():
        res = veles_db.execute_query(
            f"SELECT WatchID, SearchPhrase, AdvEngineID, URL FROM hits WHERE "
            f"AdvEngineID != 0 AND SearchPhrase != '' LIMIT {TOP_K}",
            {},
        )
        return [
            (r.get("payload", {}).get("WatchID"), r.get("payload", {}).get("SearchPhrase"),
             r.get("payload", {}).get("AdvEngineID"), r.get("payload", {}).get("URL"))
            for r in res
        ]

    queries.append(("Qx_adv_search", f"AdvEngineID!=0 AND SearchPhrase!='' → 4 cols LIMIT {TOP_K}",
                     ch_adv, veles_adv, "Custom"))

    # --- Extra: Single predicate high-selectivity ---
    def ch_mobile():
        return ch_client.query(
            f"SELECT WatchID, OS, UserID FROM hits "
            f"WHERE IsMobile = 1 "
            f"LIMIT {TOP_K}"
        ).result_rows

    def veles_mobile():
        res = veles_db.execute_query(
            f"SELECT WatchID, OS, UserID FROM hits WHERE "
            f"IsMobile = 1 LIMIT {TOP_K}",
            {},
        )
        return [(r.get("payload", {}).get("WatchID"), r.get("payload", {}).get("OS"), r.get("payload", {}).get("UserID")) for r in res]

    queries.append(("Qx_mobile_filter", f"WHERE IsMobile=1 → 3 cols LIMIT {TOP_K}",
                     ch_mobile, veles_mobile, "Custom"))

    return queries


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[dict], machine: dict, row_count: int):
    print("\n" + "=" * 78)
    print("  VelesDB vs ClickHouse — ClickBench-Adapted Benchmark")
    print("=" * 78)
    print(f"  Dataset:  ClickBench hits ({row_count:,} rows, real Yandex.Metrica data)")
    print(f"  OS:       {machine.get('os', 'WSL2')}")
    print(f"  VelesDB:  {machine.get('velesdb', '?')}")
    print(f"  CH:       {machine.get('clickhouse', '?')}")
    print(f"  Rounds:   {MEASURE_ROUNDS} (warmup: {WARMUP_ROUNDS})")
    print(f"  Top-K:    {TOP_K}")
    print(f"  Fairness: Same LIMIT on both, same data, same process")
    print("=" * 78)

    for r in results:
        ch = r["clickhouse"]
        vl = r["velesdb"]

        ratio = ch["median"] / vl["median"] if vl["median"] > 0 else 0
        if ratio > 1:
            winner = f"VelesDB {ratio:.1f}x faster"
        elif ratio > 0:
            winner = f"ClickHouse {1/ratio:.1f}x faster"
        else:
            winner = "N/A"

        print(f"\n  [{r['origin']}] {r['name']}")
        print(f"    {r['description']}")
        print(f"    {'Engine':<14} {'Median':>12} {'P99':>12} {'Mean':>12} {'Min':>12}")
        print(f"    {'─' * 62}")
        print(f"    {'ClickHouse':<14} {fmt_time(ch['median']):>12} {fmt_time(ch['p99']):>12} "
              f"{fmt_time(ch['mean']):>12} {fmt_time(ch['min']):>12}")
        print(f"    {'VelesDB':<14} {fmt_time(vl['median']):>12} {fmt_time(vl['p99']):>12} "
              f"{fmt_time(vl['mean']):>12} {fmt_time(vl['min']):>12}")
        print(f"    → {winner}")
        print(f"    Results: CH={r['ch_count']}, VelesDB={r['veles_count']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global MEASURE_ROUNDS, WARMUP_ROUNDS, TOP_K

    parser = argparse.ArgumentParser(description="VelesDB vs ClickHouse — ClickBench adapted")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--ch-host", default="localhost")
    parser.add_argument("--ch-port", type=int, default=8123)
    parser.add_argument("--parquet", default=PARQUET_PATH)
    parser.add_argument("--skip-ch-import", action="store_true",
                        help="Skip ClickHouse import (data already loaded by bash script)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Result limit for both engines (default: 100)")
    args = parser.parse_args()

    MEASURE_ROUNDS = args.rounds
    WARMUP_ROUNDS = args.warmup
    TOP_K = args.top_k

    print("VelesDB vs ClickHouse — ClickBench-Adapted Benchmark")
    print("=" * 55)

    machine = {
        "os": "WSL2",
        "velesdb": velesdb.__version__,
    }

    # Connect to ClickHouse
    print("\n  Connecting to ClickHouse...")
    try:
        ch_client = clickhouse_connect.get_client(host=args.ch_host, port=args.ch_port)
        ch_ver = str(ch_client.command("SELECT version()"))
        machine["clickhouse"] = ch_ver
        print(f"  ClickHouse {ch_ver}")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Import ClickBench data into ClickHouse
    if args.skip_ch_import:
        row_count = int(ch_client.command("SELECT count() FROM hits"))
        ch_load = 0.0
        print(f"\n  ClickHouse: {row_count:,} rows (pre-loaded)")
    else:
        print(f"\n  Importing ClickBench data from {args.parquet}...")
        t0 = time.perf_counter()
        row_count = setup_clickhouse(ch_client)
        ch_load = time.perf_counter() - t0
        print(f"  ClickHouse: {row_count:,} rows in {ch_load:.2f}s")

    # Read rows for VelesDB
    t0 = time.perf_counter()
    rows = load_clickbench_rows(ch_client)
    read_time = time.perf_counter() - t0
    print(f"  Read {len(rows):,} rows in {read_time:.2f}s")

    # Setup VelesDB
    db_path = "/tmp/velesdb_clickbench"
    print(f"\n  Loading into VelesDB ({len(PAYLOAD_COLUMNS)} columns per point)...")
    t0 = time.perf_counter()
    veles_col, veles_db = setup_velesdb(rows, db_path)
    veles_load = time.perf_counter() - t0
    print(f"  VelesDB: {len(rows):,} points in {veles_load:.2f}s")
    del rows  # Free memory

    # Define and run queries
    print(f"\n  Running {MEASURE_ROUNDS} rounds per query (warmup: {WARMUP_ROUNDS})...")
    queries = define_queries(ch_client, veles_col, veles_db)

    all_results = []
    for name, desc, ch_fn, veles_fn, origin in queries:
        print(f"    {name}...")
        ch_m = measure(ch_fn)
        veles_m = measure(veles_fn)

        all_results.append({
            "name": name,
            "description": desc,
            "origin": origin,
            "clickhouse": {k: v for k, v in ch_m.items() if k != "_result"},
            "velesdb": {k: v for k, v in veles_m.items() if k != "_result"},
            "ch_count": len(ch_m["_result"]) if ch_m["_result"] else 0,
            "veles_count": len(veles_m["_result"]) if veles_m["_result"] else 0,
        })

    # Output
    if args.json:
        output = {
            "benchmark": "clickbench-adapted",
            "machine": machine,
            "dataset": {"source": "ClickBench hits", "rows": row_count},
            "config": {"rounds": MEASURE_ROUNDS, "warmup": WARMUP_ROUNDS,
                       "dimension": DIMENSION, "top_k": TOP_K,
                       "payload_columns": len(PAYLOAD_COLUMNS)},
            "load_times": {"clickhouse": ch_load, "velesdb": veles_load},
            "results": all_results,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_results(all_results, machine, row_count)

    # Cleanup
    del veles_col, veles_db
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    print("\n  Done.")


if __name__ == "__main__":
    main()
