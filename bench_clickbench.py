#!/usr/bin/env python3
"""
VelesDB vs ClickHouse — ClickBench-Adapted Benchmark
=====================================================

Uses REAL ClickBench data (Yandex.Metrica, 1M rows).
Both engines run in Docker, accessed via HTTP.
"""

import argparse
import json
import math
import statistics
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

WARMUP_ROUNDS = 20
MEASURE_ROUNDS = 100
DIMENSION = 128
BATCH_SIZE = 1000
TOP_K = 100
PARQUET_PATH = "/var/lib/clickhouse/user_files/hits_1m.parquet"

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
# Setup
# ---------------------------------------------------------------------------


def setup_clickhouse(ch_client, parquet_path):
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
    ch_client.command(f"INSERT INTO hits SELECT * FROM file('{parquet_path}', Parquet)")
    return int(ch_client.command("SELECT count() FROM hits"))


def load_clickbench_rows(ch_client) -> list[dict]:
    print("  Reading rows from ClickHouse for VelesDB import...")
    cols = ", ".join(PAYLOAD_COLUMNS)
    result = ch_client.query(f"SELECT {cols} FROM hits")
    rows = []
    for row in result.result_rows:
        d = {}
        for i, col in enumerate(PAYLOAD_COLUMNS):
            val = row[i]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            d[col] = val
        rows.append(d)
    return rows


def setup_velesdb(client: VelesDBClient, rows: list[dict]):
    client.delete_collection("hits")
    client.create_collection("hits", dimension=DIMENSION, metric="cosine")

    total = len(rows)
    for start in range(0, total, BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        points = []
        for i, row in enumerate(batch):
            idx = start + i
            points.append({
                "id": idx,
                "vector": pseudo_embedding(idx, DIMENSION),
                "payload": row,
            })
        client.upsert_points("hits", points)
        if (start + BATCH_SIZE) % 50000 == 0 or start + BATCH_SIZE >= total:
            print(f"    {min(start + BATCH_SIZE, total):,}/{total:,}")


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def define_queries(ch_client, veles: VelesDBClient):
    queries = []

    sample_uid = int(ch_client.command("SELECT UserID FROM hits WHERE UserID != 0 LIMIT 1"))

    def ch_q20():
        return ch_client.query(f"SELECT UserID FROM hits WHERE UserID = {sample_uid} LIMIT {TOP_K}").result_rows

    def veles_q20():
        res = veles.execute_query(f"SELECT UserID FROM hits WHERE UserID = {sample_uid} LIMIT {TOP_K}")
        return [(r.get("payload", {}).get("UserID"),) for r in res]

    queries.append(("Q20_point_lookup", f"WHERE UserID = X → LIMIT {TOP_K}", ch_q20, veles_q20, "Q20"))

    def ch_q21():
        return ch_client.query(f"SELECT WatchID, URL FROM hits WHERE URL LIKE '%google%' LIMIT {TOP_K}").result_rows

    def veles_q21():
        res = veles.execute_query(f"SELECT WatchID, URL FROM hits WHERE URL LIKE '%google%' LIMIT {TOP_K}")
        return [(r.get("payload", {}).get("WatchID"), r.get("payload", {}).get("URL")) for r in res]

    queries.append(("Q21_text_filter", f"WHERE URL LIKE '%google%' → LIMIT {TOP_K}", ch_q21, veles_q21, "Q21"))

    def ch_q37():
        return ch_client.query(
            f"SELECT URL, Title, CounterID FROM hits "
            f"WHERE CounterID = 62 AND DontCountHits = 0 AND IsRefresh = 0 AND URL != '' LIMIT {TOP_K}"
        ).result_rows

    def veles_q37():
        res = veles.execute_query(
            f"SELECT URL, Title, CounterID FROM hits WHERE "
            f"CounterID = 62 AND DontCountHits = 0 AND IsRefresh = 0 AND URL != '' LIMIT {TOP_K}")
        return [(r.get("payload", {}).get("URL"), r.get("payload", {}).get("Title"),
                 r.get("payload", {}).get("CounterID")) for r in res]

    queries.append(("Q37_dashboard_4pred", "CounterID=62 AND 3 flags → 3 cols", ch_q37, veles_q37, "Q37"))

    def ch_mobile():
        return ch_client.query(f"SELECT WatchID, OS, UserID FROM hits WHERE IsMobile = 1 LIMIT {TOP_K}").result_rows

    def veles_mobile():
        res = veles.execute_query(f"SELECT WatchID, OS, UserID FROM hits WHERE IsMobile = 1 LIMIT {TOP_K}")
        return [(r.get("payload", {}).get("WatchID"), r.get("payload", {}).get("OS"),
                 r.get("payload", {}).get("UserID")) for r in res]

    queries.append(("Qx_mobile_filter", f"WHERE IsMobile=1 → 3 cols LIMIT {TOP_K}", ch_mobile, veles_mobile, "Custom"))

    return queries


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[dict], machine: dict, row_count: int):
    print("\n" + "=" * 78)
    print("  VelesDB vs ClickHouse — ClickBench-Adapted Benchmark")
    print("=" * 78)
    print(f"  Dataset:  ClickBench hits ({row_count:,} rows)")
    print(f"  Runtime:  All engines in Docker, accessed via HTTP")
    print(f"  VelesDB:  {machine.get('velesdb', '?')}")
    print(f"  CH:       {machine.get('clickhouse', '?')}")
    print(f"  Rounds:   {MEASURE_ROUNDS} (warmup: {WARMUP_ROUNDS})")
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

        print(f"\n  [{r['origin']}] {r['name']}: {r['description']}")
        print(f"    {'Engine':<14} {'Median':>12} {'P99':>12} {'Mean':>12}")
        print(f"    {'─' * 50}")
        print(f"    {'ClickHouse':<14} {fmt_time(ch['median']):>12} {fmt_time(ch['p99']):>12} {fmt_time(ch['mean']):>12}")
        print(f"    {'VelesDB':<14} {fmt_time(vl['median']):>12} {fmt_time(vl['p99']):>12} {fmt_time(vl['mean']):>12}")
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
    parser.add_argument("--velesdb-host", default="localhost")
    parser.add_argument("--velesdb-port", type=int, default=8080)
    parser.add_argument("--ch-host", default="localhost")
    parser.add_argument("--ch-port", type=int, default=8123)
    parser.add_argument("--parquet", default=PARQUET_PATH)
    parser.add_argument("--skip-ch-import", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    MEASURE_ROUNDS = args.rounds
    WARMUP_ROUNDS = args.warmup
    TOP_K = args.top_k

    print("VelesDB vs ClickHouse — ClickBench-Adapted Benchmark")
    print("=" * 55)

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
        print(f"  ClickHouse {machine['clickhouse']} OK")
    except Exception as e:
        print(f"  ERROR ClickHouse: {e}")
        sys.exit(1)

    # Import data
    if args.skip_ch_import:
        row_count = int(ch_client.command("SELECT count() FROM hits"))
        print(f"\n  ClickHouse: {row_count:,} rows (pre-loaded)")
    else:
        print(f"\n  Importing ClickBench data...")
        t0 = time.perf_counter()
        row_count = setup_clickhouse(ch_client, args.parquet)
        print(f"  ClickHouse: {row_count:,} rows in {time.perf_counter() - t0:.2f}s")

    rows = load_clickbench_rows(ch_client)
    print(f"\n  Loading into VelesDB ({len(PAYLOAD_COLUMNS)} columns per point)...")
    t0 = time.perf_counter()
    setup_velesdb(veles, rows)
    veles_load = time.perf_counter() - t0
    print(f"  VelesDB: {len(rows):,} points in {veles_load:.2f}s")
    del rows

    # Run
    print(f"\n  Running {MEASURE_ROUNDS} rounds per query (warmup: {WARMUP_ROUNDS})...")
    queries = define_queries(ch_client, veles)

    all_results = []
    for name, desc, ch_fn, veles_fn, origin in queries:
        print(f"    {name}...")
        ch_m = measure(ch_fn)
        veles_m = measure(veles_fn)
        all_results.append({
            "name": name, "description": desc, "origin": origin,
            "clickhouse": {k: v for k, v in ch_m.items() if k != "_result"},
            "velesdb": {k: v for k, v in veles_m.items() if k != "_result"},
            "ch_count": len(ch_m["_result"]) if ch_m["_result"] else 0,
            "veles_count": len(veles_m["_result"]) if veles_m["_result"] else 0,
        })

    if args.json:
        output = {
            "benchmark": "clickbench-adapted", "machine": machine,
            "dataset": {"rows": row_count}, "results": all_results,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_results(all_results, machine, row_count)

    veles.delete_collection("hits")
    print("\n  Done.")


if __name__ == "__main__":
    main()
