#!/usr/bin/env python3
"""
VelesDB vs Memgraph — Graph Traversal Benchmark
=================================================

Generates a synthetic social network (LDBC-style) and benchmarks:
  - BFS traversal (1-hop, 2-hop, 3-hop)
  - DFS traversal
  - Pattern matching
  - Multi-hop cross-label traversal

Fairness:
  - Both engines run in Docker containers
  - Both accessed via network (HTTP / Bolt) from the same Python process
  - Same graph, same queries, same machine
"""

import argparse
import json
import random
import statistics
import sys
import time

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j not installed (pip install neo4j)")
    sys.exit(1)

from velesdb_client import VelesDBClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_ROUNDS = 10
MEASURE_ROUNDS = 50

N_PERSONS = 50000
N_COMPANIES = 500
N_CITIES = 200
AVG_KNOWS_DEGREE = 20

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


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------


def generate_graph(seed=42):
    rng = random.Random(seed)
    print(f"  Generating graph: {N_PERSONS:,} persons, "
          f"{N_COMPANIES:,} companies, {N_CITIES:,} cities...")

    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
                   "Grace", "Henry", "Iris", "Jack", "Kate", "Leo",
                   "Mia", "Nick", "Olivia", "Paul", "Quinn", "Rose",
                   "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
                   "Yuki", "Zara"]
    persons = []
    for i in range(N_PERSONS):
        persons.append({
            "id": i, "name": f"{rng.choice(first_names)}_{i}",
            "age": rng.randint(18, 80),
            "city_id": rng.randint(0, N_CITIES - 1),
            "company_id": rng.randint(0, N_COMPANIES - 1),
        })

    companies = [{"id": N_PERSONS + i, "name": f"Company_{i}",
                  "sector": rng.choice(["Tech", "Finance", "Health", "Retail", "Energy"])}
                 for i in range(N_COMPANIES)]

    cities = [{"id": N_PERSONS + N_COMPANIES + i, "name": f"City_{i}",
               "country": rng.choice(["FR", "US", "DE", "JP", "BR"])}
              for i in range(N_CITIES)]

    edges = []
    edge_id = 0
    total_knows = N_PERSONS * AVG_KNOWS_DEGREE // 2
    for _ in range(total_knows):
        a = int(rng.paretovariate(1.5)) % N_PERSONS
        b = int(rng.paretovariate(1.5)) % N_PERSONS
        if a != b:
            edges.append({"id": edge_id, "source": a, "target": b,
                          "label": "KNOWS",
                          "properties": {"since": rng.randint(2000, 2025)}})
            edge_id += 1

    for p in persons:
        edges.append({"id": edge_id, "source": p["id"],
                      "target": N_PERSONS + p["company_id"], "label": "WORKS_AT"})
        edge_id += 1

    for p in persons:
        edges.append({"id": edge_id, "source": p["id"],
                      "target": N_PERSONS + N_COMPANIES + p["city_id"], "label": "LIVES_IN"})
        edge_id += 1

    print(f"  Generated: {len(persons):,} persons, {len(edges):,} edges total")
    return {"persons": persons, "companies": companies, "cities": cities, "edges": edges}


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------


def setup_memgraph(driver, graph_data) -> float:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
    try:
        with driver.session() as session:
            session.run("DROP INDEX ON :Person(id)").consume()
            session.run("DROP INDEX ON :Company(id)").consume()
            session.run("DROP INDEX ON :City(id)").consume()
    except Exception:
        pass

    t0 = time.perf_counter()
    batch_size = 5000
    with driver.session() as session:
        session.run("CREATE INDEX ON :Person(id)").consume()
        session.run("CREATE INDEX ON :Company(id)").consume()
        session.run("CREATE INDEX ON :City(id)").consume()

        persons = graph_data["persons"]
        for start in range(0, len(persons), batch_size):
            batch = persons[start:start + batch_size]
            session.run(
                "UNWIND $batch AS p CREATE (n:Person {id: p.id, name: p.name, age: p.age})",
                batch=batch).consume()

        session.run("UNWIND $batch AS c CREATE (n:Company {id: c.id, name: c.name})",
                     batch=graph_data["companies"]).consume()
        session.run("UNWIND $batch AS c CREATE (n:City {id: c.id, name: c.name})",
                     batch=graph_data["cities"]).consume()

        edges = graph_data["edges"]
        knows = [e for e in edges if e["label"] == "KNOWS"]
        worksat = [e for e in edges if e["label"] == "WORKS_AT"]
        livesin = [e for e in edges if e["label"] == "LIVES_IN"]

        for start in range(0, len(knows), batch_size):
            batch = knows[start:start + batch_size]
            session.run(
                "UNWIND $batch AS e "
                "MATCH (a:Person {id: e.source}), (b:Person {id: e.target}) "
                "CREATE (a)-[:KNOWS {since: e.properties.since}]->(b)",
                batch=batch).consume()

        for start in range(0, len(worksat), batch_size):
            batch = worksat[start:start + batch_size]
            session.run(
                "UNWIND $batch AS e "
                "MATCH (a:Person {id: e.source}), (b:Company {id: e.target}) "
                "CREATE (a)-[:WORKS_AT]->(b)", batch=batch).consume()

        for start in range(0, len(livesin), batch_size):
            batch = livesin[start:start + batch_size]
            session.run(
                "UNWIND $batch AS e "
                "MATCH (a:Person {id: e.source}), (b:City {id: e.target}) "
                "CREATE (a)-[:LIVES_IN]->(b)", batch=batch).consume()

    return time.perf_counter() - t0


def setup_velesdb_graph(client: VelesDBClient, graph_data) -> float:
    """Load graph into VelesDB via HTTP. Returns load_time."""
    col_name = "social"
    client.delete_collection(col_name)
    # Create a graph-capable collection (minimal dimension for graph-only use)
    client.create_collection(col_name, dimension=3, metric="cosine")

    t0 = time.perf_counter()

    for p in graph_data["persons"]:
        client.store_node_payload(col_name, p["id"], {
            "_labels": ["Person"], "name": p["name"], "age": p["age"],
        })
    for c in graph_data["companies"]:
        client.store_node_payload(col_name, c["id"], {
            "_labels": ["Company"], "name": c["name"],
        })
    for c in graph_data["cities"]:
        client.store_node_payload(col_name, c["id"], {
            "_labels": ["City"], "name": c["name"],
        })

    edges = graph_data["edges"]
    for e in edges:
        client.add_edge(col_name, e)

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmark queries
# ---------------------------------------------------------------------------


def define_graph_queries(mg_driver, veles: VelesDBClient, col_name: str):
    queries = []

    with mg_driver.session() as s:
        result = s.run(
            "MATCH (p:Person)-[:KNOWS]->(q:Person) "
            "WITH p, count(q) AS deg ORDER BY abs(deg - 20) LIMIT 1 "
            "RETURN p.id AS id, deg").single()
        source_id = int(result["id"])
        actual_deg = int(result["deg"])
    print(f"  Source node: {source_id} (KNOWS out-degree: {actual_deg})")

    # Q1: BFS 1-hop
    def mg_bfs1():
        with mg_driver.session() as s:
            return [r["id"] for r in s.run(
                "MATCH (a:Person {id: $src})-[:KNOWS]->(b:Person) RETURN b.id AS id LIMIT 100",
                src=source_id)]

    def veles_bfs1():
        res = veles.traverse_bfs(col_name, source_id, max_depth=1, rel_types=["KNOWS"], limit=100)
        return [r["target_id"] for r in res][:100]

    queries.append(("BFS_1hop", "1-hop KNOWS neighbors", mg_bfs1, veles_bfs1))

    # Q2: BFS 2-hop
    def mg_bfs2():
        with mg_driver.session() as s:
            return [r["id"] for r in s.run(
                "MATCH (a:Person {id: $src})-[:KNOWS*1..2]->(b:Person) "
                "RETURN DISTINCT b.id AS id LIMIT 1000", src=source_id)]

    def veles_bfs2():
        res = veles.traverse_bfs(col_name, source_id, max_depth=2, rel_types=["KNOWS"], limit=1000)
        return list(set(r["target_id"] for r in res))[:1000]

    queries.append(("BFS_2hop", "2-hop KNOWS (friends of friends) LIMIT 1000", mg_bfs2, veles_bfs2))

    # Q3: BFS 3-hop
    def mg_bfs3():
        with mg_driver.session() as s:
            try:
                return [r["id"] for r in s.run(
                    "MATCH (a:Person {id: $src})-[:KNOWS*1..3]->(b:Person) "
                    "RETURN DISTINCT b.id AS id LIMIT 5000", src=source_id)]
            except Exception:
                return []

    def veles_bfs3():
        res = veles.traverse_bfs(col_name, source_id, max_depth=3, rel_types=["KNOWS"], limit=5000)
        return list(set(r["target_id"] for r in res))[:5000]

    queries.append(("BFS_3hop", "3-hop KNOWS LIMIT 5000", mg_bfs3, veles_bfs3))

    # Q4: DFS 3-hop
    def mg_dfs():
        with mg_driver.session() as s:
            try:
                return [r["id"] for r in s.run(
                    "MATCH path = (a:Person {id: $src})-[:KNOWS*1..3]->(b:Person) "
                    "RETURN DISTINCT b.id AS id LIMIT 500", src=source_id)]
            except Exception:
                return []

    def veles_dfs():
        res = veles.traverse_dfs(col_name, source_id, max_depth=3, rel_types=["KNOWS"], limit=500)
        return list(set(r["target_id"] for r in res))[:500]

    queries.append(("DFS_3hop", "DFS 3-hop KNOWS LIMIT 500", mg_dfs, veles_dfs))

    # Q5: Multi-hop KNOWS → WORKS_AT
    def mg_multi():
        with mg_driver.session() as s:
            try:
                return [r["company"] for r in s.run(
                    "MATCH (a:Person {id: $src})-[:KNOWS]->(b:Person)-[:WORKS_AT]->(c:Company) "
                    "RETURN DISTINCT c.name AS company LIMIT 50", src=source_id)]
            except Exception:
                return []

    def veles_multi():
        res = veles.traverse_bfs(col_name, source_id, max_depth=2, limit=5000)
        companies = set()
        for r in res:
            if r["depth"] == 2 and r.get("target_id", 0) >= N_PERSONS:
                companies.add(r["target_id"])
        return list(companies)[:50]

    queries.append(("Multi_knows_works", "Person-[:KNOWS]->Person-[:WORKS_AT]->Company", mg_multi, veles_multi))

    return queries


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[dict], machine: dict, graph_stats: dict):
    print("\n" + "=" * 78)
    print("  VelesDB vs Memgraph — Graph Traversal Benchmark")
    print("=" * 78)
    print(f"  Graph:    {graph_stats['persons']:,} persons, {graph_stats['edges']:,} edges")
    print(f"  Runtime:  All engines in Docker, accessed via network")
    print(f"  VelesDB:  {machine.get('velesdb', '?')}")
    print(f"  Memgraph: {machine.get('memgraph', '?')}")
    print(f"  Rounds:   {MEASURE_ROUNDS} (warmup: {WARMUP_ROUNDS})")
    print("=" * 78)

    for r in results:
        mg = r["memgraph"]
        vl = r["velesdb"]
        ratio = mg["median"] / vl["median"] if vl["median"] > 0 else 0
        if ratio > 1:
            winner = f"VelesDB {ratio:.1f}x faster"
        elif ratio > 0:
            winner = f"Memgraph {1/ratio:.1f}x faster"
        else:
            winner = "N/A"

        print(f"\n  {r['name']}: {r['description']}")
        print(f"    {'Engine':<12} {'Median':>12} {'P99':>12} {'Mean':>12}")
        print(f"    {'─' * 48}")
        print(f"    {'Memgraph':<12} {fmt_time(mg['median']):>12} "
              f"{fmt_time(mg['p99']):>12} {fmt_time(mg['mean']):>12}")
        print(f"    {'VelesDB':<12} {fmt_time(vl['median']):>12} "
              f"{fmt_time(vl['p99']):>12} {fmt_time(vl['mean']):>12}")
        print(f"    → {winner}")
        print(f"    Results: Memgraph={r['mg_count']}, VelesDB={r['veles_count']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global MEASURE_ROUNDS, WARMUP_ROUNDS

    parser = argparse.ArgumentParser(description="VelesDB vs Memgraph — Graph Benchmark")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--velesdb-host", default="localhost")
    parser.add_argument("--velesdb-port", type=int, default=8080)
    parser.add_argument("--mg-host", default="127.0.0.1")
    parser.add_argument("--mg-port", type=int, default=7687)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    MEASURE_ROUNDS = args.rounds
    WARMUP_ROUNDS = args.warmup

    print("VelesDB vs Memgraph — Graph Traversal Benchmark")
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

    # Connect to Memgraph
    print("  Connecting to Memgraph...")
    try:
        mg_driver = GraphDatabase.driver(f"bolt://{args.mg_host}:{args.mg_port}", auth=("", ""))
        mg_driver.verify_connectivity()
        try:
            with mg_driver.session() as s:
                ver = s.run("CALL mg.version() YIELD version RETURN version").single()
                machine["memgraph"] = ver["version"] if ver else "v3.x"
        except Exception:
            machine["memgraph"] = "v3.x"
        print(f"  Memgraph {machine['memgraph']} OK")
    except Exception as e:
        print(f"  ERROR connecting to Memgraph: {e}")
        sys.exit(1)

    # Generate graph
    graph_data = generate_graph()
    graph_stats = {
        "persons": len(graph_data["persons"]),
        "companies": len(graph_data["companies"]),
        "cities": len(graph_data["cities"]),
        "edges": len(graph_data["edges"]),
    }

    # Load into Memgraph
    print(f"\n  Loading into Memgraph...")
    mg_load = setup_memgraph(mg_driver, graph_data)
    print(f"  Memgraph: {mg_load:.2f}s")

    # Load into VelesDB
    col_name = "social"
    print(f"\n  Loading into VelesDB...")
    velesdb_load = setup_velesdb_graph(veles, graph_data)
    print(f"  VelesDB: {velesdb_load:.2f}s")

    del graph_data

    # Run queries
    print(f"\n  Running {MEASURE_ROUNDS} rounds per query (warmup: {WARMUP_ROUNDS})...")
    queries = define_graph_queries(mg_driver, veles, col_name)

    all_results = []
    for name, desc, mg_fn, veles_fn in queries:
        print(f"    {name}...")
        mg_m = measure(mg_fn)
        veles_m = measure(veles_fn)
        all_results.append({
            "name": name, "description": desc,
            "memgraph": {k: v for k, v in mg_m.items() if k != "_result"},
            "velesdb": {k: v for k, v in veles_m.items() if k != "_result"},
            "mg_count": len(mg_m["_result"]) if mg_m["_result"] else 0,
            "veles_count": len(veles_m["_result"]) if veles_m["_result"] else 0,
        })

    if args.json:
        output = {
            "benchmark": "graph-traversal", "machine": machine, "graph": graph_stats,
            "config": {"rounds": MEASURE_ROUNDS, "warmup": WARMUP_ROUNDS},
            "load_times": {"memgraph": mg_load, "velesdb": velesdb_load},
            "results": all_results,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_results(all_results, machine, graph_stats)

    veles.delete_collection(col_name)
    mg_driver.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
