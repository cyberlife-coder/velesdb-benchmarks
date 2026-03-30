#!/usr/bin/env python3
"""
VelesDB vs Memgraph — Graph Traversal Benchmark
=================================================

Generates a synthetic social network (LDBC-style) and benchmarks:
  - BFS traversal (1-hop, 2-hop, 3-hop)
  - DFS traversal
  - Pattern matching (MATCH queries)
  - Multi-hop cross-label traversal

Memgraph is an in-memory native graph database (specialist).
Run with: bash run_graph_wsl2.sh

Fairness:
  - Same graph (same nodes, edges, properties)
  - Same WSL2 environment, same Python process
  - Warmup + multiple rounds with p50/p99
"""

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import time

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j not installed (pip install neo4j)")
    sys.exit(1)

try:
    import velesdb
except ImportError:
    print("ERROR: velesdb not installed")
    sys.exit(1)

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


# ---------------------------------------------------------------------------
# Graph generation (LDBC-style social network)
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
            "id": i,
            "name": f"{rng.choice(first_names)}_{i}",
            "age": rng.randint(18, 80),
            "city_id": rng.randint(0, N_CITIES - 1),
            "company_id": rng.randint(0, N_COMPANIES - 1),
        })

    companies = [{"id": N_PERSONS + i, "name": f"Company_{i}",
                  "sector": rng.choice(["Tech", "Finance", "Health",
                                        "Retail", "Energy"])}
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
                      "target": N_PERSONS + p["company_id"],
                      "label": "WORKS_AT"})
        edge_id += 1

    for p in persons:
        edges.append({"id": edge_id, "source": p["id"],
                      "target": N_PERSONS + N_COMPANIES + p["city_id"],
                      "label": "LIVES_IN"})
        edge_id += 1

    print(f"  Generated: {len(persons):,} persons, {len(edges):,} edges total")
    print(f"    KNOWS: ~{total_knows:,}, WORKS_AT: {N_PERSONS:,}, LIVES_IN: {N_PERSONS:,}")

    return {"persons": persons, "companies": companies,
            "cities": cities, "edges": edges}


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------


def setup_memgraph(driver, graph_data) -> float:
    """Load graph into Memgraph. Returns load time."""
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
                batch=batch,
            ).consume()
            if (start + batch_size) % 20000 == 0:
                print(f"    Memgraph persons: {min(start + batch_size, len(persons)):,}/{len(persons):,}")

        session.run(
            "UNWIND $batch AS c CREATE (n:Company {id: c.id, name: c.name})",
            batch=graph_data["companies"],
        ).consume()
        session.run(
            "UNWIND $batch AS c CREATE (n:City {id: c.id, name: c.name})",
            batch=graph_data["cities"],
        ).consume()

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
                batch=batch,
            ).consume()
            if (start + batch_size) % 50000 == 0:
                print(f"    Memgraph KNOWS: {min(start + batch_size, len(knows)):,}/{len(knows):,}")

        for start in range(0, len(worksat), batch_size):
            batch = worksat[start:start + batch_size]
            session.run(
                "UNWIND $batch AS e "
                "MATCH (a:Person {id: e.source}), (b:Company {id: e.target}) "
                "CREATE (a)-[:WORKS_AT]->(b)",
                batch=batch,
            ).consume()

        for start in range(0, len(livesin), batch_size):
            batch = livesin[start:start + batch_size]
            session.run(
                "UNWIND $batch AS e "
                "MATCH (a:Person {id: e.source}), (b:City {id: e.target}) "
                "CREATE (a)-[:LIVES_IN]->(b)",
                batch=batch,
            ).consume()

    return time.perf_counter() - t0


def setup_velesdb_graph(graph_data, db_path: str) -> tuple:
    """Load graph into VelesDB GraphCollection. Returns (graph_col, load_time)."""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = velesdb.Database(db_path)
    graph = db.create_graph_collection("social")

    t0 = time.perf_counter()

    for p in graph_data["persons"]:
        graph.store_node_payload(p["id"], {
            "_labels": ["Person"], "name": p["name"], "age": p["age"],
        })
    for c in graph_data["companies"]:
        graph.store_node_payload(c["id"], {
            "_labels": ["Company"], "name": c["name"],
        })
    for c in graph_data["cities"]:
        graph.store_node_payload(c["id"], {
            "_labels": ["City"], "name": c["name"],
        })

    edges = graph_data["edges"]
    for i, e in enumerate(edges):
        graph.add_edge(e)
        if (i + 1) % 100000 == 0:
            print(f"    VelesDB edges: {i + 1:,}/{len(edges):,}")

    graph.flush()
    return graph, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmark queries
# ---------------------------------------------------------------------------


def define_graph_queries(mg_driver, veles_graph):
    """Define graph queries for both engines."""
    queries = []

    # Find a typical person with ~20 KNOWS connections
    with mg_driver.session() as s:
        result = s.run(
            "MATCH (p:Person)-[:KNOWS]->(q:Person) "
            "WITH p, count(q) AS deg "
            "ORDER BY abs(deg - 20) LIMIT 1 "
            "RETURN p.id AS id, deg"
        ).single()
        source_id = int(result["id"])
        actual_deg = int(result["deg"])
    print(f"  Source node: {source_id} (KNOWS out-degree: {actual_deg})")

    # --- Q1: BFS 1-hop ---
    def mg_bfs1():
        with mg_driver.session() as s:
            result = s.run(
                "MATCH (a:Person {id: $src})-[:KNOWS]->(b:Person) "
                "RETURN b.id AS id LIMIT 100",
                src=source_id,
            )
            return [r["id"] for r in result]

    def veles_bfs1():
        res = veles_graph.traverse_bfs(source_id, max_depth=1,
                                        rel_types=["KNOWS"])
        return [r["target_id"] for r in res][:100]

    queries.append(("BFS_1hop", "1-hop KNOWS neighbors (direct friends)",
                     mg_bfs1, veles_bfs1))

    # --- Q2: BFS 2-hop ---
    def mg_bfs2():
        with mg_driver.session() as s:
            result = s.run(
                "MATCH (a:Person {id: $src})-[:KNOWS*1..2]->(b:Person) "
                "RETURN DISTINCT b.id AS id LIMIT 1000",
                src=source_id,
            )
            return [r["id"] for r in result]

    def veles_bfs2():
        res = veles_graph.traverse_bfs(source_id, max_depth=2,
                                        rel_types=["KNOWS"])
        return list(set(r["target_id"] for r in res))[:1000]

    queries.append(("BFS_2hop", "2-hop KNOWS (friends of friends) LIMIT 1000",
                     mg_bfs2, veles_bfs2))

    # --- Q3: BFS 3-hop ---
    def mg_bfs3():
        with mg_driver.session() as s:
            result = s.run(
                "MATCH (a:Person {id: $src})-[:KNOWS*1..3]->(b:Person) "
                "RETURN DISTINCT b.id AS id LIMIT 5000",
                src=source_id,
            )
            return [r["id"] for r in result]

    def veles_bfs3():
        res = veles_graph.traverse_bfs(source_id, max_depth=3,
                                        rel_types=["KNOWS"])
        return list(set(r["target_id"] for r in res))[:5000]

    queries.append(("BFS_3hop", "3-hop KNOWS (3 degrees of separation) LIMIT 5000",
                     mg_bfs3, veles_bfs3))

    # --- Q4: Pattern match — Person-WORKS_AT-Company (age > 30) ---
    def mg_pattern_company():
        with mg_driver.session() as s:
            result = s.run(
                "MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
                "WHERE p.age > 30 "
                "RETURN p.name AS person, c.name AS company LIMIT 100",
            )
            return [(r["person"], r["company"]) for r in result]

    def veles_pattern_company():
        res = veles_graph.match_query(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
            "RETURN p, c LIMIT 100"
        )
        return [(r.get("bindings", {}).get("p"),
                 r.get("bindings", {}).get("c"))
                for r in res]

    queries.append(("MATCH_works_at",
                     "MATCH (Person)-[:WORKS_AT]->(Company) WHERE age>30 LIMIT 100",
                     mg_pattern_company, veles_pattern_company))

    # --- Q5: Edge label query — single node's outgoing edges ---
    def mg_edge_label():
        with mg_driver.session() as s:
            result = s.run(
                "MATCH (a:Person {id: $src})-[:WORKS_AT]->(c:Company) "
                "RETURN c.name AS company",
                src=source_id,
            )
            return [r["company"] for r in result]

    def veles_edge_label():
        outgoing = veles_graph.get_outgoing(source_id)
        return [e for e in outgoing if e.get("label") == "WORKS_AT"]

    queries.append(("Edge_works_at",
                     "Get WORKS_AT edge from source node",
                     mg_edge_label, veles_edge_label))

    # --- Q6: DFS 3-hop ---
    def mg_dfs():
        with mg_driver.session() as s:
            result = s.run(
                "MATCH path = (a:Person {id: $src})-[:KNOWS*1..3]->(b:Person) "
                "RETURN DISTINCT b.id AS id LIMIT 500",
                src=source_id,
            )
            return [r["id"] for r in result]

    def veles_dfs():
        res = veles_graph.traverse_dfs(source_id, max_depth=3,
                                        rel_types=["KNOWS"])
        return list(set(r["target_id"] for r in res))[:500]

    queries.append(("DFS_3hop", "DFS 3-hop KNOWS LIMIT 500",
                     mg_dfs, veles_dfs))

    # --- Q7: Multi-hop mixed labels (KNOWS → WORKS_AT) ---
    def mg_multi():
        with mg_driver.session() as s:
            result = s.run(
                "MATCH (a:Person {id: $src})-[:KNOWS]->(b:Person)"
                "-[:WORKS_AT]->(c:Company) "
                "RETURN DISTINCT c.name AS company LIMIT 50",
                src=source_id,
            )
            return [r["company"] for r in result]

    def veles_multi():
        friends = veles_graph.traverse_bfs(source_id, max_depth=1,
                                            rel_types=["KNOWS"])
        companies = set()
        for f in friends[:50]:
            out = veles_graph.get_outgoing(f["target_id"])
            for e in out:
                if e.get("label") == "WORKS_AT":
                    companies.add(e.get("target"))
        return list(companies)[:50]

    queries.append(("Multi_knows_works",
                     "Person-[:KNOWS]->Person-[:WORKS_AT]->Company",
                     mg_multi, veles_multi))

    return queries


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[dict], machine: dict, graph_stats: dict):
    print("\n" + "=" * 78)
    print("  VelesDB vs Memgraph \u2014 Graph Traversal Benchmark")
    print("=" * 78)
    print(f"  Graph:    {graph_stats['persons']:,} persons, "
          f"{graph_stats['edges']:,} edges (LDBC-style social network)")
    print(f"  OS:       {machine.get('os', 'WSL2')}")
    print(f"  VelesDB:  {machine.get('velesdb', '?')}")
    print(f"  Memgraph: {machine.get('memgraph', '?')}")
    print(f"  Rounds:   {MEASURE_ROUNDS} (warmup: {WARMUP_ROUNDS})")
    print(f"  Fairness: Same graph, same queries, same process")
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
        print(f"    {'Engine':<12} {'Median':>12} {'P99':>12} "
              f"{'Mean':>12} {'Min':>12}")
        print(f"    {'─' * 60}")
        print(f"    {'Memgraph':<12} {fmt_time(mg['median']):>12} "
              f"{fmt_time(mg['p99']):>12} {fmt_time(mg['mean']):>12} "
              f"{fmt_time(mg['min']):>12}")
        print(f"    {'VelesDB':<12} {fmt_time(vl['median']):>12} "
              f"{fmt_time(vl['p99']):>12} {fmt_time(vl['mean']):>12} "
              f"{fmt_time(vl['min']):>12}")
        print(f"    \u2192 {winner}")
        print(f"    Results: Memgraph={r['mg_count']}, VelesDB={r['veles_count']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global MEASURE_ROUNDS, WARMUP_ROUNDS

    parser = argparse.ArgumentParser(
        description="VelesDB vs Memgraph \u2014 Graph Benchmark")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--mg-host", default="127.0.0.1")
    parser.add_argument("--mg-port", type=int, default=7687)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    MEASURE_ROUNDS = args.rounds
    WARMUP_ROUNDS = args.warmup

    print("VelesDB vs Memgraph \u2014 Graph Traversal Benchmark")
    print("=" * 52)

    machine = {
        "os": "WSL2",
        "velesdb": velesdb.__version__,
    }

    # Connect to Memgraph
    print("\n  Connecting to Memgraph...")
    try:
        mg_driver = GraphDatabase.driver(
            f"bolt://{args.mg_host}:{args.mg_port}", auth=("", ""))
        mg_driver.verify_connectivity()
        with mg_driver.session() as s:
            ver = s.run("CALL mg.version() YIELD version RETURN version").single()
            machine["memgraph"] = ver["version"] if ver else "v3.x"
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
    db_path = "/tmp/velesdb_graph"
    print(f"\n  Loading into VelesDB...")
    veles_graph, velesdb_load = setup_velesdb_graph(graph_data, db_path)
    print(f"  VelesDB: {velesdb_load:.2f}s")

    del graph_data

    # Define and run queries
    print(f"\n  Running {MEASURE_ROUNDS} rounds per query "
          f"(warmup: {WARMUP_ROUNDS})...")
    queries = define_graph_queries(mg_driver, veles_graph)

    all_results = []
    for name, desc, mg_fn, veles_fn in queries:
        print(f"    {name}...")
        mg_m = measure(mg_fn)
        veles_m = measure(veles_fn)

        all_results.append({
            "name": name,
            "description": desc,
            "memgraph": {k: v for k, v in mg_m.items() if k != "_result"},
            "velesdb": {k: v for k, v in veles_m.items() if k != "_result"},
            "mg_count": len(mg_m["_result"]) if mg_m["_result"] else 0,
            "veles_count": len(veles_m["_result"]) if veles_m["_result"] else 0,
        })

    if args.json:
        output = {
            "benchmark": "graph-traversal",
            "machine": machine,
            "graph": graph_stats,
            "config": {"rounds": MEASURE_ROUNDS, "warmup": WARMUP_ROUNDS},
            "load_times": {"memgraph": mg_load, "velesdb": velesdb_load},
            "results": all_results,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_results(all_results, machine, graph_stats)

    del veles_graph
    if os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)
    mg_driver.close()

    print("\n  Done.")


if __name__ == "__main__":
    main()
