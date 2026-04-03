#!/usr/bin/env python3
"""Quick VelesDB vs Memgraph benchmark — reduced graph for fast iteration."""
import os, shutil, time, random, statistics
from neo4j import GraphDatabase
import velesdb

N_PERSONS = 5000
N_COMPANIES = 100
AVG_DEGREE = 10
ROUNDS = 10
WARMUP = 3

def pct(data, p):
    s = sorted(data)
    k = (len(s)-1)*(p/100); f=int(k); c=f+1
    return s[f] if c>=len(s) else s[f]+(k-f)*(s[c]-s[f])

def fmt(sec):
    return f"{sec*1e6:.0f}us" if sec<0.001 else f"{sec*1e3:.1f}ms"

def bench(fn, warmup=WARMUP, rounds=ROUNDS):
    for _ in range(warmup): fn()
    times = []
    for _ in range(rounds):
        t0=time.perf_counter(); r=fn(); times.append(time.perf_counter()-t0)
    return {"p50": pct(times,50), "p99": pct(times,99), "mean": statistics.mean(times), "_r": r}

# Generate graph
rng = random.Random(42)
edges = []
eid = 0
for src in range(N_PERSONS):
    for _ in range(AVG_DEGREE):
        tgt = rng.randint(0, N_PERSONS-1)
        if src != tgt:
            edges.append({"id": eid, "source": src, "target": tgt, "label": "KNOWS"})
            eid += 1
# Add WORKS_AT edges
for pid in range(N_PERSONS):
    cid = N_PERSONS + (pid % N_COMPANIES)
    edges.append({"id": eid, "source": pid, "target": cid, "label": "WORKS_AT"})
    eid += 1
print(f"Graph: {N_PERSONS} nodes, {len(edges)} edges ({len(edges)-N_PERSONS} KNOWS + {N_PERSONS} WORKS_AT)")

# Memgraph
mg = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("",""))
mg.verify_connectivity()
with mg.session() as s:
    s.run("MATCH (n) DETACH DELETE n").consume()
    try: s.run("DROP INDEX ON :Person(id)").consume()
    except: pass
    try: s.run("DROP INDEX ON :Company(id)").consume()
    except: pass
    s.run("CREATE INDEX ON :Person(id)").consume()
    s.run("CREATE INDEX ON :Company(id)").consume()
    for start in range(0, N_PERSONS, 5000):
        batch = [{"id":i,"name":f"P{i}"} for i in range(start, min(start+5000,N_PERSONS))]
        s.run("UNWIND $b AS p CREATE (:Person {id:p.id,name:p.name})", b=batch).consume()
    companies = [{"id": N_PERSONS+i, "name": f"C{i}"} for i in range(N_COMPANIES)]
    s.run("UNWIND $b AS c CREATE (:Company {id:c.id,name:c.name})", b=companies).consume()
    knows = [e for e in edges if e["label"]=="KNOWS"]
    worksat = [e for e in edges if e["label"]=="WORKS_AT"]
    for start in range(0, len(knows), 5000):
        batch = knows[start:start+5000]
        s.run("UNWIND $b AS e MATCH (a:Person{id:e.source}),(b:Person{id:e.target}) CREATE (a)-[:KNOWS]->(b)", b=batch).consume()
    for start in range(0, len(worksat), 5000):
        batch = worksat[start:start+5000]
        s.run("UNWIND $b AS e MATCH (a:Person{id:e.source}),(b:Company{id:e.target}) CREATE (a)-[:WORKS_AT]->(b)", b=batch).consume()
print("Memgraph loaded")

# Find source
with mg.session() as s:
    r = s.run("MATCH (p:Person)-[:KNOWS]->(q) WITH p,count(q) AS d ORDER BY abs(d-10) LIMIT 1 RETURN p.id AS id, d").single()
    SRC = int(r["id"]); print(f"Source: {SRC} (degree {r['d']})")

# VelesDB
db_path = "/tmp/veles_quick_bench"
if os.path.exists(db_path): shutil.rmtree(db_path)
db = velesdb.Database(db_path)
g = db.create_graph_collection("test")
t0 = time.perf_counter()
batch_size = 10000
for start in range(0, len(edges), batch_size):
    g.add_edges_batch(edges[start:start+batch_size])
vload = time.perf_counter()-t0
g.flush()
print(f"VelesDB loaded in {vload:.2f}s ({len(edges)/vload:.0f} edges/s)")

# Benchmarks
print(f"\n{'Query':<30} {'Memgraph p50':>14} {'VelesDB p50':>14} {'Ratio':>12} {'MG#':>6} {'VDB#':>6}")
print("-"*86)

queries = [
    ("BFS 1-hop",
     lambda: (lambda s: [r["id"] for r in s.run("MATCH (a:Person{id:$s})-[:KNOWS]->(b) RETURN b.id AS id LIMIT 100",s=SRC)])(mg.session().__enter__()),
     lambda: [r["target_id"] for r in g.traverse_bfs(SRC, max_depth=1, rel_types=["KNOWS"], limit=100)]),
    ("BFS 2-hop",
     lambda: (lambda s: [r["id"] for r in s.run("MATCH (a:Person{id:$s})-[:KNOWS*1..2]->(b) RETURN DISTINCT b.id AS id LIMIT 500",s=SRC)])(mg.session().__enter__()),
     lambda: [r["target_id"] for r in g.traverse_bfs(SRC, max_depth=2, rel_types=["KNOWS"], limit=500)]),
    ("BFS 3-hop (limit 200)",
     lambda: (lambda s: [r["id"] for r in s.run("MATCH (a:Person{id:$s})-[:KNOWS*1..3]->(b) RETURN DISTINCT b.id AS id LIMIT 200",s=SRC)])(mg.session().__enter__()),
     lambda: [r["target_id"] for r in g.traverse_bfs(SRC, max_depth=3, rel_types=["KNOWS"], limit=200)]),
    ("Multi KNOWS→WORKS_AT",
     lambda: (lambda s: [r["company"] for r in s.run("MATCH (a:Person{id:$s})-[:KNOWS]->(b:Person)-[:WORKS_AT]->(c:Company) RETURN DISTINCT c.name AS company LIMIT 50",s=SRC)])(mg.session().__enter__()),
     lambda: list({r["target_id"] for r in g.traverse_bfs(SRC, max_depth=2, limit=5000) if r["depth"]==2 and r["target_id"]>=N_PERSONS})[:50]),
]

for name, mg_fn, vdb_fn in queries:
    try:
        mg_r = bench(mg_fn)
    except Exception as e:
        mg_r = {"p50": 999, "p99": 999, "mean": 999, "_r": []}
        print(f"  [Memgraph timeout on {name}]")
    vdb_r = bench(vdb_fn)
    ratio = mg_r["p50"]/vdb_r["p50"] if vdb_r["p50"]>0 else 0
    winner = f"VDB {ratio:.1f}x" if ratio>1 else f"MG {1/ratio:.1f}x" if ratio>0 else "N/A"
    mg_n = len(mg_r["_r"]) if mg_r["_r"] else 0
    vdb_n = len(vdb_r["_r"]) if vdb_r["_r"] else 0
    print(f"{name:<30} {fmt(mg_r['p50']):>14} {fmt(vdb_r['p50']):>14} {winner:>12} {mg_n:>6} {vdb_n:>6}")

shutil.rmtree(db_path, ignore_errors=True)
mg.close()
print("\nDone.")
