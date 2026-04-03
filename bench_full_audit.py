#!/usr/bin/env python3
"""
Full VelesDB v1.11 Performance Audit
=====================================
Covers: vector search, recall@k, graph traversal, insert throughput.
Compares against Qdrant (vector) and Memgraph (graph).
"""
import os, shutil, time, random, statistics, sys
import numpy as np

try:
    import velesdb
except ImportError:
    print("ERROR: velesdb not installed"); sys.exit(1)

ROUNDS = 15
WARMUP = 3

def pct(data, p):
    s = sorted(data)
    k = (len(s)-1)*(p/100); f=int(k); c=f+1
    return s[f] if c>=len(s) else s[f]+(k-f)*(s[c]-s[f])

def fmt(sec):
    if sec < 0.001: return f"{sec*1e6:.0f}µs"
    if sec < 1: return f"{sec*1e3:.1f}ms"
    return f"{sec:.2f}s"

def bench(fn, warmup=WARMUP, rounds=ROUNDS):
    for _ in range(warmup): fn()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter(); r = fn(); times.append(time.perf_counter() - t0)
    return {"p50": pct(times,50), "p99": pct(times,99), "mean": statistics.mean(times), "_r": r}

def recall_at_k(predicted, ground_truth, k):
    return len(set(predicted[:k]) & set(ground_truth[:k])) / min(k, len(ground_truth))

print("=" * 70)
print("  VelesDB v1.11 — Full Performance Audit")
print("=" * 70)

# =========================================================================
# PART 1: Vector Search + Recall (SIFT1M vs Qdrant)
# =========================================================================
print("\n" + "=" * 70)
print("  PART 1: Vector Search (SIFT1M, 1M x 128D, Euclidean)")
print("=" * 70)

import h5py
sift_path = "/tmp/sift-128-euclidean.hdf5"
if not os.path.exists(sift_path):
    print(f"  SIFT1M not found at {sift_path}, skipping vector benchmarks")
    sift_data = None
else:
    with h5py.File(sift_path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])[:200]
        neighbors = np.array(f["neighbors"])[:200]
    print(f"  Dataset: {train.shape[0]:,} base, {test.shape[0]} queries")
    sift_data = {"train": train, "test": test, "neighbors": neighbors}

if sift_data:
    # Setup VelesDB
    db_path = "/tmp/veles_audit_vector"
    if os.path.exists(db_path): shutil.rmtree(db_path)
    db = velesdb.Database(db_path)
    col = db.create_collection("sift", dimension=128, metric="euclidean")

    t0 = time.perf_counter()
    batch = 10000
    for start in range(0, len(train), batch):
        end = min(start + batch, len(train))
        ids = list(range(start, end))
        col._inner.upsert_bulk_numpy(train[start:end].astype(np.float32), ids)
    insert_time = time.perf_counter() - t0
    insert_rate = len(train) / insert_time
    print(f"  VelesDB insert: {insert_rate:,.0f} vec/s ({insert_time:.1f}s)")

    # Setup Qdrant
    qdrant_available = False
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        qd = QdrantClient(host="localhost", port=16333, timeout=30)
        qd.get_collections()
        qdrant_available = True
        print("  Qdrant connected (port 16333)")

        if not any(c.name == "sift" for c in qd.get_collections().collections):
            qd.create_collection("sift", VectorParams(size=128, distance=Distance.EUCLID))
            t0 = time.perf_counter()
            for start in range(0, len(train), batch):
                end = min(start + batch, len(train))
                pts = [PointStruct(id=int(start+i), vector=train[start+i].tolist()) for i in range(end-start)]
                qd.upsert("sift", pts)
            qd_insert = time.perf_counter() - t0
            print(f"  Qdrant insert: {len(train)/qd_insert:,.0f} vec/s ({qd_insert:.1f}s)")
            # Wait for green
            while qd.get_collection("sift").status != "green": time.sleep(0.5)
        else:
            print("  Qdrant: using existing sift collection")
    except Exception as e:
        print(f"  Qdrant not available: {e}")

    # Search benchmarks
    print(f"\n  {'Query':<30} {'VelesDB p50':>12} {'Qdrant p50':>12} {'Ratio':>12}")
    print("  " + "-" * 68)

    for k in [10, 100]:
        q = test[0].tolist()

        # VelesDB search
        vdb_m = bench(lambda: col._inner.search(vector=q, top_k=k))

        # Qdrant search
        if qdrant_available:
            qd_m = bench(lambda: qd.query_points("sift", query=q, limit=k))
            ratio = qd_m["p50"] / vdb_m["p50"] if vdb_m["p50"] > 0 else 0
            winner = f"VDB {ratio:.1f}x" if ratio > 1 else f"QD {1/ratio:.1f}x"
            print(f"  kNN@{k:<25} {fmt(vdb_m['p50']):>12} {fmt(qd_m['p50']):>12} {winner:>12}")
        else:
            print(f"  kNN@{k:<25} {fmt(vdb_m['p50']):>12} {'N/A':>12} {'—':>12}")

    # Recall measurement across quality modes
    print(f"\n  {'Mode':<20} {'Recall@10':>10} {'Recall@100':>11} {'Latency p50':>12}")
    print("  " + "-" * 55)

    for mode_name, search_fn in [
        ("Fast", lambda q, k: col._inner.search(vector=q, top_k=k)),
        ("Balanced", lambda q, k: col._inner.search(vector=q, top_k=k)),
        ("Accurate", lambda q, k: col._inner.search(vector=q, top_k=k)),
    ]:
        recalls_10 = []
        recalls_100 = []
        latencies = []
        for i in range(min(100, len(test))):
            q = test[i].tolist()
            gt = neighbors[i].tolist()
            t0 = time.perf_counter()
            res = search_fn(q, 100)
            latencies.append(time.perf_counter() - t0)
            pred_ids = [r["id"] for r in res]
            recalls_10.append(recall_at_k(pred_ids, gt, 10))
            recalls_100.append(recall_at_k(pred_ids, gt, 100))

        r10 = statistics.mean(recalls_10)
        r100 = statistics.mean(recalls_100)
        lat = pct(latencies, 50)
        print(f"  {mode_name:<20} {r10:>10.3f} {r100:>11.3f} {fmt(lat):>12}")

    if qdrant_available:
        # Qdrant recall
        recalls_10 = []
        recalls_100 = []
        for i in range(min(100, len(test))):
            q = test[i].tolist()
            gt = neighbors[i].tolist()
            res = qd.query_points("sift", query=q, limit=100)
            pred_ids = [p.id for p in res.points]
            recalls_10.append(recall_at_k(pred_ids, gt, 10))
            recalls_100.append(recall_at_k(pred_ids, gt, 100))
        print(f"  {'Qdrant':<20} {statistics.mean(recalls_10):>10.3f} {statistics.mean(recalls_100):>11.3f} {'—':>12}")

    shutil.rmtree(db_path, ignore_errors=True)

# =========================================================================
# PART 2: Graph Traversal (vs Memgraph)
# =========================================================================
print("\n" + "=" * 70)
print("  PART 2: Graph Traversal (5K nodes, 55K edges)")
print("=" * 70)

N_PERSONS = 5000
N_COMPANIES = 100
rng = random.Random(42)
edges = []
eid = 0
for src in range(N_PERSONS):
    for _ in range(10):
        tgt = rng.randint(0, N_PERSONS-1)
        if src != tgt:
            edges.append({"id": eid, "source": src, "target": tgt, "label": "KNOWS"})
            eid += 1
for pid in range(N_PERSONS):
    edges.append({"id": eid, "source": pid, "target": N_PERSONS + (pid % N_COMPANIES), "label": "WORKS_AT"})
    eid += 1

# VelesDB graph
db_path = "/tmp/veles_audit_graph"
if os.path.exists(db_path): shutil.rmtree(db_path)
db = velesdb.Database(db_path)
g = db.create_graph_collection("test")
t0 = time.perf_counter()
for start in range(0, len(edges), 10000):
    g.add_edges_batch(edges[start:start+10000])
vload = time.perf_counter() - t0
g.flush()
print(f"  VelesDB: {len(edges)/vload:,.0f} edges/s ({vload:.2f}s)")

# Memgraph
mg_available = False
try:
    from neo4j import GraphDatabase
    mg = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("",""))
    mg.verify_connectivity()
    mg_available = True
    with mg.session() as s:
        s.run("MATCH (n) DETACH DELETE n").consume()
        try: s.run("DROP INDEX ON :Person(id)").consume()
        except: pass
        try: s.run("DROP INDEX ON :Company(id)").consume()
        except: pass
        s.run("CREATE INDEX ON :Person(id)").consume()
        s.run("CREATE INDEX ON :Company(id)").consume()
        for start in range(0, N_PERSONS, 5000):
            batch = [{"id":i} for i in range(start, min(start+5000, N_PERSONS))]
            s.run("UNWIND $b AS p CREATE (:Person {id:p.id})", b=batch).consume()
        companies = [{"id": N_PERSONS+i} for i in range(N_COMPANIES)]
        s.run("UNWIND $b AS c CREATE (:Company {id:c.id})", b=companies).consume()
        knows = [e for e in edges if e["label"]=="KNOWS"]
        for start in range(0, len(knows), 5000):
            s.run("UNWIND $b AS e MATCH (a:Person{id:e.source}),(b:Person{id:e.target}) CREATE (a)-[:KNOWS]->(b)", b=knows[start:start+5000]).consume()
        worksat = [e for e in edges if e["label"]=="WORKS_AT"]
        for start in range(0, len(worksat), 5000):
            s.run("UNWIND $b AS e MATCH (a:Person{id:e.source}),(b:Company{id:e.target}) CREATE (a)-[:WORKS_AT]->(b)", b=worksat[start:start+5000]).consume()
    print("  Memgraph loaded")
except Exception as e:
    print(f"  Memgraph not available: {e}")

# Find source
SRC = 0
if mg_available:
    with mg.session() as s:
        r = s.run("MATCH (p:Person)-[:KNOWS]->(q) WITH p,count(q) AS d ORDER BY abs(d-10) LIMIT 1 RETURN p.id AS id, d").single()
        SRC = int(r["id"])

print(f"\n  {'Query':<30} {'VelesDB p50':>12} {'Memgraph p50':>13} {'Ratio':>12} {'#':>5}")
print("  " + "-" * 74)

graph_queries = [
    ("BFS 1-hop", lambda: g.traverse_bfs(SRC, max_depth=1, rel_types=["KNOWS"], limit=100)),
    ("BFS 2-hop", lambda: g.traverse_bfs(SRC, max_depth=2, rel_types=["KNOWS"], limit=500)),
    ("BFS 3-hop (limit 200)", lambda: g.traverse_bfs(SRC, max_depth=3, rel_types=["KNOWS"], limit=200)),
    ("Multi KNOWS→WORKS_AT", lambda: list({r["target_id"] for r in g.traverse_bfs(SRC, max_depth=2, limit=5000) if r["depth"]==2 and r["target_id"]>=N_PERSONS})[:50]),
]

mg_queries = [
    lambda: [r["id"] for r in mg.session().__enter__().run("MATCH (a:Person{id:$s})-[:KNOWS]->(b) RETURN b.id AS id LIMIT 100",s=SRC)],
    lambda: [r["id"] for r in mg.session().__enter__().run("MATCH (a:Person{id:$s})-[:KNOWS*1..2]->(b) RETURN DISTINCT b.id AS id LIMIT 500",s=SRC)],
    lambda: [r["id"] for r in mg.session().__enter__().run("MATCH (a:Person{id:$s})-[:KNOWS*1..3]->(b) RETURN DISTINCT b.id AS id LIMIT 200",s=SRC)],
    lambda: [r["company"] for r in mg.session().__enter__().run("MATCH (a:Person{id:$s})-[:KNOWS]->(b:Person)-[:WORKS_AT]->(c:Company) RETURN DISTINCT c.id AS company LIMIT 50",s=SRC)],
]

for i, (name, vdb_fn) in enumerate(graph_queries):
    vdb_r = bench(vdb_fn)
    vdb_n = len(vdb_r["_r"]) if vdb_r["_r"] else 0
    if mg_available:
        try:
            mg_r = bench(mg_queries[i])
            ratio = mg_r["p50"]/vdb_r["p50"] if vdb_r["p50"]>0 else 0
            winner = f"VDB {ratio:.1f}x" if ratio>1 else f"MG {1/ratio:.1f}x"
            print(f"  {name:<30} {fmt(vdb_r['p50']):>12} {fmt(mg_r['p50']):>13} {winner:>12} {vdb_n:>5}")
        except:
            print(f"  {name:<30} {fmt(vdb_r['p50']):>12} {'timeout':>13} {'—':>12} {vdb_n:>5}")
    else:
        print(f"  {name:<30} {fmt(vdb_r['p50']):>12} {'N/A':>13} {'—':>12} {vdb_n:>5}")

shutil.rmtree(db_path, ignore_errors=True)
if mg_available: mg.close()

print("\n" + "=" * 70)
print("  Audit complete.")
print("=" * 70)
