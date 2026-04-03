#!/usr/bin/env python3
"""Diagnose Q20 (point lookup) and Q21 (LIKE) on 1M rows."""
import velesdb, time, os, shutil, math, json
import numpy as np

DIMENSION = 128
N = 200_000  # Smaller than 1M but enough to show scaling

def pseudo_embedding(i, dim=DIMENSION):
    return [(math.sin(i * 0.01 + j * 0.01)) for j in range(dim)]

db_path = "/tmp/veles_diag_q20"
if os.path.exists(db_path): shutil.rmtree(db_path)
db = velesdb.Database(db_path)
col = db.create_collection("test", dimension=DIMENSION, metric="cosine")

# Create indexes BEFORE insert
for idx in ["UserID", "CounterID", "IsMobile"]:
    col.create_index(idx)
print("Indexes created")

# Insert with numpy+json path
print(f"Inserting {N} points...")
batch = 5000
t0 = time.perf_counter()
for start in range(0, N, batch):
    end = min(start + batch, N)
    n_batch = end - start
    vectors = np.array([pseudo_embedding(start+i) for i in range(n_batch)], dtype=np.float32)
    ids = list(range(start, end))
    payloads = [json.dumps({
        "UserID": 1000000 + i,
        "CounterID": 62 if i < 200 else (i % 1000),
        "IsMobile": 1 if i % 5 == 0 else 0,
        "URL": f"https://example.com/page/{i}" + ("/google/search" if i % 100 == 0 else ""),
        "Title": f"Page {i} about google technology" if i % 100 == 0 else f"Page {i}",
    }) for i in range(start, end)]
    col._inner.upsert_bulk_numpy_json(vectors, ids, payloads)
insert_time = time.perf_counter() - t0
print(f"Inserted {N} in {insert_time:.1f}s ({N/insert_time:.0f} vec/s)")

# Test Q20: UserID point lookup
target_uid = 1000042
print(f"\n=== Q20: UserID = {target_uid} ===")
t0 = time.perf_counter()
for _ in range(10):
    res = db.execute_query(f"SELECT * FROM test WHERE UserID = {target_uid} LIMIT 10")
t1 = time.perf_counter()
print(f"  VelesQL path: {(t1-t0)/10*1000:.1f}ms, results={len(res)}")

# Test Q21: URL LIKE '%google%'
print(f"\n=== Q21: URL LIKE '%google%' ===")
t0 = time.perf_counter()
for _ in range(10):
    res = db.execute_query("SELECT * FROM test WHERE URL LIKE '%google%' LIMIT 100")
t1 = time.perf_counter()
print(f"  VelesQL path: {(t1-t0)/10*1000:.1f}ms, results={len(res)}")

# Check BM25 directly
print(f"\n=== BM25 direct search for 'google' ===")
bm25_results = col._inner.text_search("google", 100)
print(f"  BM25 results: {len(bm25_results)}")
if bm25_results:
    print(f"  First 3 IDs: {[r['id'] for r in bm25_results[:3]]}")

# Check if index is populated
print(f"\n=== Index check ===")
info = col._inner.info()
print(f"  Collection info: {json.dumps(info, indent=2, default=str)}")

shutil.rmtree(db_path, ignore_errors=True)
print("\nDone.")
