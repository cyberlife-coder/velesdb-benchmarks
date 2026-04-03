#!/usr/bin/env python3
"""Diagnose Q20 with tracing enabled."""
import velesdb, time, os, shutil, math, json, logging
import numpy as np

# Enable tracing
logging.basicConfig(level=logging.DEBUG)
os.environ["RUST_LOG"] = "velesdb_core=debug"

DIMENSION = 128
N = 10_000  # Small for fast diagnosis

def pseudo_embedding(i, dim=DIMENSION):
    return [(math.sin(i * 0.01 + j * 0.01)) for j in range(dim)]

db_path = "/tmp/veles_diag_trace"
if os.path.exists(db_path): shutil.rmtree(db_path)
db = velesdb.Database(db_path)
col = db.create_collection("test", dimension=DIMENSION, metric="cosine")

# Create index BEFORE insert
col.create_index("UserID")
print("Index created on UserID")

# Insert
batch = 5000
for start in range(0, N, batch):
    end = min(start + batch, N)
    n_batch = end - start
    vectors = np.array([pseudo_embedding(start+i) for i in range(n_batch)], dtype=np.float32)
    ids = list(range(start, end))
    payloads = [json.dumps({"UserID": 1000000 + i}) for i in range(start, end)]
    col._inner.upsert_bulk_numpy_json(vectors, ids, payloads)
print(f"Inserted {N}")

# Test: does the index have entries?
# Try a direct search via the collection
target_uid = 1000042
print(f"\n=== Testing UserID = {target_uid} ===")

# Method 1: VelesQL
t0 = time.perf_counter()
res = db.execute_query(f"SELECT * FROM test WHERE UserID = {target_uid} LIMIT 10")
t1 = time.perf_counter()
print(f"VelesQL: {(t1-t0)*1000:.1f}ms, results={len(res)}")

# Method 2: search with filter (vector path)
# This uses the bitmap prefilter which should use the index
t0 = time.perf_counter()
res2 = col._inner.search(vector=[0.5]*DIMENSION, top_k=10, filter={"condition": {"type": "eq", "field": "UserID", "value": 1000042}})
t1 = time.perf_counter()
print(f"search+filter: {(t1-t0)*1000:.1f}ms, results={len(res2)}")

shutil.rmtree(db_path, ignore_errors=True)
