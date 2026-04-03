#!/usr/bin/env python3
"""Quick insert throughput test — dict vs numpy path."""
import velesdb, time, os, shutil
import numpy as np

n = 100000
dim = 128
vectors = np.random.RandomState(42).randn(n, dim).astype(np.float32)
ids = np.arange(n, dtype=np.uint64)

# Method 1: upsert via dicts (slow path)
db_path = "/tmp/veles_insert_dict"
if os.path.exists(db_path): shutil.rmtree(db_path)
db = velesdb.Database(db_path)
col = db.create_collection("test", dimension=dim, metric="euclidean")
batch = 10000
t0 = time.perf_counter()
for start in range(0, n, batch):
    end = min(start + batch, n)
    points = [{"id": int(i), "vector": vectors[i].tolist()} for i in range(start, end)]
    col.upsert(points)
t1 = time.perf_counter()
print(f"Dict path:  {n/(t1-t0):>8.0f} vec/s ({t1-t0:.1f}s for {n} x {dim}D)")
shutil.rmtree(db_path, ignore_errors=True)

# Method 2: upsert_bulk_numpy (fast path)
db_path2 = "/tmp/veles_insert_numpy"
if os.path.exists(db_path2): shutil.rmtree(db_path2)
db2 = velesdb.Database(db_path2)
col2 = db2.create_collection("test", dimension=dim, metric="euclidean")
t0 = time.perf_counter()
for start in range(0, n, batch):
    end = min(start + batch, n)
    col2._inner.upsert_bulk_numpy(vectors[start:end], ids[start:end].tolist())
t1 = time.perf_counter()
print(f"Numpy path: {n/(t1-t0):>8.0f} vec/s ({t1-t0:.1f}s for {n} x {dim}D)")
shutil.rmtree(db_path2, ignore_errors=True)

print("Done")
