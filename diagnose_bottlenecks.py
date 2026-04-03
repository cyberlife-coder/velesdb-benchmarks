#!/usr/bin/env python3
"""Diagnose exact bottlenecks for each slow ClickBench query."""
import velesdb, time, os, shutil, math
import numpy as np

DIMENSION = 128
BATCH_SIZE = 5000

def pseudo_embedding(i, dim=DIMENSION):
    return [(math.sin(i * 0.01 + j * 0.01)) for j in range(dim)]

# Setup
db_path = "/tmp/veles_diagnose"
if os.path.exists(db_path): shutil.rmtree(db_path)
db = velesdb.Database(db_path)
col = db.create_collection("test", dimension=DIMENSION, metric="cosine")

# Create indexes
for idx_name in ["CounterID", "UserID", "IsMobile", "IsRefresh",
                 "DontCountHits", "IsLink", "IsDownload",
                 "AdvEngineID", "TraficSourceID"]:
    try:
        col.create_index(idx_name)
    except:
        pass

# Insert 100K points with realistic payloads
N = 100_000
print(f"Inserting {N} points...")
t0 = time.perf_counter()
for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    points = []
    for i in range(start, end):
        payload = {
            "UserID": 1000000 + i,
            "CounterID": 62 if i < 100 else (i % 1000),
            "IsMobile": 1 if i % 5 == 0 else 0,
            "IsRefresh": 0,
            "DontCountHits": 0,
            "IsLink": 1 if i % 20 == 0 else 0,
            "IsDownload": 0,
            "AdvEngineID": 1 if i < 500 else 0,
            "TraficSourceID": -1 if i % 10 == 0 else 6 if i % 10 == 1 else 0,
            "URL": f"https://example.com/page/{i}" + ("/google" if i % 100 == 0 else ""),
            "Title": f"Page {i}",
            "SearchPhrase": f"query {i}" if i < 500 else "",
        }
        points.append({"id": i, "vector": pseudo_embedding(i), "payload": payload})
    col.upsert(points)
print(f"Inserted in {time.perf_counter()-t0:.1f}s")

# Diagnostic queries
print("\n=== DIAGNOSTIC: Where is time spent? ===\n")

# Test 1: Simple Eq on indexed field (should be fast via index)
target_uid = 1000042
t0 = time.perf_counter()
for _ in range(10):
    res = db.execute_query(f"SELECT * FROM test WHERE UserID = {target_uid} LIMIT 10")
t1 = time.perf_counter()
print(f"Q20-like (UserID = X, indexed):     {(t1-t0)/10*1000:.1f}ms  results={len(res)}")

# Test 2: Skip vector search test (filter API format issue)
print(f"Q20-like (search+filter, vector):   skipped (filter format)")

# Test 3: AND with indexed field (CounterID=62)
t0 = time.perf_counter()
for _ in range(10):
    res = db.execute_query(
        "SELECT * FROM test WHERE CounterID = 62 AND DontCountHits = 0 AND IsRefresh = 0 LIMIT 100")
t1 = time.perf_counter()
print(f"Q37-like (CounterID=62 AND ...):    {(t1-t0)/10*1000:.1f}ms  results={len(res)}")

# Test 4: IsMobile = 1 (high selectivity ~20%)
t0 = time.perf_counter()
for _ in range(10):
    res = db.execute_query("SELECT * FROM test WHERE IsMobile = 1 LIMIT 100")
t1 = time.perf_counter()
print(f"Qx-like (IsMobile=1, indexed):      {(t1-t0)/10*1000:.1f}ms  results={len(res)}")

# Test 5: URL LIKE (no index, full scan)
t0 = time.perf_counter()
for _ in range(10):
    res = db.execute_query("SELECT * FROM test WHERE URL LIKE '%google%' LIMIT 100")
t1 = time.perf_counter()
print(f"Q21-like (URL LIKE, no index):      {(t1-t0)/10*1000:.1f}ms  results={len(res)}")

# Test 6: AdvEngineID != 0 (not indexable)
t0 = time.perf_counter()
for _ in range(10):
    res = db.execute_query("SELECT * FROM test WHERE AdvEngineID != 0 AND SearchPhrase != '' LIMIT 100")
t1 = time.perf_counter()
print(f"Qx-like (AdvEngine!=0, no index):   {(t1-t0)/10*1000:.1f}ms  results={len(res)}")

# Test 7: Raw get by ID (baseline for point lookup)
t0 = time.perf_counter()
for _ in range(1000):
    res = col.get([42])
t1 = time.perf_counter()
print(f"Raw get(id=42):                     {(t1-t0)/1000*1000:.3f}ms")

# Test 8: Component timing
print("\n=== COMPONENT TIMING ===")

t0 = time.perf_counter()
for _ in range(100):
    res = db.execute_query("SELECT * FROM test WHERE CounterID = 62 LIMIT 100")
t1 = time.perf_counter()
print(f"CounterID=62 (indexed, ~100 rows):  {(t1-t0)/100*1000:.1f}ms  results={len(res)}")

t0 = time.perf_counter()
for _ in range(100):
    res = db.execute_query("SELECT * FROM test WHERE CounterID = 999 LIMIT 100")
t1 = time.perf_counter()
print(f"CounterID=999 (indexed, ~100 rows): {(t1-t0)/100*1000:.1f}ms  results={len(res)}")

t0 = time.perf_counter()
for _ in range(10):
    res = db.execute_query("SELECT * FROM test LIMIT 100")
t1 = time.perf_counter()
print(f"SELECT * LIMIT 100 (no filter):     {(t1-t0)/10*1000:.1f}ms  results={len(res)}")

shutil.rmtree(db_path, ignore_errors=True)
print("\nDone.")
