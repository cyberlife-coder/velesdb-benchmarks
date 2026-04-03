#!/bin/bash
set -e
echo "Downloading ClickBench 1M rows via ClickHouse Docker..."

# Method 1: Direct download of the full parquet (14GB) is too large.
# Method 2: Use clickhouse-client inside Docker to query and export.
docker exec bench-clickhouse clickhouse-client --query \
  "SELECT * FROM url('https://datasets.clickhouse.com/hits_compatible/hits.parquet', Parquet) LIMIT 1000000 INTO OUTFILE '/tmp/hits_1m.parquet' FORMAT Parquet"

echo "Copying parquet from container..."
docker cp bench-clickhouse:/tmp/hits_1m.parquet /tmp/hits_1m.parquet

ls -lh /tmp/hits_1m.parquet
echo "Done."
