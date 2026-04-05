#!/bin/bash
set -e

echo "=== Starting Qdrant ==="
docker stop bench-qdrant 2>/dev/null || true
docker rm bench-qdrant 2>/dev/null || true
docker run -d --name bench-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
echo "Qdrant starting..."

echo "=== Starting ClickHouse ==="
docker stop bench-clickhouse 2>/dev/null || true
docker rm bench-clickhouse 2>/dev/null || true
docker run -d --name bench-clickhouse -p 8123:8123 -p 9000:9000 clickhouse/clickhouse-server:latest
echo "ClickHouse starting..."

echo "=== Waiting 10s for services ==="
sleep 10

echo "=== Status ==="
docker ps --filter "name=bench-" --format "{{.Names}}: {{.Status}}"
