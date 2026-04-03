#!/bin/bash
docker stop bench-memgraph 2>/dev/null
docker rm bench-memgraph 2>/dev/null
docker run -d --name bench-memgraph -p 7687:7687 -p 7444:7444 memgraph/memgraph:latest --bolt-port=7687 --query-execution-timeout-sec=600
sleep 5
docker ps --filter name=bench-memgraph --format '{{.Status}}'
