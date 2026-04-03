#!/usr/bin/env python3
from neo4j import GraphDatabase
d = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("", ""))
d.verify_connectivity()
print("Memgraph OK")
d.close()
