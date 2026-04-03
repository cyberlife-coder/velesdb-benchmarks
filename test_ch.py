#!/usr/bin/env python3
import clickhouse_connect
c = clickhouse_connect.get_client(host="localhost", port=8123)
print(c.command("SELECT version()"))
