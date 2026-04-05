"""
VelesDB HTTP Client for benchmarks.
Wraps the REST API so benchmarks use the same network path as competitors.
"""

import json
from typing import Any

import requests


class VelesDBClient:
    """Thin HTTP client for VelesDB REST API."""

    def __init__(self, host: str = "localhost", port: int = 8080, timeout: int = 120):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        r = self.session.get(f"{self.base}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def ready(self) -> bool:
        try:
            r = self.session.get(f"{self.base}/ready", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def wait_ready(self, retries: int = 30, interval: float = 1.0):
        import time
        for _ in range(retries):
            if self.ready():
                return True
            time.sleep(interval)
        raise RuntimeError("VelesDB server not ready")

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def create_collection(self, name: str, dimension: int, metric: str = "cosine") -> dict:
        r = self.session.post(
            f"{self.base}/collections",
            json={"name": name, "dimension": dimension, "metric": metric},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def delete_collection(self, name: str):
        r = self.session.delete(f"{self.base}/collections/{name}", timeout=self.timeout)
        # 404 is fine (already deleted)
        if r.status_code not in (200, 204, 404):
            r.raise_for_status()

    def get_collection(self, name: str) -> dict:
        r = self.session.get(f"{self.base}/collections/{name}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_collections(self) -> list:
        r = self.session.get(f"{self.base}/collections", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Points
    # ------------------------------------------------------------------

    def upsert_points(self, collection: str, points: list[dict]):
        """Upsert a batch of points. Each point: {id, vector, payload?}."""
        r = self.session.post(
            f"{self.base}/collections/{collection}/points",
            json={"points": points},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 10,
        filter: dict | None = None,
        mode: str | None = None,
    ) -> list[dict]:
        body: dict[str, Any] = {"vector": vector, "top_k": top_k}
        if filter:
            body["filter"] = filter
        if mode:
            body["mode"] = mode
        r = self.session.post(
            f"{self.base}/collections/{collection}/search",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("results", data)

    # ------------------------------------------------------------------
    # VelesQL
    # ------------------------------------------------------------------

    def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        body: dict[str, Any] = {"query": query, "params": params or {}}
        r = self.session.post(
            f"{self.base}/query",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("results", data)

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def create_graph_collection(self, name: str, dimension: int = 3) -> dict:
        """Create a collection that will hold graph data.
        We create a minimal vector collection, then use graph endpoints."""
        return self.create_collection(name, dimension=dimension, metric="cosine")

    def add_edge(self, collection: str, edge: dict):
        r = self.session.post(
            f"{self.base}/collections/{collection}/graph/edges",
            json=edge,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def add_edges_batch(self, collection: str, edges: list[dict]):
        """Add edges one by one (server has no batch endpoint yet)."""
        for e in edges:
            self.add_edge(collection, e)

    def store_node_payload(self, collection: str, node_id: int, payload: dict):
        r = self.session.put(
            f"{self.base}/collections/{collection}/graph/nodes/{node_id}/payload",
            json={"payload": payload},
            timeout=self.timeout,
        )
        r.raise_for_status()

    def get_node_payload(self, collection: str, node_id: int) -> dict:
        r = self.session.get(
            f"{self.base}/collections/{collection}/graph/nodes/{node_id}/payload",
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def traverse_bfs(
        self,
        collection: str,
        source: int,
        max_depth: int = 3,
        limit: int = 100,
        rel_types: list[str] | None = None,
    ) -> list[dict]:
        body: dict[str, Any] = {
            "source": source,
            "strategy": "bfs",
            "max_depth": max_depth,
            "limit": limit,
        }
        if rel_types:
            body["rel_types"] = rel_types
        r = self.session.post(
            f"{self.base}/collections/{collection}/graph/traverse",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def traverse_dfs(
        self,
        collection: str,
        source: int,
        max_depth: int = 3,
        limit: int = 100,
        rel_types: list[str] | None = None,
    ) -> list[dict]:
        body: dict[str, Any] = {
            "source": source,
            "strategy": "dfs",
            "max_depth": max_depth,
            "limit": limit,
        }
        if rel_types:
            body["rel_types"] = rel_types
        r = self.session.post(
            f"{self.base}/collections/{collection}/graph/traverse",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def get_outgoing(self, collection: str, node_id: int) -> list[dict]:
        r = self.session.get(
            f"{self.base}/collections/{collection}/graph/nodes/{node_id}/edges",
            params={"direction": "out"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def match_query(self, collection: str, query: str, vector: list | None = None) -> list[dict]:
        body: dict[str, Any] = {"query": query}
        if vector:
            body["vector"] = vector
        r = self.session.post(
            f"{self.base}/collections/{collection}/match",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def create_index(self, collection: str, label: str):
        r = self.session.post(
            f"{self.base}/collections/{collection}/indexes",
            json={"label": label, "property": "name"},
            timeout=self.timeout,
        )
        # Ignore errors (some columns may not support indexing)
        return r.status_code < 400

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"VelesDBClient({self.base})"
