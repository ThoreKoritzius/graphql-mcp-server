from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from schema_indexer import EmbeddingStore, index_schema_text
from schema_navigation import build_field_nodes
import server

SCHEMA = """
type Query {
  users(limit: Int = 10): [User!]!
  user(id: ID!): User
  reportsConnection(first: Int): ReportConnection!
}

type User {
  id: ID!
  name: String!
  orders: [Order!]!
}

type Order {
  id: ID!
  total: Float!
  status: String!
}

type ReportConnection {
  edges: [ReportEdge!]!
  totalCount: Int!
}

type ReportEdge {
  node: Report!
}

type Report {
  id: ID!
  title: String!
}

type AdminOnly {
  secret: String!
}
"""


class FakeEmbedder:
    def __init__(self, model: str = "fake-embed"):
        self.model = model

    def _vectorize(self, text: str) -> np.ndarray:
        buckets = np.zeros(16, dtype=np.float32)
        for idx, ch in enumerate(text.lower()):
            buckets[(ord(ch) + idx) % len(buckets)] += 1.0
        norm = np.linalg.norm(buckets)
        if norm:
            buckets /= norm
        return buckets

    def embed_many(self, texts):
        return np.vstack([self._vectorize(text) for text in texts])

    def embed_one(self, text: str):
        return self._vectorize(text)


class NavigationIndexTests(unittest.TestCase):
    def test_build_field_nodes_includes_coordinates(self):
        nodes = build_field_nodes(SCHEMA)
        by_key = {(node.type_name, node.field_name): node for node in nodes}

        self.assertEqual(
            by_key[("Order", "total")].coordinates,
            "Query.user(id: <ID!>) -> User.orders -> Order.total",
        )
        self.assertEqual(
            by_key[("Query", "users")].coordinates,
            "Query.users(limit: <Int>)",
        )
        self.assertIsNone(by_key[("AdminOnly", "secret")].coordinates)

    def test_hybrid_search_finds_nested_field_from_fuzzy_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = FakeEmbedder()
            store = EmbeddingStore(Path(tmpdir), embedding_model=embedder.model)
            index_schema_text(SCHEMA, data_dir=Path(tmpdir), embedder=embedder, store=store)

            results = store.hybrid_search("orde totl", embedder.embed_one("orde totl"), limit=5)

            self.assertEqual(results[0]["type_name"], "Order")
            self.assertEqual(results[0]["field_name"], "total")
            self.assertTrue(results[0]["coordinates"].startswith("Query."))
            self.assertGreater(results[0]["lexical_score"], 0.7)

    def test_list_types_returns_coordinates_query_and_select(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = FakeEmbedder()
            store = EmbeddingStore(Path(tmpdir), embedding_model=embedder.model)

            original_embedder = server.embedder
            original_store = server.store
            original_data_dir = server.DATA_DIR
            original_endpoint = server.ENDPOINT_URL
            original_schema_text = server.SCHEMA_TEXT
            original_schema_source = dict(server.SCHEMA_SOURCE)
            original_model = server.EMBED_MODEL
            original_path = server.SCHEMA_PATH
            try:
                server.DATA_DIR = Path(tmpdir)
                server.ENDPOINT_URL = "http://example.test/graphql"
                server.SCHEMA_TEXT = SCHEMA
                server.SCHEMA_SOURCE = {"kind": "endpoint", "url": "http://example.test/graphql", "headers": []}
                server.EMBED_MODEL = embedder.model
                server.embedder = embedder
                server.store = store
                server.SCHEMA_PATH = Path("<endpoint>")

                nested = server.list_types("order total", limit=3)
                query_root = server.list_types("users", limit=3)

                nested_top = nested[0]
                self.assertEqual(nested_top["type"], "Order")
                self.assertEqual(nested_top["field"], "total")
                self.assertIn("coordinates", nested_top)
                self.assertNotIn("query", nested_top)

                query_match = next(item for item in query_root if item["field"] == "users")
                self.assertIn("coordinates", query_match)
                self.assertIn("query", query_match)
                self.assertTrue(query_match["query"].startswith("query { users"))
            finally:
                server.embedder = original_embedder
                server.store = original_store
                server.DATA_DIR = original_data_dir
                server.ENDPOINT_URL = original_endpoint
                server.SCHEMA_TEXT = original_schema_text
                server.SCHEMA_SOURCE = original_schema_source
                server.EMBED_MODEL = original_model
                server.SCHEMA_PATH = original_path


if __name__ == "__main__":
    unittest.main()
