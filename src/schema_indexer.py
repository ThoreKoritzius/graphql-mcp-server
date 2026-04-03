"""
Schema indexing utilities for the GraphQL MCP server.

This module builds a structured field-navigation index from a GraphQL SDL schema,
embeds the generated search documents via an OpenAI-compatible embeddings endpoint,
and persists normalized vectors plus field metadata to disk for embedding-similarity retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from config import load_embedder_config
from embedding_client import EmbeddingClient
from schema_navigation import build_field_nodes

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_SCHEMA_PATH = Path(__file__).parent / "schema.graphql"
DEFAULT_EMBED_BATCH_SIZE = 128
INDEX_VERSION = 3


class EmbeddingStore:
    def __init__(self, data_dir: Path, embedding_model: str):
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.meta_path = data_dir / "metadata.json"
        self.vectors_path = data_dir / "vectors.npz"

        self._vectors: np.ndarray | None = None
        self._items: list[dict[str, Any]] | None = None
        self._meta: dict[str, Any] | None = None

    def is_ready(self) -> bool:
        return self.meta_path.exists() and self.vectors_path.exists()

    def load(self) -> dict[str, Any]:
        if self._meta and self._vectors is not None and self._items is not None:
            return self._meta

        if not self.is_ready():
            raise FileNotFoundError(f"Index not found in {self.data_dir}. Run the indexer first.")

        self._meta = json.loads(self.meta_path.read_text())
        if self._meta.get("embedding_model") != self.embedding_model:
            raise ValueError(
                "Embedding model mismatch: "
                f"{self._meta.get('embedding_model')} vs {self.embedding_model}"
            )
        if self._meta.get("index_version") != INDEX_VERSION:
            raise ValueError(
                f"Index version mismatch: {self._meta.get('index_version')} vs {INDEX_VERSION}"
            )

        self._items = self._meta["items"]
        self._vectors = np.load(self.vectors_path)["vectors"]
        if len(self._items) != len(self._vectors):
            raise ValueError("Corrupt index: item count does not match vector count")
        return self._meta

    def save(
        self,
        vectors: np.ndarray,
        items: list[dict[str, Any]],
        schema_sha: str,
        schema_source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.data_dir.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "index_version": INDEX_VERSION,
            "embedding_model": self.embedding_model,
            "schema_sha": schema_sha,
            "items": items,
        }
        if schema_source is not None:
            meta["schema_source"] = schema_source

        np.savez_compressed(self.vectors_path, vectors=vectors)
        self.meta_path.write_text(json.dumps(meta, indent=2))

        self._vectors = vectors
        self._items = items
        self._meta = meta
        return meta

    def search(self, query_vector: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
        if self._vectors is None or self._items is None:
            self.load()

        assert self._vectors is not None and self._items is not None
        if not len(self._items):
            return []

        limit = max(1, min(limit, len(self._items)))
        scores = self._vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:limit]
        return [
            {
                **self._items[idx],
                "score": float(scores[idx]),
            }
            for idx in top_indices
        ]


def _resolve_embedder(
    embed_model: str | None,
    embedder: EmbeddingClient | None,
) -> EmbeddingClient:
    if embedder is not None:
        return embedder
    config = load_embedder_config()
    resolved_model = embed_model or config.model
    return EmbeddingClient(config=config, model=resolved_model)


def _render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return f"[{'-' * width}] 0/0 (0%)"
    ratio = min(1.0, current / total)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(ratio * 100)
    return f"[{bar}] {current}/{total} ({percent}%)"


def _embed_with_progress(embedder: EmbeddingClient, texts: list[str]) -> np.ndarray:
    total = len(texts)
    if total == 0:
        return np.zeros((0, 0), dtype=np.float32)

    import os
    import sys

    batch_size_raw = os.environ.get("GRAPHQL_EMBED_BATCH_SIZE", str(DEFAULT_EMBED_BATCH_SIZE))
    try:
        batch_size = max(1, int(batch_size_raw))
    except ValueError:
        batch_size = DEFAULT_EMBED_BATCH_SIZE

    vectors: list[np.ndarray] = []
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        vectors.append(embedder.embed_many(batch))
        current = min(start + len(batch), total)
        progress = _render_progress(current, total)
        print(f"\rIndexing embeddings {progress}", end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr, flush=True)
    return np.vstack(vectors)


def compute_schema_sha(schema_text: str) -> str:
    return hashlib.sha256(schema_text.encode("utf-8")).hexdigest()


def index_schema_text(
    schema_text: str,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    embed_model: str | None = None,
    embedder: EmbeddingClient | None = None,
    store: EmbeddingStore | None = None,
    schema_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes = build_field_nodes(schema_text)
    items = [asdict(node) for node in nodes]
    search_texts = [item["search_text"] for item in items]

    embedder = _resolve_embedder(embed_model, embedder)
    vectors = _embed_with_progress(embedder, search_texts)
    schema_sha = compute_schema_sha(schema_text)

    store = store or EmbeddingStore(data_dir=data_dir, embedding_model=embedder.model)
    meta = store.save(vectors, items, schema_sha=schema_sha, schema_source=schema_source)
    meta["count"] = len(items)
    meta["indexed"] = True
    return meta


def index_schema(
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    embed_model: str | None = None,
    embedder: EmbeddingClient | None = None,
    store: EmbeddingStore | None = None,
    schema_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_source = schema_source
    if resolved_source is None:
        try:
            resolved_source = {"kind": "file", "path": str(schema_path.resolve())}
        except Exception:
            resolved_source = {"kind": "file", "path": str(schema_path)}
    return index_schema_text(
        schema_path.read_text(),
        data_dir=data_dir,
        embed_model=embed_model,
        embedder=embedder,
        store=store,
        schema_source=resolved_source,
    )


def ensure_index_text(
    schema_text: str,
    *,
    schema_source: dict[str, Any],
    data_dir: Path = DEFAULT_DATA_DIR,
    embed_model: str | None = None,
    embedder: EmbeddingClient | None = None,
    store: EmbeddingStore | None = None,
    force: bool = False,
) -> dict[str, Any]:
    embedder = _resolve_embedder(embed_model, embedder)
    store = store or EmbeddingStore(data_dir=data_dir, embedding_model=embedder.model)

    if not force and store.is_ready():
        schema_sha = compute_schema_sha(schema_text)
        try:
            meta = store.load()
        except Exception:
            return index_schema_text(
                schema_text,
                data_dir=data_dir,
                embed_model=embedder.model,
                embedder=embedder,
                store=store,
                schema_source=schema_source,
            )

        stored_source = meta.get("schema_source")
        if meta.get("schema_sha") == schema_sha and (stored_source is None or stored_source == schema_source):
            meta["count"] = len(meta.get("items", []))
            meta["indexed"] = False
            return meta

    return index_schema_text(
        schema_text,
        data_dir=data_dir,
        embed_model=embedder.model,
        embedder=embedder,
        store=store,
        schema_source=schema_source,
    )


def ensure_index(
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    embed_model: str | None = None,
    embedder: EmbeddingClient | None = None,
    store: EmbeddingStore | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if not schema_path.exists():
        raise FileNotFoundError(
            "Schema file not found. Provide --schema or set GRAPHQL_SCHEMA_PATH, "
            "or use GRAPHQL_ENDPOINT_URL for live endpoint mode."
        )
    embedder = _resolve_embedder(embed_model, embedder)
    store = store or EmbeddingStore(data_dir=data_dir, embedding_model=embedder.model)
    try:
        schema_source = {"kind": "file", "path": str(schema_path.resolve())}
    except Exception:
        schema_source = {"kind": "file", "path": str(schema_path)}

    if not force and store.is_ready():
        schema_text = schema_path.read_text()
        schema_sha = compute_schema_sha(schema_text)
        try:
            meta = store.load()
        except Exception:
            return index_schema(
                schema_path=schema_path,
                data_dir=data_dir,
                embed_model=embedder.model,
                embedder=embedder,
                store=store,
                schema_source=schema_source,
            )

        stored_source = meta.get("schema_source")
        if meta.get("schema_sha") == schema_sha and (stored_source is None or stored_source == schema_source):
            meta["count"] = len(meta.get("items", []))
            meta["indexed"] = False
            return meta

    return index_schema(
        schema_path=schema_path,
        data_dir=data_dir,
        embed_model=embedder.model,
        embedder=embedder,
        store=store,
        schema_source=schema_source,
    )


def search_index(
    query: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    embed_model: str | None = None,
    embedder: EmbeddingClient | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    embedder = _resolve_embedder(embed_model, embedder)
    store = EmbeddingStore(data_dir=data_dir, embedding_model=embedder.model)
    meta = store.load()
    query_vector = embedder.embed_one(query)
    results = store.search(query_vector, limit=limit)
    for item in results:
        item["schema_sha"] = meta.get("schema_sha")
    return results


def cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Embedding model to use (defaults from env)")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH, help="Path to the GraphQL schema file")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to store data files")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    index_parser = subparsers.add_parser("index", help="Index the schema into persistent embeddings")
    index_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH, help="Path to the GraphQL schema file")

    search_parser = subparsers.add_parser("search", help="Search the persisted index with a natural language query")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")

    args = parser.parse_args(argv)

    config = load_embedder_config()
    model_arg = args.model or config.model
    embedder = EmbeddingClient(config=config, model=model_arg)

    if args.command == "search":
        limit = max(1, min(getattr(args, "limit", 5), 20))
        ensure_index(
            schema_path=getattr(args, "schema", DEFAULT_SCHEMA_PATH),
            data_dir=getattr(args, "data_dir", DEFAULT_DATA_DIR),
            embed_model=model_arg,
            embedder=embedder,
            force=False,
        )
        results = search_index(
            query=args.query,
            data_dir=getattr(args, "data_dir", DEFAULT_DATA_DIR),
            embed_model=model_arg,
            embedder=embedder,
            limit=limit,
        )
        print(json.dumps(results, indent=2))
        return 0

    meta = index_schema(
        schema_path=args.schema,
        data_dir=args.data_dir,
        embed_model=model_arg,
        embedder=embedder,
    )
    print(
        f"Indexed {meta['count']} fields from {args.schema} "
        f"using {meta['embedding_model']} (schema sha {meta['schema_sha']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
