"""
Schema indexing utilities for the GraphQL MCP server.

This module expands a GraphQL SDL schema into executable paths rooted at Query/
Mutation (e.g., Query.user(id).posts.title), embeds each path summary via an
OpenAI-compatible embeddings endpoint, and persists normalized vectors to disk
for fast similarity search. It is used by the MCP server and the CLI to build
and search the local index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from graphql import GraphQLList, GraphQLNonNull, GraphQLObjectType, build_schema

from config import load_embedder_config
from embedding_client import EmbeddingClient

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_SCHEMA_PATH = Path(__file__).parent / "schema.graphql"
DEFAULT_MAX_PATH_DEPTH = int(os.environ.get("GRAPHQL_INDEX_MAX_DEPTH", "4"))
DEFAULT_EMBED_BATCH_SIZE = int(os.environ.get("GRAPHQL_EMBED_BATCH_SIZE", "128"))
logger = logging.getLogger("graphql-mcp")


@dataclass
class PathEntry:
    root_type: str
    path: str
    return_type: str
    summary: str


def describe_type(graphql_type) -> str:
    if isinstance(graphql_type, GraphQLNonNull):
        return f"{describe_type(graphql_type.of_type)}!"
    if isinstance(graphql_type, GraphQLList):
        return f"[{describe_type(graphql_type.of_type)}]"
    return str(graphql_type)


def _base_gql_type(graphql_type):
    base = graphql_type
    while isinstance(base, (GraphQLNonNull, GraphQLList)):
        base = base.of_type
    return base


def _format_args_inline(args: dict) -> tuple[str, str]:
    arg_parts = []
    arg_docs = []
    for arg_name, arg in args.items():
        arg_type = describe_type(arg.type)
        arg_parts.append(f"{arg_name}: {arg_type}")
        if arg.description:
            arg_docs.append(f"{arg_name}: {arg.description}")
    return ", ".join(arg_parts), "; ".join(arg_docs)


def _build_path_entries(schema_text: str, max_depth: int) -> List[PathEntry]:
    schema = build_schema(schema_text)
    entries: List[PathEntry] = []

    roots = [schema.query_type, schema.mutation_type, schema.subscription_type]
    for root in [root for root in roots if root]:
        root_name = root.name
        _walk_paths(
            root_type=root,
            root_name=root_name,
            parent_parts=[],
            depth=1,
            max_depth=max_depth,
            entries=entries,
            stack=[root_name],
        )
    return entries


def _walk_paths(
    root_type: GraphQLObjectType,
    root_name: str,
    parent_parts: list[str],
    depth: int,
    max_depth: int,
    entries: list[PathEntry],
    stack: list[str],
) -> None:
    if depth > max_depth:
        return

    for field_name, field in sorted(root_type.fields.items()):
        args_inline, args_docs = _format_args_inline(field.args)
        segment = f"{field_name}({args_inline})" if args_inline else field_name
        path = ".".join([root_name] + parent_parts + [segment])
        return_type = describe_type(field.type)

        summary_parts = [f"{path} -> {return_type}"]
        if field.description:
            summary_parts.append(f"desc: {field.description}")
        if args_inline:
            summary_parts.append(f"args: {args_inline}")
        if args_docs:
            summary_parts.append(f"arg_desc: {args_docs}")

        entries.append(
            PathEntry(
                root_type=root_name,
                path=path,
                return_type=return_type,
                summary=" | ".join(summary_parts),
            )
        )

        base_type = _base_gql_type(field.type)
        if isinstance(base_type, GraphQLObjectType):
            if base_type.name in stack:
                continue
            _walk_paths(
                root_type=base_type,
                root_name=root_name,
                parent_parts=parent_parts + [segment],
                depth=depth + 1,
                max_depth=max_depth,
                entries=entries,
                stack=stack + [base_type.name],
            )


class EmbeddingStore:
    def __init__(self, data_dir: Path, embedding_model: str):
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.meta_path = data_dir / "metadata.json"
        self.vectors_path = data_dir / "vectors.npz"

        self._vectors: np.ndarray | None = None
        self._items: list[dict] | None = None
        self._meta: dict | None = None

    def is_ready(self) -> bool:
        return self.meta_path.exists() and self.vectors_path.exists()

    def load(self) -> dict:
        if self._meta and self._vectors is not None and self._items is not None:
            return self._meta

        if not self.is_ready():
            raise FileNotFoundError(
                f"Index not found in {self.data_dir}. Run the indexer first."
            )

        self._meta = json.loads(self.meta_path.read_text())
        if self._meta.get("embedding_model") != self.embedding_model:
            raise ValueError(
                "Embedding model mismatch: "
                f"{self._meta.get('embedding_model')} vs {self.embedding_model}"
            )

        self._items = self._meta["items"]
        self._vectors = np.load(self.vectors_path)["vectors"]
        return self._meta

    def save(
        self,
        vectors: np.ndarray,
        items: list[dict],
        schema_sha: str,
        schema_source: dict | None = None,
        max_depth: int | None = None,
    ) -> dict:
        self.data_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "embedding_model": self.embedding_model,
            "schema_sha": schema_sha,
            "items": items,
        }
        if schema_source is not None:
            meta["schema_source"] = schema_source
        if max_depth is not None:
            meta["max_depth"] = max_depth

        np.savez_compressed(self.vectors_path, vectors=vectors)
        self.meta_path.write_text(json.dumps(meta, indent=2))

        self._vectors = vectors
        self._items = items
        self._meta = meta
        return meta

    def search(self, query_vector: np.ndarray, limit: int = 5) -> list[dict]:
        if self._vectors is None or self._items is None:
            self.load()

        assert self._vectors is not None and self._items is not None

        limit = max(1, min(limit, len(self._items)))
        scores = self._vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:limit]

        return [
            {
                "root_type": self._items[idx]["root_type"],
                "path": self._items[idx]["path"],
                "return_type": self._items[idx]["return_type"],
                "summary": self._items[idx]["summary"],
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


def compute_schema_sha(schema_text: str) -> str:
    return hashlib.sha256(schema_text.encode("utf-8")).hexdigest()


def index_schema_text(
    schema_text: str,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    embed_model: str | None = None,
    max_depth: int = DEFAULT_MAX_PATH_DEPTH,
    embedder: EmbeddingClient | None = None,
    store: EmbeddingStore | None = None,
    schema_source: dict | None = None,
) -> dict:
    items = _build_path_entries(schema_text, max_depth=max_depth)
    summaries = [item.summary for item in items]
    embedder = _resolve_embedder(embed_model, embedder)
    vectors = _embed_in_batches(embedder, summaries, batch_size=DEFAULT_EMBED_BATCH_SIZE)

    schema_sha = compute_schema_sha(schema_text)
    store = store or EmbeddingStore(data_dir=data_dir, embedding_model=embedder.model)
    meta = store.save(
        vectors,
        [asdict(item) for item in items],
        schema_sha=schema_sha,
        schema_source=schema_source,
        max_depth=max_depth,
    )
    meta["count"] = len(items)
    meta["indexed"] = True
    return meta


def _embed_in_batches(
    embedder: EmbeddingClient,
    texts: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    size = max(1, int(batch_size))
    total = len(texts)
    batches = (total + size - 1) // size
    chunks: list[np.ndarray] = []
    for idx in range(batches):
        start = idx * size
        end = min(start + size, total)
        logger.info(
            "Embedding %s/%s %s",
            end,
            total,
            _format_progress(end, total),
        )
        chunk = embedder.embed_many(texts[start:end])
        chunks.append(chunk)
    return np.vstack(chunks)


def _format_progress(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[--------------------] 0%"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {int(ratio * 100)}%"


def index_schema(
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    embed_model: str | None = None,
    max_depth: int = DEFAULT_MAX_PATH_DEPTH,
    embedder: EmbeddingClient | None = None,
    store: EmbeddingStore | None = None,
    schema_source: dict | None = None,
) -> dict:
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
        max_depth=max_depth,
        embedder=embedder,
        store=store,
        schema_source=resolved_source,
    )


def ensure_index_text(
    schema_text: str,
    *,
    schema_source: dict,
    data_dir: Path = DEFAULT_DATA_DIR,
    embed_model: str | None = None,
    max_depth: int = DEFAULT_MAX_PATH_DEPTH,
    embedder: EmbeddingClient | None = None,
    store: EmbeddingStore | None = None,
    force: bool = False,
) -> dict:
    """
    Ensure a persisted embedding index exists for a given schema text.

    Rebuilds the index if missing, corrupt, model-mismatched, schema changed, or schema source changed.
    """
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
        if (
            meta.get("schema_sha") == schema_sha
            and (stored_source is None or stored_source == schema_source)
            and meta.get("max_depth") == max_depth
        ):
            meta["count"] = len(meta.get("items", []))
            meta["indexed"] = False
            return meta

    return index_schema_text(
        schema_text,
        data_dir=data_dir,
        embed_model=embedder.model,
        max_depth=max_depth,
        embedder=embedder,
        store=store,
        schema_source=schema_source,
    )


def ensure_index(
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    embed_model: str | None = None,
    max_depth: int = DEFAULT_MAX_PATH_DEPTH,
    embedder: EmbeddingClient | None = None,
    store: EmbeddingStore | None = None,
    force: bool = False,
) -> dict:
    """
    Ensure a persisted embedding index exists for the given schema.

    Rebuilds the index if missing, corrupt, model-mismatched, or if the schema file changed.
    """
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
        if (
            meta.get("schema_sha") == schema_sha
            and (stored_source is None or stored_source == schema_source)
            and meta.get("max_depth") == max_depth
        ):
            meta["count"] = len(meta.get("items", []))
            meta["indexed"] = False
            return meta

    return index_schema(
        schema_path=schema_path,
        data_dir=data_dir,
        embed_model=embedder.model,
        max_depth=max_depth,
        embedder=embedder,
        store=store,
        schema_source=schema_source,
    )


def search_index(
    query: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    embed_model: str | None = None,
    max_depth: int = DEFAULT_MAX_PATH_DEPTH,
    embedder: EmbeddingClient | None = None,
    limit: int = 5,
) -> list[dict]:
    embedder = _resolve_embedder(embed_model, embedder)
    store = EmbeddingStore(data_dir=data_dir, embedding_model=embedder.model)
    meta = store.load()
    if meta.get("max_depth") != max_depth:
        raise ValueError("Index max depth mismatch. Rebuild the index.")

    query_vector = embedder.embed_one(query)
    results = store.search(query_vector, limit=limit)
    for item in results:
        item["schema_sha"] = meta.get("schema_sha")
    return results


def cli(argv: Iterable[str] | None = None) -> int:
    # Parse arguments and set defaults properly
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Embedding model to use (defaults from env)")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH, help="Path to the GraphQL schema file")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to store data files")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_PATH_DEPTH, help="Max path depth to index")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Index subcommand
    index_parser = subparsers.add_parser("index", help="Index the schema into persistent embeddings")
    index_parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH, help="Path to the GraphQL schema file")

    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Search the persisted index with a natural language query")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")

    args = parser.parse_args(argv)

    # Get the selected model (either from --model or default)
    config = load_embedder_config()
    model_arg = args.model or config.model
    max_depth = max(1, int(args.max_depth))
    embedder = EmbeddingClient(config=config, model=model_arg)

    if args.command == "search":
        limit = max(1, min(getattr(args, "limit", 5), 20))
        ensure_index(
            schema_path=getattr(args, "schema", DEFAULT_SCHEMA_PATH),
            data_dir=getattr(args, "data_dir", DEFAULT_DATA_DIR),
            embed_model=model_arg,
            max_depth=max_depth,
            embedder=embedder,
            force=False,
        )
        results = search_index(
            query=args.query,
            data_dir=getattr(args, "data_dir", DEFAULT_DATA_DIR),
            embed_model=model_arg,
            max_depth=max_depth,
            embedder=embedder,
            limit=limit,
        )
        print(json.dumps(results, indent=2))
        return 0

    meta = index_schema(
        schema_path=args.schema,
        data_dir=args.data_dir,
        embed_model=model_arg,
        max_depth=max_depth,
        embedder=embedder,
    )
    print(
        f"Indexed {meta['count']} paths from {args.schema} "
        f"using {meta['embedding_model']} (schema sha {meta['schema_sha']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
