"""
GraphQL MCP server exposing `list_types` and `run_query` tools over a schema file or live endpoint.

What it does:
- Builds and caches a navigation index of GraphQL field nodes.
- Exposes `list_types` for fuzzy discovery with Query-root coordinates.
- Exposes `run_query` for validation/execution.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from graphql import (
    build_client_schema,
    build_schema,
    get_introspection_query,
    graphql_sync,
    print_schema,
)
from mcp.server.fastmcp import FastMCP

from config import load_embedder_config
from embedding_client import EmbeddingClient
from schema_indexer import DEFAULT_DATA_DIR, DEFAULT_SCHEMA_PATH, EmbeddingStore, ensure_index, ensure_index_text
from schema_navigation import tokenize, unwrap_named_type_name

APP_NAME = "graphql-mcp"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATHS = [Path.cwd() / ".env", _REPO_ROOT / ".env"]
for _path in _ENV_PATHS:
    if _path.exists():
        load_dotenv(_path, override=True)

DEFAULT_TRANSPORT = os.environ.get("MCP_TRANSPORT", os.environ.get("FASTMCP_TRANSPORT", "sse"))
DEFAULT_INSTRUCTIONS = (
    "You are an information lookup assistant. Treat this MCP server as an abstraction layer for GraphQL. "
    "For any user question, first call list_types with a focused query. Use coordinates to navigate from Query "
    "to nested fields. Then call run_query with a single, valid query. Avoid unnecessary tool calls."
)
MCP_INSTRUCTIONS = os.environ.get("MCP_INSTRUCTIONS", DEFAULT_INSTRUCTIONS)

SCHEMA_PATH = Path(os.environ.get("GRAPHQL_SCHEMA_PATH", str(DEFAULT_SCHEMA_PATH)))
ENDPOINT_URL = os.environ.get("GRAPHQL_ENDPOINT_URL")
DATA_DIR = Path(os.environ.get("GRAPHQL_EMBEDDER_DATA_DIR", str(DEFAULT_DATA_DIR)))
EMBEDDER_CONFIG = load_embedder_config()
EMBED_MODEL = EMBEDDER_CONFIG.model

embedder = EmbeddingClient(config=EMBEDDER_CONFIG, model=EMBED_MODEL)
store = EmbeddingStore(data_dir=DATA_DIR, embedding_model=embedder.model)
SCHEMA_SOURCE: dict[str, Any] = {"kind": "file", "path": str(SCHEMA_PATH)}
SCHEMA_TEXT: str | None = None

mcp = FastMCP(APP_NAME, instructions=MCP_INSTRUCTIONS)
mcp.dependencies = ["graphql-core", "numpy", "aiohttp"]
logger = logging.getLogger(APP_NAME)

_REMOTE_TIMEOUT_S: float = 30.0
_INDEX_LOCK = threading.Lock()
_LEAF_TYPES = {"String", "Int", "Float", "Boolean", "ID"}


def _load_endpoint_headers_from_env() -> dict[str, str]:
    raw = os.environ.get("GRAPHQL_ENDPOINT_HEADERS")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("GRAPHQL_ENDPOINT_HEADERS must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("GRAPHQL_ENDPOINT_HEADERS must be a JSON object")
    return {str(key): str(val) for key, val in parsed.items()}


_REMOTE_HEADERS: dict[str, str] = _load_endpoint_headers_from_env()


def _run_with_default_transport(
    self,
    transport: Literal["stdio", "sse", "streamable-http"] | None = None,
    mount_path: str | None = None,
):
    chosen = transport or DEFAULT_TRANSPORT
    return FastMCP.run(self, transport=chosen, mount_path=mount_path)


mcp.run = _run_with_default_transport.__get__(mcp, FastMCP)


def _run_indexing_or_exit() -> None:
    try:
        logger.info("Preparing schema index on startup...")
        meta = ensure_schema_indexed(force=False)
        if not meta.get("indexed"):
            logger.info("Schema index up-to-date (%s fields).", meta.get("count", 0))
    except Exception as exc:
        logger.error("Schema indexing failed: %s", exc)
        os._exit(1)


def configure_runtime(*, schema_path: Path, data_dir: Path, embed_model: str) -> None:
    global SCHEMA_PATH, ENDPOINT_URL, DATA_DIR, EMBED_MODEL, embedder, store, SCHEMA_SOURCE, SCHEMA_TEXT
    SCHEMA_PATH = schema_path
    ENDPOINT_URL = None
    DATA_DIR = data_dir
    EMBED_MODEL = embed_model
    embedder = EmbeddingClient(config=EMBEDDER_CONFIG, model=EMBED_MODEL)
    store = EmbeddingStore(data_dir=DATA_DIR, embedding_model=embedder.model)
    SCHEMA_SOURCE = {"kind": "file", "path": str(SCHEMA_PATH)}
    SCHEMA_TEXT = None


def configure_runtime_endpoint(
    *,
    endpoint_url: str,
    data_dir: Path,
    embed_model: str,
    schema_text: str,
    schema_source: dict[str, Any],
) -> None:
    global SCHEMA_PATH, ENDPOINT_URL, DATA_DIR, EMBED_MODEL, embedder, store, SCHEMA_SOURCE, SCHEMA_TEXT
    SCHEMA_PATH = Path("<endpoint>")
    ENDPOINT_URL = endpoint_url
    DATA_DIR = data_dir
    EMBED_MODEL = embed_model
    embedder = EmbeddingClient(config=EMBEDDER_CONFIG, model=EMBED_MODEL)
    store = EmbeddingStore(data_dir=DATA_DIR, embedding_model=embedder.model)
    SCHEMA_SOURCE = schema_source
    SCHEMA_TEXT = schema_text


def _parse_headers(raw_headers: list[str] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw in raw_headers or []:
        if ":" not in raw:
            raise ValueError(f"Invalid header (expected 'Name: Value'): {raw}")
        name, value = raw.split(":", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"Invalid header name in: {raw}")
        headers[name] = value
    return headers


def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None, timeout_s: float = 30.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)

    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"errors": [{"message": raw}]}
        raise


def _introspect_schema_sdl(endpoint_url: str, headers: dict[str, str], timeout_s: float) -> str:
    payload = {
        "query": get_introspection_query(descriptions=True),
        "operationName": "IntrospectionQuery",
        "variables": {},
    }
    result = _post_json(endpoint_url, payload, headers=headers, timeout_s=timeout_s)
    if result.get("errors"):
        raise RuntimeError(f"Introspection failed: {result['errors']}")
    data = result.get("data")
    if not data:
        raise RuntimeError("Introspection response missing 'data'.")
    schema = build_client_schema(data)
    return print_schema(schema)


def ensure_schema_indexed(*, force: bool = False) -> dict[str, Any]:
    try:
        with _INDEX_LOCK:
            if ENDPOINT_URL:
                if not SCHEMA_TEXT:
                    raise RuntimeError("Endpoint mode requires schema introspection text.")
                logger.info("Indexing schema from endpoint %s...", ENDPOINT_URL)
                meta = ensure_index_text(
                    SCHEMA_TEXT,
                    schema_source=SCHEMA_SOURCE,
                    data_dir=DATA_DIR,
                    embed_model=EMBED_MODEL,
                    embedder=embedder,
                    store=store,
                    force=force,
                )
                if meta.get("indexed"):
                    logger.info("Indexed %s fields from endpoint schema.", meta.get("count", 0))
                return meta
            logger.info("Indexing schema from file %s...", SCHEMA_PATH)
            meta = ensure_index(
                schema_path=SCHEMA_PATH,
                data_dir=DATA_DIR,
                embed_model=EMBED_MODEL,
                embedder=embedder,
                store=store,
                force=force,
            )
            if meta.get("indexed"):
                logger.info("Indexed %s fields from schema %s.", meta.get("count", 0), SCHEMA_PATH)
            return meta
    except Exception as exc:
        raise RuntimeError(f"Schema index not available for {SCHEMA_PATH}: {exc}")


def _fields_by_type(meta: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in meta.get("items", []):
        grouped.setdefault(item.get("type_name", ""), []).append(item)
    return grouped


def _format_arg_placeholders(args: list[dict[str, str]]) -> str:
    if not args:
        return ""
    rendered = ", ".join(f"{arg['name']}: <{arg['type']}>" for arg in args)
    return f"({rendered})"


def _render_selection_set(
    type_name: str,
    fields_by_type: dict[str, list[dict[str, Any]]],
    tokens: list[str],
    depth: int = 1,
    max_fields: int = 6,
) -> str | None:
    fields = list(fields_by_type.get(type_name, []))
    if not fields:
        return None

    def rank(field: dict[str, Any]) -> tuple[float, int, str]:
        score = 0.0
        field_name = field.get("field_name", "")
        if field_name in {"id", "name"}:
            score += 0.5
        if tokens and any(token in field_name.lower() for token in tokens):
            score += 0.35
        return_type = unwrap_named_type_name(field.get("return_type", ""))
        leaf_bonus = 1 if return_type in _LEAF_TYPES or field.get("is_scalar_return") else 0
        return (score, leaf_bonus, field_name)

    fields.sort(key=rank, reverse=True)
    selections: list[str] = []
    for field in fields:
        if len(selections) >= max_fields:
            break
        field_name = field.get("field_name", "")
        if field.get("is_scalar_return"):
            selections.append(field_name)
            continue
        if depth <= 0:
            continue
        nested_type = unwrap_named_type_name(field.get("return_type", ""))
        nested = _render_selection_set(
            nested_type,
            fields_by_type,
            tokens,
            depth=depth - 1,
            max_fields=max(2, max_fields // 2),
        )
        if nested:
            selections.append(f"{field_name} {nested}")

    if not selections:
        return None
    return "{ " + " ".join(selections) + " }"


def _format_query_template(node: dict[str, Any], fields_by_type: dict[str, list[dict[str, Any]]], tokens: list[str]) -> str:
    field_name = node["field_name"]
    args = _format_arg_placeholders(node.get("args", []))
    if node.get("is_scalar_return"):
        return f"query {{ {field_name}{args} }}"

    selection = _render_selection_set(
        unwrap_named_type_name(node.get("return_type", "")),
        fields_by_type,
        tokens,
        depth=2 if node.get("is_connection") else 1,
        max_fields=8 if node.get("is_connection") else 6,
    )
    selection_part = f" {selection}" if selection else ""
    return f"query {{ {field_name}{args}{selection_part} }}"


def _format_result(node: dict[str, Any], fields_by_type: dict[str, list[dict[str, Any]]], tokens: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "field": node["field_name"],
        "summary": node["summary"],
    }
    if node.get("type_name") != "Query":
        result["type"] = node.get("type_name")
    if node.get("coordinates"):
        result["coordinates"] = node["coordinates"]
    if node.get("is_query_root"):
        result["query"] = _format_query_template(node, fields_by_type, tokens)
    elif not node.get("is_scalar_return"):
        nested = _render_selection_set(
            unwrap_named_type_name(node.get("return_type", "")),
            fields_by_type,
            tokens,
            depth=1,
            max_fields=5,
        )
        if nested:
            result["select"] = f"{node['field_name']} {nested}"
    return result


@mcp.tool()
def list_types(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Fuzzy search the schema for matching type.field nodes.
    Uses the persisted navigation index (auto-builds if missing/outdated).
    """
    meta = ensure_schema_indexed(force=False)
    fields_by_type = _fields_by_type(meta)
    tokens = tokenize(query)
    capped_limit = max(1, min(limit, 20))
    query_vec = embedder.embed_one(query)
    ranked = store.search(query_vec, limit=capped_limit)
    return [_format_result(item, fields_by_type, tokens) for item in ranked]


@mcp.tool()
def run_query(query: str) -> dict[str, Any]:
    """
    Validate and run a GraphQL query against the static schema.

    Note: No resolvers are provided in local mode, so fields resolve to null;
    this is mainly for validation and shape checking.
    """
    if ENDPOINT_URL:
        try:
            result = _post_json(ENDPOINT_URL, {"query": query}, headers=_REMOTE_HEADERS, timeout_s=_REMOTE_TIMEOUT_S)
        except Exception as exc:
            raise RuntimeError(f"Endpoint query failed: {exc}")
        output: dict[str, Any] = {"valid": not bool(result.get("errors"))}
        if "errors" in result:
            output["errors"] = _augment_endpoint_errors(result["errors"])
        if "data" in result:
            output["data"] = result["data"]
        if "extensions" in result:
            output["extensions"] = result["extensions"]
        return output

    schema = build_schema(SCHEMA_PATH.read_text())
    result = graphql_sync(schema, query)
    output: dict[str, Any] = {"valid": not bool(result.errors)}
    if result.errors:
        output["errors"] = [str(err) for err in result.errors]
    if result.data is not None:
        output["data"] = result.data
    return output


def _augment_endpoint_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not errors:
        return errors
    try:
        meta = ensure_schema_indexed(force=False)
        query_fields = _fields_by_type(meta).get("Query", [])
    except Exception:
        query_fields = []

    augmented: list[dict[str, Any]] = []
    for err in errors:
        augmented.append(err)
        message = str(err.get("message", ""))
        path = err.get("path") or []
        if "Expected Iterable" in message and path:
            field_name = str(path[0])
            connection_name = f"{field_name}Connection"
            has_connection = any(field.get("field_name") == connection_name for field in query_fields)
            if has_connection:
                augmented.append(
                    {
                        "message": (
                            f"Hint: `{field_name}` returned a non-list. "
                            f"Try `{connection_name}` for a connection-based query."
                        )
                    }
                )
        if "Cannot query field" in message:
            augmented.append(
                {"message": "Hint: Run `list_types` with a focused query to get coordinates from `Query`."}
            )
    return augmented


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the GraphQL embedder MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=DEFAULT_TRANSPORT,
        help="MCP transport to run (default: sse; override with --transport or MCP_TRANSPORT env).",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--schema", type=Path, default=SCHEMA_PATH, help="Path to a GraphQL schema file (SDL).")
    source_group.add_argument("--endpoint", default=ENDPOINT_URL, help="GraphQL endpoint URL (uses introspection for indexing and proxies queries).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory for the persisted embedding index (default: data/ next to this server).",
    )
    parser.add_argument("--model", default=EMBED_MODEL, help="Embedding model to use for indexing/search queries.")
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Add an HTTP header for endpoint mode, like 'Authorization: Bearer ...' (repeatable).",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout (seconds) for endpoint introspection/querying.")
    parser.add_argument("--host", default=mcp.settings.host, help="Host for SSE/HTTP transports (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=mcp.settings.port, help="Port for SSE/HTTP transports (default: 8000).")
    parser.add_argument("--log-level", default=mcp.settings.log_level, help="Log level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--mount-path", default=mcp.settings.mount_path, help="Mount path for SSE transport (default: /).")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    _REMOTE_HEADERS = _load_endpoint_headers_from_env()
    _REMOTE_HEADERS.update(_parse_headers(args.header))
    _REMOTE_TIMEOUT_S = float(args.timeout)

    endpoint_url = args.endpoint or os.environ.get("GRAPHQL_ENDPOINT_URL")
    if endpoint_url is not None:
        endpoint_url = endpoint_url.strip()
    if endpoint_url:
        try:
            schema_text = _introspect_schema_sdl(endpoint_url, headers=_REMOTE_HEADERS, timeout_s=_REMOTE_TIMEOUT_S)
        except Exception as exc:
            raise SystemExit(
                "Failed to introspect endpoint for schema indexing. "
                "Check GRAPHQL_ENDPOINT_URL and optional GRAPHQL_ENDPOINT_HEADERS.\n"
                f"Error: {exc}"
            ) from exc
        configure_runtime_endpoint(
            endpoint_url=endpoint_url,
            data_dir=args.data_dir,
            embed_model=args.model,
            schema_text=schema_text,
            schema_source={"kind": "endpoint", "url": endpoint_url, "headers": sorted(_REMOTE_HEADERS.keys())},
        )
    else:
        if not args.schema.exists():
            checked = ", ".join(str(path) for path in _ENV_PATHS)
            raise SystemExit(
                "Schema file not found and GRAPHQL_ENDPOINT_URL not set. "
                "Provide --schema or set GRAPHQL_SCHEMA_PATH, or set GRAPHQL_ENDPOINT_URL "
                f"in one of: {checked}"
            )
        configure_runtime(schema_path=args.schema, data_dir=args.data_dir, embed_model=args.model)

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.settings.log_level = args.log_level
    mcp.settings.mount_path = args.mount_path

    print(
        f"Starting {APP_NAME} with transport={args.transport}, "
        f"host={mcp.settings.host}, port={mcp.settings.port}, "
        f"schema={SCHEMA_PATH}",
        flush=True,
    )
    threading.Thread(target=lambda: _run_indexing_or_exit(), daemon=True, name="graphql-mcp-indexer").start()
    mcp.run(transport=args.transport, mount_path=args.mount_path)
