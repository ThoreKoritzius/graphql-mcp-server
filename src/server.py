"""
GraphQL MCP server exposing `list_types` and `run_query` tools over a schema file or live endpoint.

What it does:
- Builds and caches an embedding index of executable schema paths for fuzzy search.
- Exposes `list_types` for discovery and `run_query` for validation/execution.

How `list_types` works:
- Embed the user query and search the persisted index.
- Score hits using a weighted combination of embedding similarity + heuristics.
- Build a ready-to-run `query` from each executable path.

How `run_query` works:
- Local mode: validates against the SDL schema (no resolvers, so data is null-only).
- Endpoint mode: proxies the query to the remote URL using introspection-derived SDL for indexing.

Startup notes:
- Can auto-index in a background thread.
- Supports stdio/SSE/HTTP transports configured via env or CLI flags.
"""

import os
import json
import logging
import threading
import concurrent.futures
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from graphql import (
    GraphQLList,
    GraphQLNonNull,
    GraphQLObjectType,
    build_client_schema,
    build_schema,
    get_introspection_query,
    graphql_sync,
    print_schema,
)
from mcp.server.fastmcp import FastMCP

from config import load_embedder_config
from embedding_client import EmbeddingClient
from schema_indexer import (
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_PATH_DEPTH,
    DEFAULT_SCHEMA_PATH,
    EmbeddingStore,
    ensure_index,
    ensure_index_text,
)

APP_NAME = "graphql-mcp"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATHS = [Path.cwd() / ".env", _REPO_ROOT / ".env"]
for _path in _ENV_PATHS:
    if _path.exists():
        load_dotenv(_path, override=True)

DEFAULT_TRANSPORT = os.environ.get("MCP_TRANSPORT", os.environ.get("FASTMCP_TRANSPORT", "sse"))
DEFAULT_INSTRUCTIONS = (
    "You are an information lookup assistant. Treat this MCP server as an abstraction layer for GraphQL "
    "For any user question, first call list_types with a focused query. Prefer Query fields "
    "and their query. Then call run_query with a single, valid query. Avoid unnecessary tool calls."
)
MCP_INSTRUCTIONS = os.environ.get("MCP_INSTRUCTIONS", DEFAULT_INSTRUCTIONS)

SCHEMA_PATH = Path(os.environ.get("GRAPHQL_SCHEMA_PATH", str(DEFAULT_SCHEMA_PATH)))
ENDPOINT_URL = os.environ.get("GRAPHQL_ENDPOINT_URL")
DATA_DIR = Path(os.environ.get("GRAPHQL_EMBEDDER_DATA_DIR", str(DEFAULT_DATA_DIR)))
EMBEDDER_CONFIG = load_embedder_config()
EMBED_MODEL = EMBEDDER_CONFIG.model
MAX_PATH_DEPTH = int(os.environ.get("GRAPHQL_INDEX_MAX_DEPTH", str(DEFAULT_MAX_PATH_DEPTH)))
EMBED_QUERY_TIMEOUT_S = float(os.environ.get("GRAPHQL_EMBED_QUERY_TIMEOUT_S", "20"))
INDEX_READY_TIMEOUT_S = float(os.environ.get("GRAPHQL_INDEX_READY_TIMEOUT_S", "5"))
LIST_TYPES_INCLUDE_QUERY = os.environ.get("GRAPHQL_LIST_TYPES_INCLUDE_QUERY", "false").lower() in {"1", "true", "yes"}

embedder = EmbeddingClient(config=EMBEDDER_CONFIG, model=EMBED_MODEL)
store = EmbeddingStore(data_dir=DATA_DIR, embedding_model=embedder.model)
SCHEMA_SOURCE: dict = {"kind": "file", "path": str(SCHEMA_PATH)}
SCHEMA_TEXT: str | None = None
_SCHEMA_CACHE = None
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
_REMOTE_TIMEOUT_S: float = 30.0
_INDEX_LOCK = threading.Lock()
_INDEX_READY = False
_INDEX_ERROR: str | None = None
_SCALAR_TYPES = {"String", "Int", "Float", "Boolean", "ID"}
_AGGREGATE_KEYWORDS = {"count", "total", "sum", "avg", "average", "how many", "number of"}
_AGGREGATE_FIELD_PATTERNS = {"count", "total", "sum", "avg", "aggregate"}
_LIST_QUERY_KEYWORDS = {"items", "list", "all", "show", "find", "get", "fetch", "search"}

mcp = FastMCP(APP_NAME, instructions=MCP_INSTRUCTIONS)
mcp.dependencies = ["graphql-core", "numpy", "aiohttp"]
logger = logging.getLogger(APP_NAME)


def _run_with_default_transport(
    self,
    transport: Literal["stdio", "sse", "streamable-http"] | None = None,
    mount_path: str | None = None,
):
    chosen = transport or DEFAULT_TRANSPORT
    return FastMCP.run(self, transport=chosen, mount_path=mount_path)


mcp.run = _run_with_default_transport.__get__(mcp, FastMCP)


def _run_indexing_background() -> None:
    global _INDEX_READY, _INDEX_ERROR
    try:
        if ENDPOINT_URL:
            logger.info("Startup indexing (endpoint): %s", ENDPOINT_URL)
        else:
            logger.info("Startup indexing (schema): %s", SCHEMA_PATH)
        meta = ensure_schema_indexed(force=False, allow_build=True)
        if meta.get("indexed"):
            logger.info("Startup indexing complete: %s paths.", meta.get("count", 0))
        else:
            logger.info("Index already up-to-date.")
        _INDEX_READY = True
    except Exception as exc:
        _INDEX_ERROR = str(exc)
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
    global _SCHEMA_CACHE
    _SCHEMA_CACHE = None


def configure_runtime_endpoint(
    *,
    endpoint_url: str,
    data_dir: Path,
    embed_model: str,
    schema_text: str,
    schema_source: dict,
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
    global _SCHEMA_CACHE
    _SCHEMA_CACHE = None


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
    for k, v in (headers or {}).items():
        req.add_header(k, v)

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


def _parse_signature(signature: str) -> tuple[str, str, list[tuple[str, str]], str]:
    left, _, return_type = signature.partition(" -> ")
    args: list[tuple[str, str]] = []
    if "(" in left and left.endswith(")"):
        base, args_str = left[:-1].split("(", 1)
        for part in args_str.split(", "):
            if not part:
                continue
            name, _, type_str = part.partition(": ")
            if name and type_str:
                args.append((name, type_str))
    else:
        base = left
    type_name, _, field_name = base.partition(".")
    return type_name, field_name, args, return_type


def _base_type(type_str: str) -> str:
    base = type_str.strip()
    while True:
        base = base.rstrip("!")
        if base.startswith("[") and base.endswith("]"):
            base = base[1:-1].strip()
            continue
    return base.rstrip("!")


def _base_gql_type(graphql_type):
    base = graphql_type
    while isinstance(base, (GraphQLNonNull, GraphQLList)):
        base = base.of_type
    return base


def _tokenize(text: str) -> list[str]:
    tokens = []
    current = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                token = "".join(current)
                if token:
                    tokens.append(token)
                current = []
    if current:
        token = "".join(current)
        if token:
            tokens.append(token)
    return tokens


def _token_score(tokens: list[str], *values: str) -> int:
    score = 0
    haystack = " ".join(values).lower()
    for token in tokens:
        if token and token in haystack:
            score += 1
    return score


def _is_aggregate_query(query: str) -> bool:
    """Check if the query is asking for aggregate/count operations."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in _AGGREGATE_KEYWORDS)


def _is_aggregate_field(field_name: str) -> bool:
    """Check if a field name looks like an aggregate operation."""
    field_lower = field_name.lower()
    return any(pattern in field_lower for pattern in _AGGREGATE_FIELD_PATTERNS)


def _is_connection_field(field_name: str) -> bool:
    """Check if a field name is a Connection (cursor-based pagination) field."""
    return field_name.lower().endswith("connection")


def _is_list_return(return_type: str) -> bool:
    return return_type.strip().startswith("[")


def _is_list_query(tokens: list[str]) -> bool:
    return any(token in _LIST_QUERY_KEYWORDS for token in tokens)


def _token_match_fields(type_name: str, fields_by_type: dict[str, list[dict]], tokens: list[str]) -> bool:
    for field in fields_by_type.get(type_name, []):
        if _token_score(tokens, field.get("field_name", ""), field.get("summary", "")) > 0:
            return True
    return False


def _token_match_return_type(return_type: str, fields_by_type: dict[str, list[dict]], tokens: list[str]) -> bool:
    base_type = _base_type(return_type)
    if _token_match_fields(base_type, fields_by_type, tokens):
        return True
    if base_type.endswith("Connection"):
        candidate = base_type[: -len("Connection")]
        if candidate and candidate in fields_by_type:
            return _token_match_fields(candidate, fields_by_type, tokens)
    return False

def _describe_type(graphql_type) -> str:
    if isinstance(graphql_type, GraphQLNonNull):
        return f"{_describe_type(graphql_type.of_type)}!"
    if isinstance(graphql_type, GraphQLList):
        return f"[{_describe_type(graphql_type.of_type)}]"
    return str(graphql_type)


def _load_schema():
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    if ENDPOINT_URL:
        if not SCHEMA_TEXT:
            raise RuntimeError("Endpoint mode requires schema introspection text.")
        _SCHEMA_CACHE = build_schema(SCHEMA_TEXT)
    else:
        _SCHEMA_CACHE = build_schema(SCHEMA_PATH.read_text())
    return _SCHEMA_CACHE


def _fields_by_type_from_schema(schema) -> dict[str, list[dict]]:
    fields_by_type: dict[str, list[dict]] = {}
    for type_name, gql_type in sorted(schema.type_map.items()):
        if type_name.startswith("__"):
            continue
        if not isinstance(gql_type, GraphQLObjectType):
            continue
        for field_name, field in sorted(gql_type.fields.items()):
            arg_parts = [
                f"{arg_name}: {_describe_type(arg.type)}"
                for arg_name, arg in field.args.items()
            ]
            arg_list = ", ".join(arg_parts)
            return_type = _describe_type(field.type)
            signature = (
                f"{type_name}.{field_name}({arg_list}) -> {return_type}"
                if arg_list
                else f"{type_name}.{field_name} -> {return_type}"
            )
            fields_by_type.setdefault(type_name, []).append(
                {
                    "type_name": type_name,
                    "field_name": field_name,
                    "args": [(arg_name, _describe_type(arg.type)) for arg_name, arg in field.args.items()],
                    "return_type": return_type,
                    "summary": signature,
                }
            )
    return fields_by_type


def _split_path(path: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in path:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "." and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _parse_path_segment(segment: str) -> tuple[str, list[tuple[str, str]]]:
    if "(" in segment and segment.endswith(")"):
        name, args_str = segment[:-1].split("(", 1)
        args: list[tuple[str, str]] = []
        for part in args_str.split(", "):
            if not part:
                continue
            arg_name, _, arg_type = part.partition(": ")
            if arg_name and arg_type:
                args.append((arg_name, arg_type))
        return name, args
    return segment, []


def _operation_keyword(root_name: str) -> str:
    if root_name == "Mutation":
        return "mutation"
    if root_name == "Subscription":
        return "subscription"
    return "query"


def _resolve_return_type(schema, root_name: str, segments: list[str]):
    gql_type = schema.type_map.get(root_name)
    for segment in segments:
        if not isinstance(gql_type, GraphQLObjectType):
            return None
        field_name, _ = _parse_path_segment(segment)
        field = gql_type.fields.get(field_name)
        if field is None:
            return None
        gql_type = _base_gql_type(field.type)
    return gql_type


def _format_args(args: list[tuple[str, str]]) -> str:
    if not args:
        return ""
    rendered = ", ".join(f"{name}: <{arg_type}>" for name, arg_type in args)
    return f"({rendered})"


def _render_selection_set(
    type_name: str,
    fields_by_type: dict[str, list[dict]],
    tokens: list[str],
    depth: int = 1,
    max_fields: int = 6,
) -> str | None:
    fields = list(fields_by_type.get(type_name, []))
    if not fields:
        return None

    def rank(field: dict) -> tuple[int, int, str]:
        base = _token_score(tokens, field["field_name"], field.get("summary", ""))
        if field["field_name"] in {"id", "name"}:
            base += 2
        return_type = field.get("return_type", "")
        if _base_type(return_type) in _SCALAR_TYPES:
            return (base + 1, 1, field["field_name"])
        return (base, 0, field["field_name"])

    fields.sort(key=rank, reverse=True)

    selections: list[str] = []
    for field in fields:
        if len(selections) >= max_fields:
            break
        return_type = field.get("return_type", "")
        base_type = _base_type(return_type)
        if base_type in _SCALAR_TYPES:
            selections.append(field["field_name"])
            continue
        if depth <= 0:
            continue
        nested = _render_selection_set(
            base_type,
            fields_by_type,
            tokens,
            depth=depth - 1,
            max_fields=max(2, max_fields // 2),
        )
        if nested:
            selections.append(f"{field['field_name']} {nested}")

    if not selections:
        return None
    return "{ " + " ".join(selections) + " }"


def ensure_schema_indexed(*, force: bool = False, allow_build: bool = True) -> dict:
    try:
        if not allow_build:
            if not _INDEX_LOCK.acquire(blocking=False):
                raise RuntimeError("Indexing in progress. Try again shortly.")
            try:
                if _INDEX_ERROR:
                    raise RuntimeError(_INDEX_ERROR)
                if ENDPOINT_URL:
                    if not SCHEMA_TEXT:
                        raise RuntimeError("Endpoint mode requires schema introspection text.")
                    if not store.is_ready():
                        raise RuntimeError("Indexing in progress. Try again shortly.")
                    return store.load()
                if not store.is_ready():
                    raise RuntimeError("Indexing in progress. Try again shortly.")
                return store.load()
            finally:
                _INDEX_LOCK.release()

        with _INDEX_LOCK:
            if _INDEX_ERROR:
                raise RuntimeError(_INDEX_ERROR)
            if ENDPOINT_URL:
                if not SCHEMA_TEXT:
                    raise RuntimeError("Endpoint mode requires schema introspection text.")
                meta = ensure_index_text(
                    SCHEMA_TEXT,
                    schema_source=SCHEMA_SOURCE,
                    data_dir=DATA_DIR,
                    embed_model=EMBED_MODEL,
                    max_depth=MAX_PATH_DEPTH,
                    embedder=embedder,
                    store=store,
                    force=force,
                )
                return meta
            meta = ensure_index(
                schema_path=SCHEMA_PATH,
                data_dir=DATA_DIR,
                embed_model=EMBED_MODEL,
                max_depth=MAX_PATH_DEPTH,
                embedder=embedder,
                store=store,
                force=force,
            )
            return meta
    except Exception as exc:
        raise RuntimeError(f"Schema index not available for {SCHEMA_PATH}: {exc}")


@mcp.tool()
def list_types(query: str, limit: int = 20) -> list:
    """
    Fuzzy search the schema for matching executable schema paths.
    Uses the persisted embedding index (auto-builds if missing/outdated).
    """
    if not _INDEX_READY and not store.is_ready():
        raise RuntimeError("Indexing in progress. Try again shortly.")
    logger.info("list_types start: query=%r limit=%s", query, limit)
    meta = ensure_schema_indexed(force=False, allow_build=False)
    logger.info("list_types: index loaded")
    schema = _load_schema()
    fields_by_type = _fields_by_type_from_schema(schema)
    logger.info("list_types: schema loaded")
    normalized_query = query.strip().lower()
    tokens = _tokenize(query)
    is_aggregate = _is_aggregate_query(query)
    is_list_query = _is_list_query(tokens)

    capped_limit = max(1, min(limit, 20))
    if normalized_query in {"query", "queries", "root"}:
        raw_items = meta.get("items", [])
        results = []
        for item in raw_items:
            path = item.get("path", "")
            parts = _split_path(path)
            if item.get("root_type") == "Query" and len(parts) == 2:
                results.append(
                    {
                        "root_type": item.get("root_type"),
                        "path": path,
                        "return_type": item.get("return_type"),
                        "summary": item.get("summary"),
                        "score": 1.0,
                    }
                )
            if len(results) >= capped_limit:
                break
    else:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(embedder.embed_one, query)
                query_vec = future.result(timeout=EMBED_QUERY_TIMEOUT_S)
        except concurrent.futures.TimeoutError as exc:
            raise RuntimeError(
                f"Embedding query timed out after {EMBED_QUERY_TIMEOUT_S}s. "
                "Check your embeddings endpoint or increase GRAPHQL_EMBED_QUERY_TIMEOUT_S."
            ) from exc
        logger.info("list_types: embedding ready")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(store.search, query_vec, capped_limit)
                results = future.result(timeout=EMBED_QUERY_TIMEOUT_S)
        except concurrent.futures.TimeoutError as exc:
            raise RuntimeError(
                f"Index search timed out after {EMBED_QUERY_TIMEOUT_S}s. "
                "Reduce GRAPHQL_INDEX_MAX_DEPTH or limit, or increase GRAPHQL_EMBED_QUERY_TIMEOUT_S."
            ) from exc
        logger.info("list_types: search complete (%s results)", len(results))

    def _item_context(item: dict) -> dict:
        path = item.get("path", "")
        parts = _split_path(path)
        root_type = item.get("root_type") or (parts[0] if parts else "")
        field_name, _ = _parse_path_segment(parts[-1]) if parts else ("", [])
        return_type = item.get("return_type") or item.get("summary", "").rsplit(" -> ", 1)[-1]
        is_query_type = root_type == "Query"
        is_agg_field = _is_aggregate_field(field_name)
        is_conn_field = _is_connection_field(field_name)
        is_list_field = _is_list_return(return_type)
        token_match = _token_match_return_type(return_type, fields_by_type, tokens)
        score = float(item.get("score", 0.0))

        if is_aggregate:
            heuristic = (
                (0.30 if is_query_type else 0.0)
                + (0.25 if is_agg_field else 0.0)
                + (0.10 if is_conn_field else 0.0)
            )
        else:
            heuristic = (
                (0.30 if is_query_type else 0.0)
                + (0.20 if token_match else 0.0)
                + (0.15 if is_list_query and is_conn_field else 0.0)
                + (0.05 if is_list_query and is_list_field else 0.0)
                - (0.20 if is_list_query and is_agg_field else 0.0)
            )
        combined = score + heuristic

        return {
            "path": path,
            "root_type": root_type,
            "return_type": return_type,
            "token_match": token_match,
            "combined": combined,
        }

    contexts = {id(item): _item_context(item) for item in results}
    results.sort(key=lambda item: contexts[id(item)]["combined"], reverse=True)

    if results:
        max_score = max(contexts[id(item)]["combined"] for item in results)
        score_cutoff = max_score * 0.75
        min_keep = min(3, len(results))
        filtered: list[dict] = []
        for item in results:
            ctx = contexts[id(item)]
            if (
                len(filtered) < min_keep
                or ctx["combined"] >= score_cutoff
                or ctx["token_match"]
            ):
                filtered.append(item)
            if len(filtered) >= capped_limit:
                break
        results = filtered

    if not LIST_TYPES_INCLUDE_QUERY:
        logger.info("list_types: query generation disabled")
        return [{"path": item.get("path", ""), "summary": item.get("summary", "")} for item in results]

    def _format_results() -> list[dict]:
        formatted = []
        for item in results:
            path = item.get("path", "")
            summary = item.get("summary", "")
            parts = _split_path(path)
            root_type = item.get("root_type") or (parts[0] if parts else "")
            entry = {"path": path, "summary": summary}

            if root_type in {"Query", "Mutation", "Subscription"} and len(parts) > 1:
                segments = parts[1:]
                leaf_type = _resolve_return_type(schema, root_type, segments)

                def build_selection(segment_list: list[str]) -> str:
                    name, args = _parse_path_segment(segment_list[0])
                    arg_part = _format_args(args)
                    if len(segment_list) == 1:
                        if isinstance(leaf_type, GraphQLObjectType):
                            selection = _render_selection_set(
                                leaf_type.name,
                                fields_by_type,
                                tokens,
                                depth=1,
                                max_fields=5,
                            )
                            selection_part = f" {selection}" if selection else ""
                            return f"{name}{arg_part}{selection_part}"
                        return f"{name}{arg_part}"
                    return f"{name}{arg_part} {{ {build_selection(segment_list[1:])} }}"

                selection = build_selection(segments)
                entry["query"] = f"{_operation_keyword(root_type)} {{ {selection} }}"

            # Compact output: path + summary, plus query when the path is executable from a root op.
            compact = {"path": entry["path"], "summary": entry["summary"]}
            if "query" in entry:
                compact["query"] = entry["query"]
            formatted.append(compact)
        return formatted

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_format_results)
            formatted = future.result(timeout=EMBED_QUERY_TIMEOUT_S)
    except concurrent.futures.TimeoutError as exc:
        raise RuntimeError(
            f"Result formatting timed out after {EMBED_QUERY_TIMEOUT_S}s. "
            "Reduce GRAPHQL_INDEX_MAX_DEPTH or limit."
        ) from exc
    logger.info("list_types: formatted %s results", len(formatted))
    return formatted


@mcp.tool()
def run_query(query: str) -> dict:
    """
    Validate and run a GraphQL query against the static schema.

    Note: No resolvers are provided, so fields resolve to null;
    this is mainly for validation and shape checking.
    """
    if ENDPOINT_URL:
        try:
            payload = {"query": query}
            result = _post_json(ENDPOINT_URL, payload, headers=_REMOTE_HEADERS, timeout_s=_REMOTE_TIMEOUT_S)
        except Exception as exc:
            raise RuntimeError(f"Endpoint query failed: {exc}")
        output: dict = {"valid": not bool(result.get("errors"))}
        if "errors" in result:
            output["errors"] = result["errors"]
        if "data" in result:
            output["data"] = result["data"]
        if "extensions" in result:
            output["extensions"] = result["extensions"]
        return output

    schema = build_schema(SCHEMA_PATH.read_text())
    result = graphql_sync(schema, query)
    output = {"valid": not bool(result.errors)}
    if result.errors:
        output["errors"] = [str(err) for err in result.errors]
    if result.data is not None:
        output["data"] = result.data
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the GraphQL embedder MCP server."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=DEFAULT_TRANSPORT,
        help="MCP transport to run (default: sse; override with --transport or MCP_TRANSPORT env).",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--schema",
        type=Path,
        default=SCHEMA_PATH,
        help="Path to a GraphQL schema file (SDL).",
    )
    source_group.add_argument(
        "--endpoint",
        default=ENDPOINT_URL,
        help="GraphQL endpoint URL (uses introspection for indexing and proxies queries).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory for the persisted embedding index (default: data/ next to this server).",
    )
    parser.add_argument(
        "--model",
        default=EMBED_MODEL,
        help="Embedding model to use for indexing/search queries.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Add an HTTP header for endpoint mode, like 'Authorization: Bearer ...' (repeatable).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout (seconds) for endpoint introspection/querying.",
    )
    parser.add_argument(
        "--host",
        default=mcp.settings.host,
        help="Host for SSE/HTTP transports (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=mcp.settings.port,
        help="Port for SSE/HTTP transports (default: 8000).",
    )
    parser.add_argument(
        "--log-level",
        default=mcp.settings.log_level,
        help="Log level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--mount-path",
        default=mcp.settings.mount_path,
        help="Mount path for SSE transport (default: /).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    _REMOTE_HEADERS = _load_endpoint_headers_from_env()
    _REMOTE_HEADERS.update(_parse_headers(args.header))
    _REMOTE_TIMEOUT_S = float(args.timeout)

    endpoint_url = args.endpoint or os.environ.get("GRAPHQL_ENDPOINT_URL")
    if endpoint_url is not None:
        endpoint_url = endpoint_url.strip()
    if endpoint_url:
        try:
            schema_text = _introspect_schema_sdl(
                endpoint_url, headers=_REMOTE_HEADERS, timeout_s=_REMOTE_TIMEOUT_S
            )
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
    threading.Thread(
        target=_run_indexing_background,
        name="graphql-mcp-indexer",
        daemon=True,
    ).start()
    mcp.run(transport=args.transport, mount_path=args.mount_path)
