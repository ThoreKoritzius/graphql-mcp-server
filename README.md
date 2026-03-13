# GraphQL schema embedder MCP server

Python MCP server for LLMs that indexes a GraphQL schema, stores embeddings per `type->field` via an embeddings endpoint, and enables fast lookup plus `run_query` execution once relevant types are identified to fetch data from your GraphQL endpoint.

## Architecture
- GraphQL schema: provide a schema file (SDL) to exercise parsing and indexing.
- Indexer: `schema_indexer.py` builds a navigation index of GraphQL field nodes, including field metadata, fuzzy-search aliases, and Query-root coordinates, then embeds the generated search text and persists to `data/metadata.json` + `data/vectors.npz`.
- Server: `server.py` exposes MCP tools `list_types` and `run_query`. The server ensures the schema index exists on startup; it only calls the embeddings endpoint when reindexing or embedding a new query.
- Persistence: `data/` is `.gitignore`'d so you can regenerate locally without polluting the repo.

![Architecture diagram](docs/architecture.svg)

## Setup
Set env vars. You can start from `.env.example`.

Environment configuration:
- `GRAPHQL_EMBED_API_KEY` (or `OPENAI_API_KEY`)
- `GRAPHQL_EMBEDDINGS_URL` (full embeddings URL)
- `GRAPHQL_EMBED_MODEL`
- `GRAPHQL_EMBED_API_KEY_HEADER` / `GRAPHQL_EMBED_API_KEY_PREFIX`
- `GRAPHQL_EMBED_HEADERS` (JSON object string for extra headers)
Endpoint auth (when using `GRAPHQL_ENDPOINT_URL`):
- `GRAPHQL_ENDPOINT_HEADERS` (JSON object string, merged with any `--header` flags)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/server.py
```

## Run the MCP server
```bash
python3 src/server.py                # SSE on 127.0.0.1:8000/sse by default
python3 src/server.py --transport sse     # explicit SSE
python3 src/server.py --transport streamable-http  # Streamable HTTP on 127.0.0.1:8000/mcp
# Or: point at a live GraphQL endpoint (requires introspection enabled)
python3 src/server.py --endpoint https://api.example.com/graphql
# Endpoint auth headers (repeat --header)
python3 src/src/server.py --endpoint https://api.example.com/graphql --header "Authorization: Bearer $TOKEN"
# Options: --host 0.0.0.0 --port 9000 --log-level DEBUG --mount-path /myapp
```
Tools:
- `list_types(query, limit=5)` – hybrid fuzzy search over GraphQL field nodes. Matches can come from `type`, `field`, `type.field`, partial misspellings, return types, and Query-root coordinates. Results include `coordinates` for how to reach a field from `Query`, plus `query` for `Query` fields and `select` for nested object fields.
- `run_query(query)` – if `--endpoint` is set, proxies the query to the endpoint; otherwise validates/runs against the local schema (no resolvers; primarily for validation/shape checking, data resolves to null).
Both indexing and querying use the same embedding model (`text-embedding-3-small` by default, override via config/env or `--model`).

Ranking (list_types):
- Hybrid scoring combines lexical fuzzy similarity and embedding similarity across the full indexed field set.
- Lexical matching is weighted higher than semantic similarity so typo-tolerant field lookup still works when embeddings are weak.
- Reachable fields and direct `Query` roots receive small secondary boosts.

Example `list_types` output:
```json
[
  {
    "field": "users",
    "summary": "Query.users(limit: Int) -> [User!]!",
    "coordinates": "Query.users(limit: <Int>)",
    "query": "query { users(limit: <Int>) { id name orders { id total status } } }"
  },
  {
    "type": "Order",
    "field": "total",
    "summary": "Order.total -> Float!",
    "coordinates": "Query.user(id: <ID!>) -> User.orders -> Order.total"
  },
  {
    "type": "User",
    "field": "orders",
    "summary": "User.orders -> [Order!]!",
    "coordinates": "Query.user(id: <ID!>) -> User.orders",
    "select": "orders { id total status }"
  }
]
```

Notes:
- `python3 src/server.py` defaults to the `sse` transport; pass `--transport streamable-http` if you want HTTP instead.
- You can also set env vars prefixed with `FASTMCP_` (e.g., `FASTMCP_HOST`, `FASTMCP_PORT`, `FASTMCP_LOG_LEVEL`) to override defaults.
- The server ensures the schema index is built on startup; if embeddings are computed, a simple progress bar is printed. Set `GRAPHQL_EMBED_BATCH_SIZE` to tune the batch size.
- The server exposes MCP `instructions` (override with `MCP_INSTRUCTIONS`) that describe the server as an abstraction layer and tell the LLM to use `list_types` then `run_query` with minimal tool calls.

## Quick test with the MCP Inspector
Requires `npm`/`npx` on PATH.

### Connect to an already-running SSE server
In one terminal (start the server):
```bash
python3 src/server.py --transport sse --port 8000
```
In another terminal (start the Inspector and point it at `/sse`):
```bash
npx @modelcontextprotocol/inspector --transport sse --server-url http://127.0.0.1:8000/sse
```

## Configure in Claude Desktop / CLI
If you're running this server locally over SSE (default), point Claude at the `/sse` URL.

```bash
claude mcp add --transport sse graphql-mcp http://127.0.0.1:8000/sse
```

You can also configure via JSON (e.g. config file):
```json
{
  "mcpServers": {
    "graphql-mcp": {
      "type": "sse",
      "url": "http://127.0.0.1:8000/sse"
    }
  }
}
```

If you expose this server behind auth, pass headers:
```bash
claude mcp add --transport sse private-graphql http://127.0.0.1:8000/sse \
  --header "Authorization: Bearer your-token-here"
```
