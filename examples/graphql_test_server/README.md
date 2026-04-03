# Local GraphQL Test Server

Tiny in-repo GraphQL server for testing `src/server.py` endpoint mode.

## Run

```bash
python3 examples/graphql_test_server/server.py
```

Defaults:
- host: `127.0.0.1`
- port: `4000`
- endpoint: `http://127.0.0.1:4000/graphql`

You can override host/port:

```bash
python3 examples/graphql_test_server/server.py --host 127.0.0.1 --port 4000
```

## Quick check

```bash
curl -s http://127.0.0.1:4000/graphql \
  -H 'Content-Type: application/json' \
  -d '{"query":"{ users { id name orders { id total status } } }"}'
```

## Use with GraphQL MCP server

In another terminal:

```bash
python3 src/server.py --transport sse --endpoint http://127.0.0.1:4000/graphql
```

If `GRAPHQL_ENDPOINT_URL` is already set in `.env`, you can also run:

```bash
python3 src/server.py --transport sse
```
