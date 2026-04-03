#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from graphql import build_schema, graphql_sync

SCHEMA_TEXT = """
type Query {
  users(limit: Int = 10): [User!]!
  user(id: ID!): User
  reportsConnection(first: Int = 5): ReportConnection!
}

type User {
  id: ID!
  name: String!
  email: String!
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
  cursor: String!
}

type Report {
  id: ID!
  title: String!
  severity: String!
}
"""

USERS = [
    {
        "id": "u1",
        "name": "Alice",
        "email": "alice@example.com",
        "orders": [
            {"id": "o101", "total": 42.5, "status": "PAID"},
            {"id": "o102", "total": 19.0, "status": "PENDING"},
        ],
    },
    {
        "id": "u2",
        "name": "Bob",
        "email": "bob@example.com",
        "orders": [
            {"id": "o103", "total": 77.25, "status": "PAID"},
        ],
    },
]

REPORTS = [
    {"id": "r1", "title": "Fraud Spike", "severity": "HIGH"},
    {"id": "r2", "title": "Chargeback Trend", "severity": "MEDIUM"},
    {"id": "r3", "title": "Latency Alert", "severity": "LOW"},
]


def build_app_schema():
    schema = build_schema(SCHEMA_TEXT)

    query_type = schema.get_type("Query")
    assert query_type is not None

    def resolve_users(_obj, _info, limit=10):
        return USERS[: max(0, int(limit))]

    def resolve_user(_obj, _info, id):
        for user in USERS:
            if user["id"] == id:
                return user
        return None

    def resolve_reports_connection(_obj, _info, first=5):
        size = max(0, int(first))
        slice_reports = REPORTS[:size]
        return {
            "edges": [
                {"node": report, "cursor": f"cursor:{idx}"}
                for idx, report in enumerate(slice_reports)
            ],
            "totalCount": len(REPORTS),
        }

    query_type.fields["users"].resolve = resolve_users
    query_type.fields["user"].resolve = resolve_user
    query_type.fields["reportsConnection"].resolve = resolve_reports_connection
    return schema


class GraphQLHandler(BaseHTTPRequestHandler):
    schema = build_app_schema()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/healthz"}:
            self._send_json({"ok": True, "service": "graphql-test-server"})
            return
        self._send_json({"error": "Not Found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/graphql":
            self._send_json({"error": "Not Found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            self._send_json({"errors": [{"message": "Invalid JSON body"}]}, status=400)
            return

        query = payload.get("query")
        if not query:
            self._send_json({"errors": [{"message": "Missing 'query'"}]}, status=400)
            return

        result = graphql_sync(
            self.schema,
            query,
            variable_values=payload.get("variables"),
            operation_name=payload.get("operationName"),
        )

        response: dict = {}
        if result.errors:
            response["errors"] = [{"message": str(err)} for err in result.errors]
        if result.data is not None:
            response["data"] = result.data
        self._send_json(response, status=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local GraphQL test server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=4000, help="Bind port")
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), GraphQLHandler)
    print(f"GraphQL test server listening on http://{args.host}:{args.port}/graphql", flush=True)
    print("Try: curl -s http://127.0.0.1:4000/graphql -H 'Content-Type: application/json' -d '{\"query\":\"{ users { id name } }\"}'", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
