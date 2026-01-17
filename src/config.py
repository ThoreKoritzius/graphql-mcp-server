from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_EMBED_TIMEOUT_S = 30.0
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATHS = [Path.cwd() / ".env", _REPO_ROOT / ".env"]
for _path in _ENV_PATHS:
    if _path.exists():
        load_dotenv(_path, override=True)


@dataclass(frozen=True)
class EmbedderConfig:
    embeddings_url: str | None
    api_key: str | None
    api_key_header: str
    api_key_prefix: str
    model: str
    timeout_s: float
    headers: dict[str, str]

    def resolved_headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        if self.api_key and self.api_key_header not in headers:
            headers[self.api_key_header] = f"{self.api_key_prefix}{self.api_key}"
        return headers


def _coerce_headers(value: object) -> dict[str, str]:
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError("GRAPHQL_EMBED_HEADERS must be a JSON object")
    return {str(key): str(val) for key, val in value.items()}


def _load_env_headers() -> dict[str, str]:
    raw_headers = os.environ.get("GRAPHQL_EMBED_HEADERS")
    if not raw_headers:
        return {}
    try:
        parsed = json.loads(raw_headers)
    except json.JSONDecodeError as exc:
        raise ValueError("GRAPHQL_EMBED_HEADERS must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("GRAPHQL_EMBED_HEADERS must be a JSON object")
    return {str(key): str(val) for key, val in parsed.items()}


def load_embedder_config() -> EmbedderConfig:
    env_headers = _load_env_headers()
    headers = dict(_coerce_headers(env_headers))

    embeddings_url = os.environ.get("GRAPHQL_EMBEDDINGS_URL") or DEFAULT_EMBEDDINGS_URL
    api_key = os.environ.get("GRAPHQL_EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_key_header = os.environ.get("GRAPHQL_EMBED_API_KEY_HEADER") or "Authorization"
    api_key_prefix = os.environ.get("GRAPHQL_EMBED_API_KEY_PREFIX")
    if api_key_prefix is None:
        api_key_prefix = "Bearer "
    model = os.environ.get("GRAPHQL_EMBED_MODEL") or DEFAULT_EMBED_MODEL

    timeout_value = os.environ.get("GRAPHQL_EMBED_TIMEOUT_S") or DEFAULT_EMBED_TIMEOUT_S
    try:
        timeout_s = float(timeout_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid GRAPHQL_EMBED_TIMEOUT_S value") from exc

    return EmbedderConfig(
        embeddings_url=embeddings_url,
        api_key=api_key,
        api_key_header=api_key_header,
        api_key_prefix=api_key_prefix,
        model=model,
        timeout_s=timeout_s,
        headers=headers,
    )
