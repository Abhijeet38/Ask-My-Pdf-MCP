"""Settings loaded from environment variables.

Single source of truth for runtime configuration. Reading env vars in
exactly one place keeps the rest of the codebase free of os.getenv calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key, "true" if default else "false").lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = _env(key, "")
    return int(raw) if raw else default


@dataclass
class Settings:
    # ---- LLM ---------------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "bedrock"))
    llm_fallback: str = field(default_factory=lambda: _env("LLM_FALLBACK", ""))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    bedrock_model_id: str = field(
        default_factory=lambda: _env(
            "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
    )
    ollama_host: str = field(
        default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434")
    )
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.2:3b"))

    # ---- OpenSearch --------------------------------------------------------
    os_host: str = field(default_factory=lambda: _env("OS_HOST", "localhost"))
    os_port: int = field(default_factory=lambda: _env_int("OS_PORT", 9200))
    os_use_ssl: bool = field(default_factory=lambda: _env_bool("OS_USE_SSL", False))
    os_use_aws_auth: bool = field(default_factory=lambda: _env_bool("OS_USE_AWS_AUTH", False))
    os_user: str = field(default_factory=lambda: _env("OS_USER"))
    os_password: str = field(default_factory=lambda: _env("OS_PASSWORD"))
    os_index: str = field(default_factory=lambda: _env("OS_INDEX", "pdf_qa_chunks"))
    os_pages_index: str = field(default_factory=lambda: _env("OS_PAGES_INDEX", "pdf_qa_pages"))

    # ---- Embedding ---------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    )
    embedding_device: str = field(default_factory=lambda: _env("EMBEDDING_DEVICE", "auto"))
    embedding_dim: int = 768  # bge-base-en-v1.5

    # ---- Ingestion / retrieval --------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")).resolve())
    chunk_tokens: int = field(default_factory=lambda: _env_int("CHUNK_TOKENS", 400))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 60))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))

    def opensearch_kwargs(self) -> dict:
        """Args accepted by opensearchpy.OpenSearch()."""
        kwargs: dict = {
            "hosts": [{"host": self.os_host, "port": self.os_port}],
            "use_ssl": self.os_use_ssl,
            "verify_certs": False,
            "ssl_show_warn": False,
        }
        if self.os_use_aws_auth:
            # AWS SigV4 auth via boto3 default credential chain. Lazy-imported
            # so that local-OpenSearch users don't need boto3+requests on the path.
            import boto3
            from opensearchpy import AWSV4SignerAuth, RequestsHttpConnection

            creds = boto3.Session(region_name=self.aws_region).get_credentials()
            kwargs["http_auth"] = AWSV4SignerAuth(creds, self.aws_region, "es")
            kwargs["connection_class"] = RequestsHttpConnection
            # AWS managed domains require valid TLS; override the local-dev defaults.
            kwargs["verify_certs"] = True
            kwargs["ssl_show_warn"] = True
        elif self.os_user and self.os_password:
            kwargs["http_auth"] = (self.os_user, self.os_password)
        return kwargs


# Singleton — read once at import time. Tests can monkey-patch fields.
settings = Settings()
