"""
config.py
Loads all settings from the environment / .env file so every other module
can import a single `settings` instance.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=False)


class Settings:
    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_provider: Literal["openai", "azure", "sentence-transformers"] = (
        os.getenv("EMBEDDING_PROVIDER", "openai")
    )
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

    # ── OpenAI ────────────────────────────────────────────────────────────
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # ── Azure OpenAI ──────────────────────────────────────────────────────
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_index_name: str = os.getenv("REDIS_INDEX_NAME", "knowledge_index")
    redis_key_prefix: str = os.getenv("REDIS_KEY_PREFIX", "doc:")

    # ── Pipeline ──────────────────────────────────────────────────────────
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "2000"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))

    # ── JSONL Export ──────────────────────────────────────────────────────
    jsonl_output_dir: Path = Path(os.getenv("JSONL_OUTPUT_DIR", "./output"))

    # ── Quality Assessment ────────────────────────────────────────────────
    quality_threshold: float = float(os.getenv("QUALITY_THRESHOLD", "0.6"))

    # ── Vector Backend ────────────────────────────────────────────────────
    # "redis" uses existing RediSearch index; "qdrant" pushes to Qdrant
    vector_backend: Literal["redis", "qdrant"] = os.getenv("VECTOR_BACKEND", "redis")  # type: ignore[assignment]

    # ── Qdrant ────────────────────────────────────────────────────────────
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "knowledge_base")

    # ── Docling Chunker ───────────────────────────────────────────────────
    # Max tokens per chunk when using HybridChunker
    docling_max_tokens: int = int(os.getenv("DOCLING_MAX_TOKENS", "512"))

    # ── MongoDB (staging store + KB ledger) ───────────────────────────────
    # Set MONGODB_URI to use a full connection string (e.g. mongodb+srv://...).
    # When set, the individual host/port/username/password/tls fields are ignored.
    mongodb_uri: str         = os.getenv("MONGODB_URI", "")
    mongodb_host: str        = os.getenv("MONGODB_HOST", "localhost")
    mongodb_port: int        = int(os.getenv("MONGODB_PORT", "27017"))
    mongodb_username: str    = os.getenv("MONGODB_USERNAME", "")
    mongodb_password: str    = os.getenv("MONGODB_PASSWORD", "")
    mongodb_auth_source: str = os.getenv("MONGODB_AUTH_SOURCE", "")
    mongodb_tls: bool        = os.getenv("MONGODB_TLS", "true").lower() not in ("0", "false", "no")
    mongodb_db_name: str     = os.getenv("MONGODB_DB_NAME", "knowledge_pipeline")
    # Prefix applied to every collection name — useful when sharing one DB
    # across multiple environments (e.g. "prod_" → prod_staging_docs).
    mongodb_collection_prefix: str = os.getenv("MONGODB_COLLECTION_PREFIX", "")

    def validate(self) -> None:
        """Raise ValueError for obviously missing required settings."""
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or set it as an environment variable."
            )
        if self.embedding_provider == "azure":
            missing = [
                k
                for k, v in {
                    "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
                    "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
                    "AZURE_OPENAI_DEPLOYMENT": self.azure_openai_deployment,
                }.items()
                if not v
            ]
            if missing:
                raise ValueError(f"Missing Azure OpenAI settings: {missing}")


settings = Settings()
