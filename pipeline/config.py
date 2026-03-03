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
