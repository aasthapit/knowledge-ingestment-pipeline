"""YAML-backed vector store configuration.

Reads vector_stores.yaml from the project root and expands ${ENV_VAR} references.
Provides the same .get() / .list_all() interface as the old MongoDB-backed store
so all callers (corpus push, search, flush) work without changes.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "vector_stores.yaml"


def _expand_env(value: Any) -> Any:
    """Recursively replace ${VAR} with os.environ values in strings."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{([^}]+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


class YamlVectorStoreConfig:
    """
    Read-only vector store config backed by vector_stores.yaml.

    Each entry in the YAML must have at minimum:
        id:   unique identifier (must match vector_store_id stored in corpora)
        name: display name
        type: redis | custom | tachyon

    String values may reference environment variables as ${VAR_NAME}.
    """

    def __init__(self, path: Path = _DEFAULT_CONFIG_PATH) -> None:
        self._path = path
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path) as fh:
            data = yaml.safe_load(fh) or {}
        for raw in data.get("vector_stores", []):
            entry: dict[str, Any] = _expand_env(dict(raw))
            vs_id = entry.get("id")
            if not vs_id:
                continue
            entry["vs_id"] = vs_id
            entry.setdefault("extra", {})
            entry.setdefault("endpoint", "")
            entry.setdefault("api_key", "")
            entry.setdefault("collection", "")
            self._entries[vs_id] = entry

    def get(self, vs_id: str) -> dict[str, Any] | None:
        entry = self._entries.get(vs_id)
        return dict(entry) if entry else None

    def list_all(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self._entries.values()]

    @property
    def config_path(self) -> Path:
        return self._path


_instance: YamlVectorStoreConfig | None = None


def get_vs_config_store() -> YamlVectorStoreConfig:
    """Return a cached YamlVectorStoreConfig instance."""
    global _instance
    if _instance is None:
        _instance = YamlVectorStoreConfig()
    return _instance


def reload_vs_config() -> YamlVectorStoreConfig:
    """Force reload from disk and return the new instance."""
    global _instance
    _instance = YamlVectorStoreConfig()
    return _instance
