"""
tachyon_client.py
Adapter between TachyonVectorStore and the function files in pipeline/tachyon/.

This file defines the contract. Copy your Tachyon pipeline functions into
pipeline/tachyon/ (auth.py, search.py, delete.py) and adjust the call sites
below to match your actual function signatures.

See TACHYON_INTEGRATION_CONTEXT.md for the expected signatures and return shapes.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class TachyonClient:
    """
    Thin wrapper around the Tachyon pipeline functions.

    Config dict keys (all sourced from vector_stores.yaml → extra):
        consumer_key, consumer_secret, api_key, usecase_id,
        apigee_url, search_url, completion_url,
        cert (tuple[str, str] — cert_path, key_path), ca_bundle
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _token(self) -> str:
        # COPILOT: import get_access_token from pipeline/tachyon/auth.py
        # and adjust kwargs to match your function signature.
        from pipeline.tachyon.auth import get_access_token  # type: ignore[import]
        return get_access_token(
            consumer_key=self._config["consumer_key"],
            consumer_secret=self._config["consumer_secret"],
            apigee_url=self._config["apigee_url"],
            cert=self._config["cert"],          # (cert_path, key_path) tuple
            ca_bundle=self._config["ca_bundle"],
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int,
        usecase_id: str,
        collection: str,
    ) -> list[dict[str, Any]]:
        """
        Submit a text query to Tachyon and return a list of result dicts.

        Each result dict must contain at least:
            chunk_id, content, score
        Optionally: source, title, section, tags, citation.

        COPILOT: import search_documents from pipeline/tachyon/search.py
        and adjust kwargs to match your function signature.
        """
        from pipeline.tachyon.search import search_documents  # type: ignore[import]
        return search_documents(
            query=query,
            top_k=top_k,
            usecase_id=usecase_id,
            collection=collection,
            token=self._token(),
            search_url=self._config["search_url"],
            api_key=self._config["api_key"],
            cert=self._config["cert"],
            ca_bundle=self._config["ca_bundle"],
        )

    # ------------------------------------------------------------------
    # Delete  (ingestion plan populates s3_file_id / vector_file_id)
    # ------------------------------------------------------------------

    def delete(self, s3_file_id: str, vector_file_id: str) -> None:
        """
        Remove an uploaded file from S3 and its corresponding vector doc.

        COPILOT: import delete_file from pipeline/tachyon/delete.py
        and adjust kwargs to match your function signature.
        """
        from pipeline.tachyon.delete import delete_file  # type: ignore[import]
        delete_file(
            s3_file_id=s3_file_id,
            vector_file_id=vector_file_id,
            token=self._token(),
            api_key=self._config["api_key"],
            cert=self._config["cert"],
            ca_bundle=self._config["ca_bundle"],
        )
