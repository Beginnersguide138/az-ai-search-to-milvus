"""Azure AI Search client wrapper for schema extraction and data export."""

from __future__ import annotations

import logging
from typing import Any, Generator

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex

from az_search_to_milvus.config import AzureSearchConfig

logger = logging.getLogger("az_search_to_milvus.clients.ai_search")


class AzureSearchClientWrapper:
    """High-level wrapper around the Azure AI Search SDK.

    Provides methods for:
    - Listing indexes
    - Retrieving index schemas
    - Extracting all documents in batches
    """

    def __init__(self, config: AzureSearchConfig) -> None:
        self.config = config
        self._credential = self._build_credential()
        self._index_client = SearchIndexClient(
            endpoint=config.endpoint,
            credential=self._credential,
        )

    def _build_credential(self) -> Any:
        if self.config.use_entra_id:
            from azure.identity import DefaultAzureCredential
            return DefaultAzureCredential()
        return AzureKeyCredential(self.config.api_key)

    def list_indexes(self) -> list[str]:
        """Return a list of all index names in the search service."""
        return [idx.name for idx in self._index_client.list_indexes()]

    def get_index(self, index_name: str | None = None) -> SearchIndex:
        """Retrieve the full index definition (schema).

        Parameters
        ----------
        index_name:
            Name of the index.  Defaults to ``config.index_name``.
        """
        name = index_name or self.config.index_name
        logger.info("インデックス '%s' のスキーマを取得中...", name)
        return self._index_client.get_index(name)

    def get_document_count(self, index_name: str | None = None) -> int:
        """Return the approximate document count for the index."""
        name = index_name or self.config.index_name
        search_client = SearchClient(
            endpoint=self.config.endpoint,
            index_name=name,
            credential=self._credential,
        )
        results = search_client.search(search_text="*", include_total_count=True, top=0)
        return results.get_count() or 0

    def extract_documents(
        self,
        index_name: str | None = None,
        *,
        batch_size: int = 1000,
        select: list[str] | None = None,
        skip_count: int = 0,
    ) -> Generator[list[dict[str, Any]], None, None]:
        """Yield batches of documents from the index.

        Uses ``search(*)`` with ordering by the key field and pagination.
        The Azure SDK's paged iterator handles continuation tokens internally.

        Parameters
        ----------
        index_name:
            Target index.
        batch_size:
            Number of documents per batch (max 1000 per Azure API limits).
        select:
            Fields to include.  ``None`` means all fields.
        skip_count:
            Number of documents to skip (for checkpoint resume).
        """
        name = index_name or self.config.index_name
        search_client = SearchClient(
            endpoint=self.config.endpoint,
            index_name=name,
            credential=self._credential,
        )

        # Determine the key field for ordering
        index_def = self.get_index(name)
        key_field = next(
            (f.name for f in index_def.fields if getattr(f, "key", False)),
            None,
        )
        order_by = f"{key_field} asc" if key_field else None

        logger.info(
            "ドキュメント抽出開始: index=%s, batch_size=%d, skip=%d",
            name, batch_size, skip_count,
        )

        effective_batch = min(batch_size, 1000)
        results = search_client.search(
            search_text="*",
            select=select or ["*"],
            include_total_count=True,
            top=effective_batch,
            skip=skip_count if skip_count < 100_000 else 0,
            order_by=order_by,
        )

        batch: list[dict[str, Any]] = []
        doc_count = 0

        for doc in results:
            # Convert to plain dict and remove Azure metadata
            record = {k: v for k, v in doc.items() if not k.startswith("@")}
            batch.append(record)
            doc_count += 1

            if len(batch) >= effective_batch:
                logger.debug("バッチ出力: %d ドキュメント (合計 %d)", len(batch), doc_count)
                yield batch
                batch = []

        if batch:
            logger.debug("最終バッチ出力: %d ドキュメント (合計 %d)", len(batch), doc_count)
            yield batch

        logger.info("ドキュメント抽出完了: 合計 %d ドキュメント", doc_count)

    def extract_all_documents(
        self,
        index_name: str | None = None,
        *,
        select: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract all documents as a flat list (convenience for small indexes)."""
        docs: list[dict[str, Any]] = []
        for batch in self.extract_documents(index_name, select=select):
            docs.extend(batch)
        return docs
