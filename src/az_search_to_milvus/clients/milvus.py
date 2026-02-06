"""Milvus / Zilliz client wrapper for collection management and data insertion."""

from __future__ import annotations

import logging
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    MilvusClient,
    connections,
    utility,
)

from az_search_to_milvus.config import MilvusConfig
from az_search_to_milvus.index_mapping import MilvusIndexConfig

logger = logging.getLogger("az_search_to_milvus.clients.milvus")


class MilvusClientWrapper:
    """High-level wrapper around the Milvus Python SDK.

    Supports both self-hosted Milvus and Zilliz Cloud.
    """

    def __init__(self, config: MilvusConfig) -> None:
        self.config = config
        self._client: MilvusClient | None = None
        self._connected = False

    def connect(self) -> None:
        """Establish connection to Milvus / Zilliz."""
        uri = self.config.effective_uri
        token = self.config.effective_token

        logger.info("Milvus に接続中: %s (db=%s)", uri, self.config.db_name)

        connect_kwargs: dict[str, Any] = {
            "uri": uri,
            "db_name": self.config.db_name,
        }
        if token:
            connect_kwargs["token"] = token

        self._client = MilvusClient(**connect_kwargs)
        self._connected = True
        logger.info("Milvus 接続成功")

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._connected = False
            logger.info("Milvus 切断")

    @property
    def client(self) -> MilvusClient:
        if not self._client:
            raise RuntimeError("Milvus に接続されていません。connect() を先に呼んでください")
        return self._client

    def collection_exists(self, name: str) -> bool:
        return self.client.has_collection(name)

    def drop_collection(self, name: str) -> None:
        if self.collection_exists(name):
            logger.warning("コレクション '%s' を削除中...", name)
            self.client.drop_collection(name)

    def create_collection(
        self,
        name: str,
        schema: CollectionSchema,
        *,
        drop_existing: bool = False,
    ) -> None:
        """Create a Milvus collection from a ``CollectionSchema``.

        Parameters
        ----------
        name:
            Collection name.
        schema:
            The ``pymilvus.CollectionSchema`` to use.
        drop_existing:
            If ``True``, drop the existing collection first.
        """
        if drop_existing:
            self.drop_collection(name)

        if self.collection_exists(name):
            logger.info("コレクション '%s' は既に存在します。スキップ", name)
            return

        logger.info("コレクション '%s' を作成中...", name)
        self.client.create_collection(
            collection_name=name,
            schema=schema,
        )
        logger.info("コレクション '%s' 作成完了", name)

    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_config: MilvusIndexConfig,
    ) -> None:
        """Create a vector index on a field."""
        logger.info(
            "インデックス作成: collection=%s, field=%s, type=%s, metric=%s",
            collection_name,
            field_name,
            index_config.index_type,
            index_config.metric_type,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=field_name,
            index_type=index_config.index_type,
            metric_type=index_config.metric_type,
            params=index_config.params,
        )
        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params,
        )
        logger.info("インデックス作成完了: %s.%s", collection_name, field_name)

    def insert_batch(
        self,
        collection_name: str,
        data: list[dict[str, Any]],
    ) -> int:
        """Insert a batch of documents.

        Returns the number of successfully inserted documents.
        """
        if not data:
            return 0

        result = self.client.insert(
            collection_name=collection_name,
            data=data,
        )
        count = result.get("insert_count", len(data))
        return count

    def load_collection(self, collection_name: str) -> None:
        """Load collection into memory for searching."""
        logger.info("コレクション '%s' をメモリにロード中...", collection_name)
        self.client.load_collection(collection_name)
        logger.info("コレクション '%s' ロード完了", collection_name)

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Return basic statistics for a collection."""
        stats = self.client.get_collection_stats(collection_name)
        return stats

    def query_count(self, collection_name: str) -> int:
        """Return the number of entities in a collection."""
        self.client.load_collection(collection_name)
        results = self.client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        if results:
            return results[0].get("count(*)", 0)
        return 0

    def sample_query(
        self,
        collection_name: str,
        *,
        limit: int = 5,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve a sample of documents for validation."""
        self.client.load_collection(collection_name)
        results = self.client.query(
            collection_name=collection_name,
            filter="",
            limit=limit,
            output_fields=output_fields or ["*"],
        )
        return results
