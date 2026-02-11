"""Milvus / Zilliz クライアントラッパー。コレクション管理とデータ挿入用。"""

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
    """Milvus Python SDK の高レベルラッパー。

    セルフホスト Milvus と Zilliz Cloud の両方に対応。
    """

    def __init__(self, config: MilvusConfig) -> None:
        self.config = config
        self._client: MilvusClient | None = None
        self._connected = False

    def connect(self) -> None:
        """Milvus / Zilliz への接続を確立する。"""
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
        """``CollectionSchema`` から Milvus コレクションを作成する。

        パラメータ
        ----------
        name:
            コレクション名。
        schema:
            使用する ``pymilvus.CollectionSchema``。
        drop_existing:
            ``True`` の場合、既存のコレクションを先に削除する。
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
        """フィールドにベクトルインデックスを作成する。"""
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
        """ドキュメントのバッチを挿入する。

        正常に挿入されたドキュメント数を返す。
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
        """検索用にコレクションをメモリにロードする。"""
        logger.info("コレクション '%s' をメモリにロード中...", collection_name)
        self.client.load_collection(collection_name)
        logger.info("コレクション '%s' ロード完了", collection_name)

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """コレクションの基本統計情報を返す。"""
        stats = self.client.get_collection_stats(collection_name)
        return stats

    def query_count(self, collection_name: str) -> int:
        """コレクション内のエンティティ数を返す。"""
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
        """検証用にドキュメントのサンプルを取得する。"""
        self.client.load_collection(collection_name)
        results = self.client.query(
            collection_name=collection_name,
            filter="",
            limit=limit,
            output_fields=output_fields or ["*"],
        )
        return results
