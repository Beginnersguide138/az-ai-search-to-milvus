"""Azure AI Search クライアントラッパー。スキーマ取得とデータエクスポート用。"""

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
    """Azure AI Search SDK の高レベルラッパー。

    提供するメソッド:
    - インデックス一覧の取得
    - インデックススキーマの取得
    - 全ドキュメントのバッチ抽出
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
        """検索サービス内の全インデックス名のリストを返す。"""
        return [idx.name for idx in self._index_client.list_indexes()]

    def get_index(self, index_name: str | None = None) -> SearchIndex:
        """インデックス定義（スキーマ）の完全な情報を取得する。

        パラメータ
        ----------
        index_name:
            インデックス名。デフォルトは ``config.index_name``。
        """
        name = index_name or self.config.index_name
        logger.info("インデックス '%s' のスキーマを取得中...", name)
        return self._index_client.get_index(name)

    def get_document_count(self, index_name: str | None = None) -> int:
        """インデックスの概算ドキュメント数を返す。"""
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
        """インデックスからドキュメントをバッチごとに yield する。

        ``search(*)`` を使用し、キーフィールドでの並び替えとページネーションを行う。
        Azure SDK のページイテレータが継続トークンを内部的に処理する。

        パラメータ
        ----------
        index_name:
            対象インデックス。
        batch_size:
            バッチあたりのドキュメント数（Azure API の上限により最大 1000）。
        select:
            含めるフィールド。``None`` の場合は全フィールド。
        skip_count:
            スキップするドキュメント数（チェックポイント再開用）。
        """
        name = index_name or self.config.index_name
        search_client = SearchClient(
            endpoint=self.config.endpoint,
            index_name=name,
            credential=self._credential,
        )

        # 並び替え用のキーフィールドを特定する
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
            # プレーンな dict に変換し、Azure メタデータを除去する
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
        """全ドキュメントをフラットなリストとして抽出する（小規模インデックス向けの簡易メソッド）。"""
        docs: list[dict[str, Any]] = []
        for batch in self.extract_documents(index_name, select=select):
            docs.extend(batch)
        return docs
