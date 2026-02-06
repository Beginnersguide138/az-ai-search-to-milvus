"""移行ツールの設定モデル。

YAML ファイルおよび環境変数からの読み込みをサポートします。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AzureSearchConfig:
    """Azure AI Search の接続設定。"""

    endpoint: str = ""
    index_name: str = ""
    api_key: str = ""
    use_entra_id: bool = False
    api_version: str = "2024-07-01"

    def resolve(self) -> None:
        """明示的に設定されていない値を環境変数から解決する。"""
        self.endpoint = self.endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT", "")
        self.api_key = self.api_key or os.environ.get("AZURE_SEARCH_API_KEY", "")
        self.index_name = self.index_name or os.environ.get("AZURE_SEARCH_INDEX_NAME", "")


@dataclass
class MilvusConfig:
    """Milvus / Zilliz の接続設定。"""

    uri: str = "http://localhost:19530"
    token: str = ""
    db_name: str = "default"
    collection_name: str = ""
    # Zilliz Cloud 設定
    use_zilliz: bool = False
    zilliz_api_key: str = ""
    zilliz_endpoint: str = ""

    def resolve(self) -> None:
        """明示的に設定されていない値を環境変数から解決する。"""
        self.uri = self.uri or os.environ.get("MILVUS_URI", "http://localhost:19530")
        self.token = self.token or os.environ.get("MILVUS_TOKEN", "")
        self.zilliz_api_key = self.zilliz_api_key or os.environ.get("ZILLIZ_API_KEY", "")
        self.zilliz_endpoint = self.zilliz_endpoint or os.environ.get("ZILLIZ_ENDPOINT", "")

    @property
    def effective_uri(self) -> str:
        if self.use_zilliz and self.zilliz_endpoint:
            return self.zilliz_endpoint
        return self.uri

    @property
    def effective_token(self) -> str:
        if self.use_zilliz and self.zilliz_api_key:
            return self.zilliz_api_key
        return self.token


@dataclass
class MigrationOptions:
    """移行の動作を制御するオプション。"""

    batch_size: int = 500
    max_workers: int = 4
    checkpoint_dir: str = ".checkpoints"
    drop_existing_collection: bool = False
    dry_run: bool = False
    # フィールドレベルのオーバーライド: {"azure_field_name": {"milvus_name": "...", "max_length": 1024}}
    field_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 移行から除外するフィールド
    exclude_fields: list[str] = field(default_factory=list)
    # Milvus 固有の拡張機能
    enable_dynamic_field: bool = True
    partition_key_field: str = ""
    varchar_max_length: int = 65_535
    array_max_capacity: int = 4096


@dataclass
class MigrationConfig:
    """トップレベルの移行設定。"""

    azure_search: AzureSearchConfig = field(default_factory=AzureSearchConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    options: MigrationOptions = field(default_factory=MigrationOptions)

    def resolve(self) -> None:
        self.azure_search.resolve()
        self.milvus.resolve()

    @classmethod
    def from_yaml(cls, path: str | Path) -> MigrationConfig:
        """YAML ファイルから設定を読み込む。"""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        config = cls()

        # Azure Search 設定
        az = raw.get("azure_search", {})
        config.azure_search = AzureSearchConfig(
            endpoint=az.get("endpoint", ""),
            index_name=az.get("index_name", ""),
            api_key=az.get("api_key", ""),
            use_entra_id=az.get("use_entra_id", False),
            api_version=az.get("api_version", "2024-07-01"),
        )

        # Milvus 設定
        mv = raw.get("milvus", {})
        config.milvus = MilvusConfig(
            uri=mv.get("uri", "http://localhost:19530"),
            token=mv.get("token", ""),
            db_name=mv.get("db_name", "default"),
            collection_name=mv.get("collection_name", ""),
            use_zilliz=mv.get("use_zilliz", False),
            zilliz_api_key=mv.get("zilliz_api_key", ""),
            zilliz_endpoint=mv.get("zilliz_endpoint", ""),
        )

        # オプション設定
        opts = raw.get("options", {})
        config.options = MigrationOptions(
            batch_size=opts.get("batch_size", 500),
            max_workers=opts.get("max_workers", 4),
            checkpoint_dir=opts.get("checkpoint_dir", ".checkpoints"),
            drop_existing_collection=opts.get("drop_existing_collection", False),
            dry_run=opts.get("dry_run", False),
            field_overrides=opts.get("field_overrides", {}),
            exclude_fields=opts.get("exclude_fields", []),
            enable_dynamic_field=opts.get("enable_dynamic_field", True),
            partition_key_field=opts.get("partition_key_field", ""),
            varchar_max_length=opts.get("varchar_max_length", 65_535),
            array_max_capacity=opts.get("array_max_capacity", 4096),
        )

        config.resolve()
        return config

    def to_dict(self) -> dict[str, Any]:
        """辞書にシリアライズする (YAML 出力 / ロギング用)。"""
        from dataclasses import asdict

        d = asdict(self)
        # シークレットをマスク
        if d["azure_search"]["api_key"]:
            d["azure_search"]["api_key"] = "***"
        if d["milvus"]["token"]:
            d["milvus"]["token"] = "***"
        if d["milvus"]["zilliz_api_key"]:
            d["milvus"]["zilliz_api_key"] = "***"
        return d
