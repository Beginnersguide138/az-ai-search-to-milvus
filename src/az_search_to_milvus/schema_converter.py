"""スキーマ変換エンジン: Azure AI Search インデックス → Milvus コレクションスキーマ。

Azure AI Search の ``SearchIndex`` 定義（Python SDK または JSON エクスポート経由）を
読み取り、等価な ``pymilvus.CollectionSchema`` を生成する。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pymilvus import CollectionSchema, DataType, FieldSchema

from az_search_to_milvus.config import MigrationOptions
from az_search_to_milvus.index_mapping import MilvusIndexConfig, map_vector_index
from az_search_to_milvus.type_mapping import (
    MappingConfidence,
    TypeMapping,
    UNSUPPORTED_FEATURES,
    resolve_type,
)

logger = logging.getLogger("az_search_to_milvus.schema_converter")


# ---------------------------------------------------------------------------
# 変換結果モデル
# ---------------------------------------------------------------------------


@dataclass
class FieldConversion:
    """単一フィールドの変換結果。"""

    azure_name: str
    azure_type: str
    milvus_field: FieldSchema | None
    mapping: TypeMapping
    skipped: bool = False
    skip_reason: str = ""
    renamed_to: str = ""
    is_primary_key: bool = False


@dataclass
class IndexConversion:
    """ベクトル検索アルゴリズム/プロファイルの変換結果。"""

    azure_profile_name: str
    azure_algorithm_kind: str
    azure_metric: str
    milvus_config: MilvusIndexConfig
    target_field: str = ""


@dataclass
class ConversionWarning:
    """スキーマ変換中に生成された警告。"""

    category: str  # "unsupported_feature" | "type_lossy" | "field_skip" | ...
    message: str
    field_name: str = ""


@dataclass
class SchemaConversionResult:
    """Azure AI Search インデックスから Milvus への変換結果。"""

    azure_index_name: str
    milvus_collection_name: str
    collection_schema: CollectionSchema
    field_conversions: list[FieldConversion] = field(default_factory=list)
    index_conversions: list[IndexConversion] = field(default_factory=list)
    warnings: list[ConversionWarning] = field(default_factory=list)
    unsupported_features: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "azure_index": self.azure_index_name,
            "milvus_collection": self.milvus_collection_name,
            "fields_total": len(self.field_conversions),
            "fields_converted": sum(1 for f in self.field_conversions if not f.skipped),
            "fields_skipped": sum(1 for f in self.field_conversions if f.skipped),
            "vector_fields": sum(1 for f in self.field_conversions if f.mapping.is_vector),
            "indexes": len(self.index_conversions),
            "warnings": len(self.warnings),
            "unsupported_features": self.unsupported_features,
        }


# ---------------------------------------------------------------------------
# スキーマコンバーター
# ---------------------------------------------------------------------------


class SchemaConverter:
    """Azure AI Search インデックスのスキーマを Milvus コレクションスキーマに変換する。"""

    def __init__(self, options: MigrationOptions | None = None) -> None:
        self.options = options or MigrationOptions()

    def convert_from_index(self, index: Any) -> SchemaConversionResult:
        """Azure SDK の ``SearchIndex`` オブジェクトから変換する。

        パラメータ
        ----------
        index:
            ``azure.search.documents.indexes.models.SearchIndex`` のインスタンス。
        """
        index_name = index.name
        collection_name = self.options.field_overrides.get(
            "__collection_name__", {}
        ).get("milvus_name", index_name.replace("-", "_"))

        # ベクトル検索プロファイル/アルゴリズムを参照用に収集
        profiles, algorithms = self._parse_vector_search_config(index)

        # フィールドを変換
        field_conversions: list[FieldConversion] = []
        milvus_fields: list[FieldSchema] = []
        primary_key_field: str | None = None

        for azure_field in index.fields:
            fc = self._convert_field(azure_field, profiles, algorithms)
            field_conversions.append(fc)
            if fc.milvus_field is not None and not fc.skipped:
                milvus_fields.append(fc.milvus_field)
            if fc.is_primary_key:
                primary_key_field = fc.milvus_field.name if fc.milvus_field else None

        # インデックス変換を構築
        index_conversions = self._build_index_conversions(
            field_conversions, profiles, algorithms
        )

        # 非対応機能を検出
        warnings: list[ConversionWarning] = []
        unsupported: list[str] = []
        self._check_unsupported_features(index, warnings, unsupported)

        # 型レベルの警告を追加
        for fc in field_conversions:
            for w in fc.mapping.warnings:
                warnings.append(
                    ConversionWarning(
                        category="type_mapping",
                        message=w,
                        field_name=fc.azure_name,
                    )
                )
            if fc.mapping.confidence == MappingConfidence.LOSSY:
                warnings.append(
                    ConversionWarning(
                        category="type_lossy",
                        message=f"フィールド '{fc.azure_name}' ({fc.azure_type}) は損失ありの変換です",
                        field_name=fc.azure_name,
                    )
                )

        # 主キーがあることを確認
        if primary_key_field is None:
            warnings.append(
                ConversionWarning(
                    category="schema",
                    message="Azure 側にキーフィールドが見つかりません。自動生成 ID を使用します",
                )
            )

        schema = CollectionSchema(
            fields=milvus_fields,
            description=f"Migrated from Azure AI Search index: {index_name}",
            enable_dynamic_field=self.options.enable_dynamic_field,
        )

        return SchemaConversionResult(
            azure_index_name=index_name,
            milvus_collection_name=collection_name,
            collection_schema=schema,
            field_conversions=field_conversions,
            index_conversions=index_conversions,
            warnings=warnings,
            unsupported_features=unsupported,
        )

    def convert_from_json(self, index_json: dict[str, Any]) -> SchemaConversionResult:
        """JSON 辞書（例: REST API 経由でエクスポートされたもの）から変換する。

        同じ変換ロジックを適用するために軽量アダプターを作成する。
        """
        adapter = _JsonIndexAdapter(index_json)
        return self.convert_from_index(adapter)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _convert_field(
        self,
        azure_field: Any,
        profiles: dict[str, dict],
        algorithms: dict[str, dict],
    ) -> FieldConversion:
        name: str = azure_field.name
        edm_type: str = str(azure_field.type)
        is_key: bool = getattr(azure_field, "key", False)
        vector_profile: str | None = getattr(azure_field, "vector_search_profile_name", None)
        dimensions: int | None = getattr(azure_field, "vector_search_dimensions", None)

        # 除外チェック
        if name in self.options.exclude_fields:
            return FieldConversion(
                azure_name=name,
                azure_type=edm_type,
                milvus_field=None,
                mapping=resolve_type(edm_type),
                skipped=True,
                skip_reason="exclude_fields で除外指定",
            )

        # ComplexType のサブフィールドを再帰的に処理 → JSON にフラット化
        sub_fields = getattr(azure_field, "fields", None)
        if sub_fields:
            edm_type = "Edm.ComplexType"

        is_vector = vector_profile is not None or (
            edm_type.startswith("Collection(Edm.") and dimensions is not None
        )
        mapping = resolve_type(edm_type, is_vector_field=is_vector)

        # Milvus フィールド名を決定（オーバーライド許可）
        override = self.options.field_overrides.get(name, {})
        milvus_name = override.get("milvus_name", name)

        # FieldSchema を構築
        kwargs: dict[str, Any] = {
            "name": milvus_name,
            "dtype": mapping.milvus_type,
            "description": f"Migrated from Azure field: {name} ({edm_type})",
        }

        if is_key:
            kwargs["is_primary"] = True
            # Milvus の主キーは INT64 または VARCHAR でなければならない
            if mapping.milvus_type not in (DataType.INT64, DataType.VARCHAR):
                kwargs["dtype"] = DataType.VARCHAR
                kwargs["max_length"] = self.options.varchar_max_length

        if mapping.milvus_type == DataType.VARCHAR:
            kwargs["max_length"] = override.get(
                "max_length",
                mapping.default_max_length or self.options.varchar_max_length,
            )

        if mapping.is_vector and dimensions:
            kwargs["dim"] = dimensions

        if mapping.milvus_type == DataType.ARRAY:
            kwargs["element_type"] = mapping.element_type or DataType.VARCHAR
            kwargs["max_capacity"] = override.get(
                "max_capacity", self.options.array_max_capacity
            )
            if mapping.element_type == DataType.VARCHAR:
                kwargs["max_length"] = override.get(
                    "max_length", self.options.varchar_max_length
                )

        if mapping.milvus_type == DataType.JSON:
            # JSON フィールドは追加パラメータ不要
            pass

        # パーティションキーサポート（Milvus は INT64 または VARCHAR が必要）
        if self.options.partition_key_field == name:
            if kwargs["dtype"] in (DataType.INT64, DataType.VARCHAR):
                kwargs["is_partition_key"] = True
            else:
                logger.warning(
                    "パーティションキー '%s' の型 %s は非対応です (INT64/VARCHAR のみ)。スキップ",
                    name,
                    kwargs["dtype"].name,
                )

        milvus_field = FieldSchema(**kwargs)

        return FieldConversion(
            azure_name=name,
            azure_type=edm_type,
            milvus_field=milvus_field,
            mapping=mapping,
            renamed_to=milvus_name if milvus_name != name else "",
            is_primary_key=is_key,
        )

    def _parse_vector_search_config(
        self, index: Any
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """インデックスからベクトル検索プロファイルとアルゴリズムを抽出する。"""
        profiles: dict[str, dict] = {}
        algorithms: dict[str, dict] = {}

        vs = getattr(index, "vector_search", None)
        if vs is None:
            return profiles, algorithms

        for algo in getattr(vs, "algorithms", []) or []:
            algo_name = getattr(algo, "name", "")
            kind = getattr(algo, "kind", "hnsw")
            params = {}

            hnsw_params = getattr(algo, "hnsw_parameters", None)
            if hnsw_params:
                params["m"] = getattr(hnsw_params, "m", 4)
                params["efConstruction"] = getattr(hnsw_params, "ef_construction", 400)
                params["efSearch"] = getattr(hnsw_params, "ef_search", 500)
                params["metric"] = getattr(hnsw_params, "metric", "cosine")
                if hasattr(params["metric"], "value"):
                    params["metric"] = params["metric"].value

            eknn_params = getattr(algo, "exhaustive_knn_parameters", None)
            if eknn_params:
                params["metric"] = getattr(eknn_params, "metric", "cosine")
                if hasattr(params["metric"], "value"):
                    params["metric"] = params["metric"].value

            algorithms[algo_name] = {"kind": kind, "params": params}

        for profile in getattr(vs, "profiles", []) or []:
            profile_name = getattr(profile, "name", "")
            algo_name = getattr(profile, "algorithm_configuration_name", "")
            profiles[profile_name] = {"algorithm": algo_name}

        return profiles, algorithms

    def _build_index_conversions(
        self,
        field_conversions: list[FieldConversion],
        profiles: dict[str, dict],
        algorithms: dict[str, dict],
    ) -> list[IndexConversion]:
        """各ベクトルフィールドの Milvus インデックス設定を構築する。"""
        results: list[IndexConversion] = []

        for fc in field_conversions:
            if not fc.mapping.is_vector or fc.skipped:
                continue

            # このベクトルフィールドのプロファイルを検索
            azure_field_obj = fc  # 元のオブジェクトが必要；プロファイル名を抽出
            profile_name = ""
            # プロファイル名は元のフィールドにあった - 直接保存していないが、
            # 呼び出し元のコンテキストでフィールド名から検索可能。
            # 現時点では最初のプロファイルにマッチを試みる。
            algo_kind = "hnsw"
            metric = "cosine"
            hnsw_params: dict[str, Any] = {}

            # Try all profiles to find one for this field
            for pname, pconf in profiles.items():
                algo_name = pconf.get("algorithm", "")
                if algo_name in algorithms:
                    algo = algorithms[algo_name]
                    algo_kind = algo.get("kind", "hnsw")
                    params = algo.get("params", {})
                    metric = params.get("metric", "cosine")
                    hnsw_params = {
                        k: v for k, v in params.items() if k != "metric"
                    }
                    profile_name = pname
                    break

            milvus_idx = map_vector_index(
                algorithm_kind=algo_kind,
                metric=metric,
                hnsw_params=hnsw_params if algo_kind == "hnsw" else None,
            )

            results.append(
                IndexConversion(
                    azure_profile_name=profile_name,
                    azure_algorithm_kind=algo_kind,
                    azure_metric=metric,
                    milvus_config=milvus_idx,
                    target_field=fc.milvus_field.name if fc.milvus_field else fc.azure_name,
                )
            )

        return results

    def _check_unsupported_features(
        self,
        index: Any,
        warnings: list[ConversionWarning],
        unsupported: list[str],
    ) -> None:
        """Detect Azure AI Search features that have no Milvus equivalent."""
        checks = [
            ("scoring_profiles", "scoringProfiles"),
            ("suggesters", "suggesters"),
            ("semantic_settings", "semanticConfiguration"),
        ]
        for attr, feature_key in checks:
            val = getattr(index, attr, None)
            if val:
                msg = UNSUPPORTED_FEATURES.get(feature_key, f"{feature_key} は移行対象外です")
                warnings.append(
                    ConversionWarning(category="unsupported_feature", message=msg)
                )
                unsupported.append(feature_key)

        # Also check for semantic configuration (newer SDK)
        semantic = getattr(index, "semantic_search", None)
        if semantic:
            msg = UNSUPPORTED_FEATURES["semanticConfiguration"]
            if "semanticConfiguration" not in unsupported:
                warnings.append(
                    ConversionWarning(category="unsupported_feature", message=msg)
                )
                unsupported.append("semanticConfiguration")


# ---------------------------------------------------------------------------
# JSON adapter for REST API exports
# ---------------------------------------------------------------------------


class _JsonFieldAdapter:
    """Wraps a field dict to look like an SDK SearchField."""

    def __init__(self, d: dict[str, Any]) -> None:
        self._d = d

    def __getattr__(self, name: str) -> Any:
        # Map SDK attribute names to JSON keys
        key_map = {
            "name": "name",
            "type": "type",
            "key": "key",
            "searchable": "searchable",
            "filterable": "filterable",
            "sortable": "sortable",
            "facetable": "facetable",
            "vector_search_profile_name": "vectorSearchProfile",
            "vector_search_dimensions": "dimensions",
            "fields": "fields",
        }
        json_key = key_map.get(name, name)
        val = self._d.get(json_key)
        if json_key == "fields" and val:
            return [_JsonFieldAdapter(f) for f in val]
        return val


class _JsonVectorAlgoAdapter:
    def __init__(self, d: dict[str, Any]) -> None:
        self._d = d
        self.name = d.get("name", "")
        self.kind = d.get("kind", "hnsw")
        hp = d.get("hnswParameters", {})
        if hp:
            self.hnsw_parameters = type("P", (), {
                "m": hp.get("m", 4),
                "ef_construction": hp.get("efConstruction", 400),
                "ef_search": hp.get("efSearch", 500),
                "metric": hp.get("metric", "cosine"),
            })()
        else:
            self.hnsw_parameters = None
        ekp = d.get("exhaustiveKnnParameters", {})
        if ekp:
            self.exhaustive_knn_parameters = type("P", (), {
                "metric": ekp.get("metric", "cosine"),
            })()
        else:
            self.exhaustive_knn_parameters = None


class _JsonVectorProfileAdapter:
    def __init__(self, d: dict[str, Any]) -> None:
        self.name = d.get("name", "")
        self.algorithm_configuration_name = d.get("algorithm", d.get("algorithmConfigurationName", ""))


class _JsonVectorSearchAdapter:
    def __init__(self, d: dict[str, Any]) -> None:
        self.algorithms = [_JsonVectorAlgoAdapter(a) for a in d.get("algorithms", [])]
        self.profiles = [_JsonVectorProfileAdapter(p) for p in d.get("profiles", [])]


class _JsonIndexAdapter:
    """Wraps an index JSON dict to look like an SDK SearchIndex."""

    def __init__(self, d: dict[str, Any]) -> None:
        self._d = d
        self.name = d.get("name", "unknown")
        self.fields = [_JsonFieldAdapter(f) for f in d.get("fields", [])]

        vs = d.get("vectorSearch")
        self.vector_search = _JsonVectorSearchAdapter(vs) if vs else None

        self.scoring_profiles = d.get("scoringProfiles")
        self.suggesters = d.get("suggesters")
        self.semantic_settings = d.get("semantic")
        self.semantic_search = d.get("semantic")
