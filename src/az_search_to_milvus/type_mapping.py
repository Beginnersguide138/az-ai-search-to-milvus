"""Azure AI Search の Edm 型から Milvus DataType への包括的な型マッピング。

Azure AI Search のフィールド型は OData Entity Data Model (Edm) に基づいています。
このモジュールは、スカラー型、コレクション (配列) 型、およびベクトル型を含む
Milvus 2.6.x DataType への完全なマッピングを提供します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pymilvus import DataType


class MappingConfidence(Enum):
    """型マッピングの信頼度レベル。"""

    EXACT = "exact"
    LOSSLESS = "lossless"  # 安全なアップキャスト、データ損失なし
    LOSSY = "lossy"  # 精度損失の可能性あり
    SEMANTIC = "semantic"  # 構造は異なるが意味的に等価
    UNSUPPORTED = "unsupported"  # Milvus に対応する型なし


@dataclass(frozen=True)
class TypeMapping:
    """Azure AI Search の型が Milvus の型にどのようにマッピングされるかを記述する。"""

    edm_type: str
    milvus_type: DataType
    confidence: MappingConfidence
    is_vector: bool = False
    element_type: DataType | None = None  # ARRAY フィールド用
    default_max_length: int | None = None  # VARCHAR 用
    notes: str = ""
    warnings: list[str] = field(default_factory=list)
    milvus_extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# スカラー型マッピング
# ---------------------------------------------------------------------------

SCALAR_TYPE_MAP: dict[str, TypeMapping] = {
    "Edm.String": TypeMapping(
        edm_type="Edm.String",
        milvus_type=DataType.VARCHAR,
        confidence=MappingConfidence.EXACT,
        default_max_length=65_535,
        notes="Azure AI Search の filterable/searchable 属性は Milvus では不要",
    ),
    "Edm.Int32": TypeMapping(
        edm_type="Edm.Int32",
        milvus_type=DataType.INT32,
        confidence=MappingConfidence.EXACT,
    ),
    "Edm.Int64": TypeMapping(
        edm_type="Edm.Int64",
        milvus_type=DataType.INT64,
        confidence=MappingConfidence.EXACT,
    ),
    "Edm.Double": TypeMapping(
        edm_type="Edm.Double",
        milvus_type=DataType.DOUBLE,
        confidence=MappingConfidence.EXACT,
    ),
    "Edm.Single": TypeMapping(
        edm_type="Edm.Single",
        milvus_type=DataType.FLOAT,
        confidence=MappingConfidence.EXACT,
        notes="スカラーの Edm.Single (非ベクトル)。ベクトルの場合は Collection(Edm.Single) を使用",
    ),
    "Edm.Boolean": TypeMapping(
        edm_type="Edm.Boolean",
        milvus_type=DataType.BOOL,
        confidence=MappingConfidence.EXACT,
    ),
    "Edm.DateTimeOffset": TypeMapping(
        edm_type="Edm.DateTimeOffset",
        milvus_type=DataType.VARCHAR,
        confidence=MappingConfidence.SEMANTIC,
        default_max_length=64,
        notes="ISO 8601 文字列として格納。Milvus にはネイティブの日時型がないため VARCHAR を使用",
        warnings=["日時の範囲フィルタリングは文字列比較になるため精度に注意"],
    ),
    "Edm.GeographyPoint": TypeMapping(
        edm_type="Edm.GeographyPoint",
        milvus_type=DataType.JSON,
        confidence=MappingConfidence.SEMANTIC,
        notes='JSON {"type":"Point","coordinates":[lon,lat]} として格納',
        warnings=["Milvus にはネイティブの地理空間インデックスがありません。geo.distance() は移行不可"],
    ),
    "Edm.ComplexType": TypeMapping(
        edm_type="Edm.ComplexType",
        milvus_type=DataType.JSON,
        confidence=MappingConfidence.SEMANTIC,
        notes="複合型は JSON フィールドとしてフラット化",
        warnings=["ネストされたフィルタリング ($filter=rooms/any()) は Milvus の JSON パスクエリに書き換え必要"],
    ),
    "Edm.Byte": TypeMapping(
        edm_type="Edm.Byte",
        milvus_type=DataType.INT16,
        confidence=MappingConfidence.LOSSLESS,
        notes="Milvus に UINT8/BYTE 型がないため INT16 にアップキャスト",
    ),
    "Edm.Int16": TypeMapping(
        edm_type="Edm.Int16",
        milvus_type=DataType.INT16,
        confidence=MappingConfidence.EXACT,
    ),
    "Edm.SByte": TypeMapping(
        edm_type="Edm.SByte",
        milvus_type=DataType.INT8,
        confidence=MappingConfidence.EXACT,
    ),
}

# ---------------------------------------------------------------------------
# コレクション (配列) 型マッピング — 非ベクトル
# ---------------------------------------------------------------------------

COLLECTION_TYPE_MAP: dict[str, TypeMapping] = {
    "Collection(Edm.String)": TypeMapping(
        edm_type="Collection(Edm.String)",
        milvus_type=DataType.ARRAY,
        confidence=MappingConfidence.EXACT,
        element_type=DataType.VARCHAR,
        notes="facetable/filterable な文字列配列。Milvus ARRAY(VARCHAR) にマッピング",
    ),
    "Collection(Edm.Int32)": TypeMapping(
        edm_type="Collection(Edm.Int32)",
        milvus_type=DataType.ARRAY,
        confidence=MappingConfidence.EXACT,
        element_type=DataType.INT32,
    ),
    "Collection(Edm.Int64)": TypeMapping(
        edm_type="Collection(Edm.Int64)",
        milvus_type=DataType.ARRAY,
        confidence=MappingConfidence.EXACT,
        element_type=DataType.INT64,
    ),
    "Collection(Edm.Double)": TypeMapping(
        edm_type="Collection(Edm.Double)",
        milvus_type=DataType.ARRAY,
        confidence=MappingConfidence.EXACT,
        element_type=DataType.DOUBLE,
    ),
}

# ---------------------------------------------------------------------------
# ベクトル型マッピング
# ---------------------------------------------------------------------------

VECTOR_TYPE_MAP: dict[str, TypeMapping] = {
    "Collection(Edm.Single)": TypeMapping(
        edm_type="Collection(Edm.Single)",
        milvus_type=DataType.FLOAT_VECTOR,
        confidence=MappingConfidence.EXACT,
        is_vector=True,
        notes="float32 ベクトル。パラメータ 1:1 マッピング",
    ),
    "Collection(Edm.Half)": TypeMapping(
        edm_type="Collection(Edm.Half)",
        milvus_type=DataType.FLOAT16_VECTOR,
        confidence=MappingConfidence.EXACT,
        is_vector=True,
        notes="float16 ベクトル。Milvus 2.6.x FLOAT16_VECTOR に直接対応",
    ),
    "Collection(Edm.Int16)": TypeMapping(
        edm_type="Collection(Edm.Int16)",
        milvus_type=DataType.FLOAT_VECTOR,
        confidence=MappingConfidence.LOSSY,
        is_vector=True,
        notes="int16 量子化ベクトル。Milvus に INT16_VECTOR がないため float32 にアップキャスト",
        warnings=[
            "int16→float32 変換によりメモリ使用量が約2倍になります",
            "検索精度はほぼ同等ですが、元の量子化された表現ではなくなります",
        ],
    ),
    "Collection(Edm.SByte)": TypeMapping(
        edm_type="Collection(Edm.SByte)",
        milvus_type=DataType.FLOAT_VECTOR,
        confidence=MappingConfidence.LOSSY,
        is_vector=True,
        notes="int8 量子化ベクトル。Milvus に INT8_VECTOR がないため float32 にアップキャスト",
        warnings=[
            "int8→float32 変換によりメモリ使用量が約4倍になります",
            "元のモデルで再エンコードして float32 ベクトルを直接使用することを推奨",
        ],
    ),
    "Collection(Edm.Byte)": TypeMapping(
        edm_type="Collection(Edm.Byte)",
        milvus_type=DataType.BINARY_VECTOR,
        confidence=MappingConfidence.EXACT,
        is_vector=True,
        notes="バイナリベクトル (パックドビット)。Milvus BINARY_VECTOR に直接対応",
    ),
}


# ---------------------------------------------------------------------------
# 未サポート / AI Search 専用機能
# ---------------------------------------------------------------------------

UNSUPPORTED_FEATURES: dict[str, str] = {
    "scoringProfiles": (
        "スコアリングプロファイルは Milvus に対応する機能がありません。"
        "アプリケーション層でのランキングロジック実装を検討してください。"
    ),
    "suggesters": (
        "サジェスター (オートコンプリート) は Milvus に対応する機能がありません。"
        "前方一致検索 (prefix query) やアプリケーション層での実装を検討してください。"
    ),
    "semanticConfiguration": (
        "セマンティックランカー (Semantic Ranker) は Azure AI Search 固有の機能です。"
        "Milvus では独自のリランカーモデル (Cross-Encoder 等) を統合してください。"
    ),
    "skillsets": (
        "AI エンリッチメントスキルセット (Skillset) は Azure AI Search 固有のパイプラインです。"
        "データ取り込みパイプラインは別途構築する必要があります。"
    ),
    "indexers": (
        "インデクサーは Azure AI Search のデータ取り込み自動化機能です。"
        "Milvus では Kafka/Spark コネクタ、または CDC を使ったパイプラインを検討してください。"
    ),
    "synonymMaps": (
        "同義語マップは Milvus に直接対応しません。"
        "クエリ拡張をアプリケーション層で実装するか、埋め込みモデルで吸収してください。"
    ),
    "encryptionKey": (
        "カスタマーマネージド暗号化キー (CMK) は Azure 固有です。"
        "Milvus ではディスク暗号化 (LUKS/dm-crypt) やクラウドのボリューム暗号化を使用してください。"
    ),
    "geo.distance()": (
        "地理空間関数 geo.distance() は Milvus に対応しません。"
        "地理空間検索が必要な場合は PostGIS 等との併用を検討してください。"
    ),
}


def resolve_type(edm_type: str, *, is_vector_field: bool = False) -> TypeMapping:
    """Azure AI Search の Edm 型文字列を Milvus の TypeMapping に解決する。

    パラメータ
    ----------
    edm_type:
        Edm 型文字列 (例: ``"Edm.String"``、``"Collection(Edm.Single)"``)。
    is_vector_field:
        このフィールドに ``vectorSearchProfile`` が設定されているかどうか。
        ``Collection(Edm.Single)`` がベクトルフィールドかスカラー float 配列かを
        判別するために使用される (エッジケース)。

    戻り値
    -------
    解決された Milvus 型とメタデータを含む TypeMapping。
    """
    # ベクトルフィールド
    if is_vector_field and edm_type in VECTOR_TYPE_MAP:
        return VECTOR_TYPE_MAP[edm_type]

    # vectorSearchProfile なしの Collection(Edm.Single) → ベクトルとして扱う
    # Azure AI Search ではほぼ常にベクトルフィールドとして使用されるため
    if edm_type in VECTOR_TYPE_MAP:
        return VECTOR_TYPE_MAP[edm_type]

    # 非ベクトルコレクション
    if edm_type in COLLECTION_TYPE_MAP:
        return COLLECTION_TYPE_MAP[edm_type]

    # スカラー型
    if edm_type in SCALAR_TYPE_MAP:
        return SCALAR_TYPE_MAP[edm_type]

    # 未知の型
    return TypeMapping(
        edm_type=edm_type,
        milvus_type=DataType.JSON,
        confidence=MappingConfidence.UNSUPPORTED,
        notes=f"未知の型 '{edm_type}' — JSON フォールバック",
        warnings=[f"型 '{edm_type}' は認識できません。JSON として格納されます"],
    )


def get_all_mappings() -> list[TypeMapping]:
    """ドキュメント作成 / レポート用にすべての既知の型マッピングを返す。"""
    mappings: list[TypeMapping] = []
    mappings.extend(SCALAR_TYPE_MAP.values())
    mappings.extend(COLLECTION_TYPE_MAP.values())
    mappings.extend(VECTOR_TYPE_MAP.values())
    return mappings
