"""Azure AI Search から Milvus へのベクトルインデックスアルゴリズムマッピング。

Azure AI Search は2つのベクトル検索アルゴリズムをサポートしています:
  - hnsw (Hierarchical Navigable Small World)
  - exhaustiveKnn (ブルートフォース)

Milvus はより豊富なインデックス型を提供しています。このモジュールは Azure の
アルゴリズムを適切な Milvus のデフォルトにマッピングし、移行時の利点として
Milvus で利用可能な追加のインデックス型を紹介します。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# 距離メトリクスマッピング
# ---------------------------------------------------------------------------

METRIC_MAP: dict[str, str] = {
    "cosine": "COSINE",
    "euclidean": "L2",
    "dotProduct": "IP",
    # Azure AI Search は一部の API バージョンでキャメルケースの名前も使用する
    "hamming": "HAMMING",
}


@dataclass
class MilvusIndexConfig:
    """解決済みの Milvus インデックス設定。"""

    index_type: str
    metric_type: str
    params: dict[str, Any] = field(default_factory=dict)
    search_params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def map_vector_index(
    *,
    algorithm_kind: str,
    metric: str,
    hnsw_params: dict[str, Any] | None = None,
) -> MilvusIndexConfig:
    """Azure AI Search のベクトルアルゴリズムを Milvus インデックス設定にマッピングする。

    パラメータ
    ----------
    algorithm_kind:
        ``"hnsw"`` または ``"exhaustiveKnn"`` (Azure AI Search のアルゴリズム種別)。
    metric:
        Azure のメトリクス名: ``"cosine"``、``"euclidean"``、または ``"dotProduct"``。
    hnsw_params:
        Azure からの HNSW パラメータ (``m``、``efConstruction``、``efSearch``)。

    戻り値
    -------
    ``pymilvus`` で使用可能な MilvusIndexConfig。
    """
    metric_type = METRIC_MAP.get(metric, "COSINE")

    if algorithm_kind == "hnsw":
        params = hnsw_params or {}
        return MilvusIndexConfig(
            index_type="HNSW",
            metric_type=metric_type,
            params={
                "M": params.get("m", 4),
                "efConstruction": params.get("efConstruction", 400),
            },
            search_params={
                "ef": params.get("efSearch", 500),
            },
            notes="HNSW パラメータを Azure AI Search から 1:1 で移行",
        )

    if algorithm_kind == "exhaustiveKnn":
        return MilvusIndexConfig(
            index_type="FLAT",
            metric_type=metric_type,
            params={},
            search_params={},
            notes="Exhaustive KNN → Milvus FLAT (ブルートフォース)",
        )

    # フォールバック
    return MilvusIndexConfig(
        index_type="HNSW",
        metric_type=metric_type,
        params={"M": 16, "efConstruction": 256},
        search_params={"ef": 256},
        notes=f"未知のアルゴリズム '{algorithm_kind}' — HNSW デフォルトにフォールバック",
    )


# ---------------------------------------------------------------------------
# Milvus 専用インデックス型 (Azure AI Search に対する優位性)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MilvusOnlyIndex:
    """Milvus で利用可能だが Azure AI Search にはないインデックス型。"""

    name: str
    description: str
    use_case: str
    params_example: dict[str, Any]
    requires_gpu: bool = False


MILVUS_EXCLUSIVE_INDEXES: list[MilvusOnlyIndex] = [
    MilvusOnlyIndex(
        name="IVF_FLAT",
        description="転置ファイルインデックス + フラット量子化",
        use_case="メモリ効率と速度のバランスが良い。数百万〜数千万レコードに適する",
        params_example={"nlist": 1024},
    ),
    MilvusOnlyIndex(
        name="IVF_SQ8",
        description="IVF + スカラー量子化 (8-bit)",
        use_case="メモリ使用量を 70-75% 削減。精度のわずかな低下を許容できる場合に最適",
        params_example={"nlist": 1024},
    ),
    MilvusOnlyIndex(
        name="IVF_PQ",
        description="IVF + プロダクト量子化",
        use_case="非常に大規模なデータセットでメモリ使用量を大幅に削減",
        params_example={"nlist": 1024, "m": 8, "nbits": 8},
    ),
    MilvusOnlyIndex(
        name="SCANN",
        description="Score-aware quantization with anisotropic vector quantization",
        use_case="IVF_PQ より高精度。Google Research の SCANN ベース",
        params_example={"nlist": 1024, "with_raw_data": True},
    ),
    MilvusOnlyIndex(
        name="DiskANN",
        description="ディスクベースの ANN インデックス (Microsoft Research)",
        use_case="メモリに収まらない大規模データセット。SSD/NVMe ストレージを活用",
        params_example={},
    ),
    MilvusOnlyIndex(
        name="GPU_IVF_FLAT",
        description="GPU 上の IVF_FLAT",
        use_case="GPU を搭載した Azure VM (NC/ND シリーズ) でのハイスループット検索",
        params_example={"nlist": 1024},
        requires_gpu=True,
    ),
    MilvusOnlyIndex(
        name="GPU_IVF_PQ",
        description="GPU 上の IVF_PQ",
        use_case="GPU + 大規模データセットの高速検索",
        params_example={"nlist": 1024, "m": 8, "nbits": 8},
        requires_gpu=True,
    ),
    MilvusOnlyIndex(
        name="GPU_CAGRA",
        description="GPU ネイティブグラフベースインデックス (NVIDIA RAFT)",
        use_case="GPU 環境での最高性能を発揮。リアルタイムベクトル検索に最適",
        params_example={"intermediate_graph_degree": 64, "graph_degree": 32},
        requires_gpu=True,
    ),
    MilvusOnlyIndex(
        name="GPU_BRUTE_FORCE",
        description="GPU ブルートフォース検索",
        use_case="小〜中規模データセットでの 100% 再現率が必要な場合",
        params_example={},
        requires_gpu=True,
    ),
    MilvusOnlyIndex(
        name="SPARSE_INVERTED_INDEX",
        description="スパースベクトル用転置インデックス",
        use_case="BM25/SPLADE 等のスパース表現によるハイブリッド検索",
        params_example={"drop_ratio_build": 0.2},
    ),
    MilvusOnlyIndex(
        name="SPARSE_WAND",
        description="スパースベクトル用 WAND アルゴリズム",
        use_case="大規模スパースベクトルでの高速Top-K検索",
        params_example={"drop_ratio_build": 0.2},
    ),
]
