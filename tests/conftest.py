"""移行ツールテストスイート共有フィクスチャ。"""

from __future__ import annotations

from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Azure AI Search フィールド/インデックスのモックオブジェクト
# ---------------------------------------------------------------------------


class MockSearchField:
    """``azure.search.documents.indexes.models.SearchField`` のモック。"""

    def __init__(self, **kwargs: Any) -> None:
        self.name: str = kwargs.get("name", "")
        self.type: str = kwargs.get("type", "Edm.String")
        self.key: bool = kwargs.get("key", False)
        self.searchable: bool = kwargs.get("searchable", False)
        self.filterable: bool = kwargs.get("filterable", False)
        self.sortable: bool = kwargs.get("sortable", False)
        self.facetable: bool = kwargs.get("facetable", False)
        self.vector_search_profile_name: str | None = kwargs.get(
            "vector_search_profile_name"
        )
        self.vector_search_dimensions: int | None = kwargs.get(
            "vector_search_dimensions"
        )
        self.fields: list | None = kwargs.get("fields")


class MockHnswParameters:
    def __init__(self, m: int = 4, ef_construction: int = 400, ef_search: int = 500,
                 metric: str = "cosine") -> None:
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric


class MockVectorAlgorithm:
    def __init__(self, name: str, kind: str = "hnsw",
                 hnsw_parameters: MockHnswParameters | None = None) -> None:
        self.name = name
        self.kind = kind
        self.hnsw_parameters = hnsw_parameters
        self.exhaustive_knn_parameters = None


class MockVectorProfile:
    def __init__(self, name: str, algorithm_configuration_name: str) -> None:
        self.name = name
        self.algorithm_configuration_name = algorithm_configuration_name


class MockVectorSearch:
    def __init__(
        self,
        algorithms: list[MockVectorAlgorithm] | None = None,
        profiles: list[MockVectorProfile] | None = None,
    ) -> None:
        self.algorithms = algorithms or []
        self.profiles = profiles or []


class MockSearchIndex:
    """``azure.search.documents.indexes.models.SearchIndex`` のモック。"""

    def __init__(
        self,
        name: str = "test-index",
        fields: list[MockSearchField] | None = None,
        vector_search: MockVectorSearch | None = None,
        scoring_profiles: list | None = None,
        suggesters: list | None = None,
        semantic_settings: Any = None,
        semantic_search: Any = None,
    ) -> None:
        self.name = name
        self.fields = fields or []
        self.vector_search = vector_search
        self.scoring_profiles = scoring_profiles
        self.suggesters = suggesters
        self.semantic_settings = semantic_settings
        self.semantic_search = semantic_search


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_index() -> MockSearchIndex:
    """スカラー + ベクトルフィールドを含むシンプルなインデックス。"""
    return MockSearchIndex(
        name="products",
        fields=[
            MockSearchField(name="id", type="Edm.String", key=True, filterable=True),
            MockSearchField(name="title", type="Edm.String", searchable=True),
            MockSearchField(name="price", type="Edm.Double", filterable=True, sortable=True),
            MockSearchField(name="in_stock", type="Edm.Boolean", filterable=True),
            MockSearchField(name="tags", type="Collection(Edm.String)", filterable=True),
            MockSearchField(
                name="embedding",
                type="Collection(Edm.Single)",
                vector_search_profile_name="default-profile",
                vector_search_dimensions=1536,
            ),
        ],
        vector_search=MockVectorSearch(
            algorithms=[
                MockVectorAlgorithm(
                    name="hnsw-config",
                    kind="hnsw",
                    hnsw_parameters=MockHnswParameters(
                        m=4, ef_construction=400, ef_search=500, metric="cosine",
                    ),
                ),
            ],
            profiles=[
                MockVectorProfile(
                    name="default-profile",
                    algorithm_configuration_name="hnsw-config",
                ),
            ],
        ),
    )


@pytest.fixture()
def complex_index() -> MockSearchIndex:
    """複数ベクトルフィールドと非対応機能を含む複雑なインデックス。"""
    return MockSearchIndex(
        name="documents-v2",
        fields=[
            MockSearchField(name="doc_id", type="Edm.String", key=True),
            MockSearchField(name="title", type="Edm.String", searchable=True),
            MockSearchField(name="content", type="Edm.String", searchable=True),
            MockSearchField(name="category", type="Edm.String", filterable=True, facetable=True),
            MockSearchField(name="page_count", type="Edm.Int32", filterable=True),
            MockSearchField(name="file_size", type="Edm.Int64", sortable=True),
            MockSearchField(name="created_date", type="Edm.DateTimeOffset", filterable=True),
            MockSearchField(name="location", type="Edm.GeographyPoint", filterable=True),
            MockSearchField(name="labels", type="Collection(Edm.String)", filterable=True),
            MockSearchField(name="scores", type="Collection(Edm.Double)"),
            MockSearchField(
                name="text_embedding",
                type="Collection(Edm.Single)",
                vector_search_profile_name="text-profile",
                vector_search_dimensions=768,
            ),
            MockSearchField(
                name="image_embedding",
                type="Collection(Edm.Half)",
                vector_search_profile_name="image-profile",
                vector_search_dimensions=512,
            ),
        ],
        vector_search=MockVectorSearch(
            algorithms=[
                MockVectorAlgorithm(
                    name="hnsw-text",
                    kind="hnsw",
                    hnsw_parameters=MockHnswParameters(m=16, ef_construction=256, metric="cosine"),
                ),
                MockVectorAlgorithm(name="flat-image", kind="exhaustiveKnn"),
            ],
            profiles=[
                MockVectorProfile("text-profile", "hnsw-text"),
                MockVectorProfile("image-profile", "flat-image"),
            ],
        ),
        scoring_profiles=[{"name": "boost-recent"}],
        suggesters=[{"name": "title-suggester"}],
        semantic_search={"configurations": [{"name": "semantic-config"}]},
    )


@pytest.fixture()
def sample_documents() -> list[dict[str, Any]]:
    """データ変換テスト用のサンプル Azure AI Search ドキュメント。"""
    return [
        {
            "id": "doc-001",
            "title": "Introduction to Vector Databases",
            "price": 29.99,
            "in_stock": True,
            "tags": ["database", "vector", "AI"],
            "embedding": [0.1] * 1536,
        },
        {
            "id": "doc-002",
            "title": "Advanced Milvus Patterns",
            "price": 49.99,
            "in_stock": False,
            "tags": ["milvus", "advanced"],
            "embedding": [0.2] * 1536,
        },
        {
            "id": "doc-003",
            "title": "Cloud Migration Strategies",
            "price": 0.0,
            "in_stock": True,
            "tags": [],
            "embedding": [0.3] * 1536,
        },
    ]
