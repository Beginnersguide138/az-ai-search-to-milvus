"""Tests for vector index algorithm mapping."""

from az_search_to_milvus.index_mapping import (
    METRIC_MAP,
    MILVUS_EXCLUSIVE_INDEXES,
    map_vector_index,
)


class TestMetricMapping:
    def test_cosine(self) -> None:
        assert METRIC_MAP["cosine"] == "COSINE"

    def test_euclidean(self) -> None:
        assert METRIC_MAP["euclidean"] == "L2"

    def test_dot_product(self) -> None:
        assert METRIC_MAP["dotProduct"] == "IP"

    def test_hamming(self) -> None:
        assert METRIC_MAP["hamming"] == "HAMMING"


class TestHnswMapping:
    def test_default_hnsw(self) -> None:
        cfg = map_vector_index(algorithm_kind="hnsw", metric="cosine")
        assert cfg.index_type == "HNSW"
        assert cfg.metric_type == "COSINE"

    def test_hnsw_preserves_params(self) -> None:
        cfg = map_vector_index(
            algorithm_kind="hnsw",
            metric="euclidean",
            hnsw_params={"m": 16, "efConstruction": 256, "efSearch": 128},
        )
        assert cfg.index_type == "HNSW"
        assert cfg.metric_type == "L2"
        assert cfg.params["M"] == 16
        assert cfg.params["efConstruction"] == 256
        assert cfg.search_params["ef"] == 128

    def test_hnsw_dot_product(self) -> None:
        cfg = map_vector_index(algorithm_kind="hnsw", metric="dotProduct")
        assert cfg.metric_type == "IP"


class TestExhaustiveKnnMapping:
    def test_exhaustive_knn_maps_to_flat(self) -> None:
        cfg = map_vector_index(algorithm_kind="exhaustiveKnn", metric="cosine")
        assert cfg.index_type == "FLAT"
        assert cfg.metric_type == "COSINE"

    def test_exhaustive_knn_euclidean(self) -> None:
        cfg = map_vector_index(algorithm_kind="exhaustiveKnn", metric="euclidean")
        assert cfg.index_type == "FLAT"
        assert cfg.metric_type == "L2"


class TestUnknownAlgorithm:
    def test_unknown_falls_back_to_hnsw(self) -> None:
        cfg = map_vector_index(algorithm_kind="unknown_algo", metric="cosine")
        assert cfg.index_type == "HNSW"
        assert cfg.metric_type == "COSINE"


class TestMilvusExclusiveIndexes:
    def test_exclusive_indexes_exist(self) -> None:
        assert len(MILVUS_EXCLUSIVE_INDEXES) > 0

    def test_includes_ivf_flat(self) -> None:
        names = [idx.name for idx in MILVUS_EXCLUSIVE_INDEXES]
        assert "IVF_FLAT" in names

    def test_includes_gpu_cagra(self) -> None:
        names = [idx.name for idx in MILVUS_EXCLUSIVE_INDEXES]
        assert "GPU_CAGRA" in names

    def test_gpu_indexes_flagged(self) -> None:
        gpu_indexes = [idx for idx in MILVUS_EXCLUSIVE_INDEXES if idx.requires_gpu]
        assert len(gpu_indexes) >= 4

    def test_sparse_indexes_present(self) -> None:
        names = [idx.name for idx in MILVUS_EXCLUSIVE_INDEXES]
        assert "SPARSE_INVERTED_INDEX" in names
        assert "SPARSE_WAND" in names
