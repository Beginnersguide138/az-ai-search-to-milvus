"""データ変換のテスト (外部サービス不要のユニットテスト)。"""

from __future__ import annotations

from typing import Any

from az_search_to_milvus.config import MigrationOptions
from az_search_to_milvus.data_migrator import DataTransformer
from az_search_to_milvus.schema_converter import SchemaConverter


class TestDataTransformer:
    def _make_transformer(self, index: Any) -> DataTransformer:
        converter = SchemaConverter(MigrationOptions())
        result = converter.convert_from_index(index)
        return DataTransformer(result.field_conversions)

    def test_transform_document(self, simple_index, sample_documents) -> None:
        transformer = self._make_transformer(simple_index)
        doc = sample_documents[0]
        result = transformer.transform_document(doc)

        assert result is not None
        assert result["id"] == "doc-001"
        assert result["title"] == "Introduction to Vector Databases"
        assert result["price"] == 29.99
        assert result["in_stock"] is True
        assert result["tags"] == ["database", "vector", "AI"]
        assert len(result["embedding"]) == 1536

    def test_transform_batch(self, simple_index, sample_documents) -> None:
        transformer = self._make_transformer(simple_index)
        results = transformer.transform_batch(sample_documents)
        assert len(results) == 3

    def test_missing_key_skips_document(self, simple_index) -> None:
        transformer = self._make_transformer(simple_index)
        doc = {"title": "No ID", "price": 10.0}
        result = transformer.transform_document(doc)
        assert result is None

    def test_null_values_get_defaults(self, simple_index) -> None:
        transformer = self._make_transformer(simple_index)
        doc = {
            "id": "null-test",
            "title": None,
            "price": None,
            "in_stock": None,
            "tags": None,
        }
        result = transformer.transform_document(doc)
        assert result is not None
        assert result["id"] == "null-test"
        assert result["title"] == ""
        assert result["price"] == 0.0
        assert result["in_stock"] is False
        assert result["tags"] == []

    def test_string_truncation(self, simple_index) -> None:
        transformer = self._make_transformer(simple_index)
        long_string = "x" * 100_000
        doc = {
            "id": "trunc-test",
            "title": long_string,
            "price": 0.0,
            "embedding": [0.1] * 1536,
        }
        result = transformer.transform_document(doc)
        assert result is not None
        assert len(result["title"]) <= 65_535

    def test_key_field_detected(self, simple_index) -> None:
        transformer = self._make_transformer(simple_index)
        assert transformer.key_field == "id"

    def test_vector_coerced_to_float(self, simple_index) -> None:
        transformer = self._make_transformer(simple_index)
        doc = {
            "id": "int-vec",
            "embedding": [1, 2, 3] + [0] * 1533,
        }
        result = transformer.transform_document(doc)
        assert result is not None
        assert all(isinstance(v, float) for v in result["embedding"])

    def test_tags_coerced_to_strings(self, simple_index) -> None:
        transformer = self._make_transformer(simple_index)
        doc = {
            "id": "mixed-tags",
            "tags": [1, 2, "three"],
        }
        result = transformer.transform_document(doc)
        assert result is not None
        assert result["tags"] == ["1", "2", "three"]


class TestDataTransformerWithExclusions:
    def test_excluded_fields_not_in_output(self, simple_index) -> None:
        options = MigrationOptions(exclude_fields=["tags"])
        converter = SchemaConverter(options)
        conversion = converter.convert_from_index(simple_index)
        transformer = DataTransformer(conversion.field_conversions)

        doc = {
            "id": "excl-test",
            "title": "Test",
            "price": 10.0,
            "tags": ["a", "b"],
            "embedding": [0.1] * 1536,
        }
        result = transformer.transform_document(doc)
        assert result is not None
        assert "tags" not in result
