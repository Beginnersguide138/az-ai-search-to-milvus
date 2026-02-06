"""スキーマ変換エンジンのテスト。"""

from __future__ import annotations

from pymilvus import DataType

from az_search_to_milvus.config import MigrationOptions
from az_search_to_milvus.schema_converter import SchemaConverter
from az_search_to_milvus.type_mapping import MappingConfidence


class TestSimpleIndexConversion:
    def test_converts_all_fields(self, simple_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(simple_index)
        assert result.azure_index_name == "products"
        # 6 fields, none skipped
        assert len(result.field_conversions) == 6
        converted_count = sum(1 for fc in result.field_conversions if not fc.skipped)
        assert converted_count == 6

    def test_primary_key_detected(self, simple_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(simple_index)
        pk = next(fc for fc in result.field_conversions if fc.is_primary_key)
        assert pk.azure_name == "id"
        assert pk.milvus_field.is_primary is True
        assert pk.milvus_field.dtype == DataType.VARCHAR

    def test_vector_field_detected(self, simple_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(simple_index)
        vec = next(fc for fc in result.field_conversions if fc.mapping.is_vector)
        assert vec.azure_name == "embedding"
        assert vec.milvus_field.dtype == DataType.FLOAT_VECTOR
        assert vec.milvus_field.params.get("dim") == 1536

    def test_collection_schema_valid(self, simple_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(simple_index)
        schema = result.collection_schema
        assert schema is not None
        field_names = [f.name for f in schema.fields]
        assert "id" in field_names
        assert "embedding" in field_names

    def test_index_conversion(self, simple_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(simple_index)
        assert len(result.index_conversions) >= 1
        ic = result.index_conversions[0]
        assert ic.milvus_config.index_type == "HNSW"
        assert ic.milvus_config.metric_type == "COSINE"

    def test_tags_mapped_to_array(self, simple_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(simple_index)
        tags_fc = next(fc for fc in result.field_conversions if fc.azure_name == "tags")
        assert tags_fc.milvus_field.dtype == DataType.ARRAY

    def test_boolean_field(self, simple_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(simple_index)
        stock_fc = next(fc for fc in result.field_conversions if fc.azure_name == "in_stock")
        assert stock_fc.milvus_field.dtype == DataType.BOOL

    def test_double_field(self, simple_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(simple_index)
        price_fc = next(fc for fc in result.field_conversions if fc.azure_name == "price")
        assert price_fc.milvus_field.dtype == DataType.DOUBLE


class TestComplexIndexConversion:
    def test_multiple_vector_fields(self, complex_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(complex_index)
        vector_fields = [
            fc for fc in result.field_conversions if fc.mapping.is_vector
        ]
        assert len(vector_fields) == 2

    def test_float16_vector(self, complex_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(complex_index)
        img_fc = next(
            fc for fc in result.field_conversions if fc.azure_name == "image_embedding"
        )
        assert img_fc.milvus_field.dtype == DataType.FLOAT16_VECTOR
        assert img_fc.milvus_field.params.get("dim") == 512

    def test_datetime_maps_to_varchar(self, complex_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(complex_index)
        dt_fc = next(
            fc for fc in result.field_conversions if fc.azure_name == "created_date"
        )
        assert dt_fc.milvus_field.dtype == DataType.VARCHAR
        assert dt_fc.mapping.confidence == MappingConfidence.SEMANTIC

    def test_geography_maps_to_json(self, complex_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(complex_index)
        geo_fc = next(
            fc for fc in result.field_conversions if fc.azure_name == "location"
        )
        assert geo_fc.milvus_field.dtype == DataType.JSON

    def test_detects_unsupported_features(self, complex_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(complex_index)
        assert len(result.unsupported_features) > 0
        assert "scoringProfiles" in result.unsupported_features

    def test_warnings_generated(self, complex_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(complex_index)
        assert len(result.warnings) > 0

    def test_summary(self, complex_index) -> None:
        converter = SchemaConverter()
        result = converter.convert_from_index(complex_index)
        summary = result.summary()
        assert summary["azure_index"] == "documents-v2"
        assert summary["vector_fields"] == 2


class TestExcludeFields:
    def test_exclude_field(self, simple_index) -> None:
        options = MigrationOptions(exclude_fields=["tags"])
        converter = SchemaConverter(options)
        result = converter.convert_from_index(simple_index)
        tags_fc = next(fc for fc in result.field_conversions if fc.azure_name == "tags")
        assert tags_fc.skipped is True
        assert tags_fc.milvus_field is None


class TestFieldOverrides:
    def test_rename_field(self, simple_index) -> None:
        options = MigrationOptions(
            field_overrides={"title": {"milvus_name": "product_title"}}
        )
        converter = SchemaConverter(options)
        result = converter.convert_from_index(simple_index)
        title_fc = next(fc for fc in result.field_conversions if fc.azure_name == "title")
        assert title_fc.milvus_field.name == "product_title"
        assert title_fc.renamed_to == "product_title"


class TestPartitionKey:
    def test_partition_key_set_on_varchar(self, simple_index) -> None:
        """Milvus ではパーティションキーは INT64 または VARCHAR でなければならない。"""
        options = MigrationOptions(partition_key_field="title")
        converter = SchemaConverter(options)
        result = converter.convert_from_index(simple_index)
        title_fc = next(fc for fc in result.field_conversions if fc.azure_name == "title")
        assert title_fc.milvus_field.is_partition_key is True

    def test_partition_key_invalid_type_generates_warning(self, simple_index) -> None:
        """ARRAY フィールドはパーティションキーにできない — 警告してスキップすべき。"""
        options = MigrationOptions(partition_key_field="tags")
        converter = SchemaConverter(options)
        result = converter.convert_from_index(simple_index)
        tags_fc = next(fc for fc in result.field_conversions if fc.azure_name == "tags")
        # Should NOT be set as partition key (invalid type)
        assert tags_fc.milvus_field.is_partition_key is not True


class TestJsonConversion:
    def test_convert_from_json(self) -> None:
        index_json = {
            "name": "test-json-index",
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True},
                {"name": "title", "type": "Edm.String", "searchable": True},
                {
                    "name": "vec",
                    "type": "Collection(Edm.Single)",
                    "dimensions": 384,
                    "vectorSearchProfile": "default",
                },
            ],
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "algo1",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "m": 8,
                            "efConstruction": 200,
                            "efSearch": 100,
                            "metric": "euclidean",
                        },
                    },
                ],
                "profiles": [
                    {"name": "default", "algorithmConfigurationName": "algo1"},
                ],
            },
        }
        converter = SchemaConverter()
        result = converter.convert_from_json(index_json)
        assert result.azure_index_name == "test-json-index"
        assert len(result.field_conversions) == 3

        vec_fc = next(fc for fc in result.field_conversions if fc.azure_name == "vec")
        assert vec_fc.milvus_field.dtype == DataType.FLOAT_VECTOR
        assert vec_fc.milvus_field.params.get("dim") == 384
