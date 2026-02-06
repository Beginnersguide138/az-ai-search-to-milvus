"""Tests for the Edm → Milvus type mapping module."""

from pymilvus import DataType

from az_search_to_milvus.type_mapping import (
    COLLECTION_TYPE_MAP,
    SCALAR_TYPE_MAP,
    VECTOR_TYPE_MAP,
    MappingConfidence,
    get_all_mappings,
    resolve_type,
)


class TestScalarTypeMappings:
    def test_string_maps_to_varchar(self) -> None:
        m = resolve_type("Edm.String")
        assert m.milvus_type == DataType.VARCHAR
        assert m.confidence == MappingConfidence.EXACT

    def test_int32_maps_to_int32(self) -> None:
        m = resolve_type("Edm.Int32")
        assert m.milvus_type == DataType.INT32
        assert m.confidence == MappingConfidence.EXACT

    def test_int64_maps_to_int64(self) -> None:
        m = resolve_type("Edm.Int64")
        assert m.milvus_type == DataType.INT64
        assert m.confidence == MappingConfidence.EXACT

    def test_double_maps_to_double(self) -> None:
        m = resolve_type("Edm.Double")
        assert m.milvus_type == DataType.DOUBLE
        assert m.confidence == MappingConfidence.EXACT

    def test_single_maps_to_float(self) -> None:
        m = resolve_type("Edm.Single")
        assert m.milvus_type == DataType.FLOAT
        assert m.confidence == MappingConfidence.EXACT

    def test_boolean_maps_to_bool(self) -> None:
        m = resolve_type("Edm.Boolean")
        assert m.milvus_type == DataType.BOOL
        assert m.confidence == MappingConfidence.EXACT

    def test_datetimeoffset_maps_to_varchar(self) -> None:
        m = resolve_type("Edm.DateTimeOffset")
        assert m.milvus_type == DataType.VARCHAR
        assert m.confidence == MappingConfidence.SEMANTIC
        assert m.default_max_length == 64

    def test_geography_point_maps_to_json(self) -> None:
        m = resolve_type("Edm.GeographyPoint")
        assert m.milvus_type == DataType.JSON
        assert m.confidence == MappingConfidence.SEMANTIC
        assert len(m.warnings) > 0

    def test_complex_type_maps_to_json(self) -> None:
        m = resolve_type("Edm.ComplexType")
        assert m.milvus_type == DataType.JSON
        assert m.confidence == MappingConfidence.SEMANTIC

    def test_byte_maps_to_int16(self) -> None:
        m = resolve_type("Edm.Byte")
        assert m.milvus_type == DataType.INT16
        assert m.confidence == MappingConfidence.LOSSLESS

    def test_int16_maps_to_int16(self) -> None:
        m = resolve_type("Edm.Int16")
        assert m.milvus_type == DataType.INT16
        assert m.confidence == MappingConfidence.EXACT

    def test_sbyte_maps_to_int8(self) -> None:
        m = resolve_type("Edm.SByte")
        assert m.milvus_type == DataType.INT8
        assert m.confidence == MappingConfidence.EXACT


class TestCollectionTypeMappings:
    def test_string_collection_maps_to_array(self) -> None:
        m = resolve_type("Collection(Edm.String)")
        assert m.milvus_type == DataType.ARRAY
        assert m.element_type == DataType.VARCHAR

    def test_int32_collection_maps_to_array(self) -> None:
        m = resolve_type("Collection(Edm.Int32)")
        assert m.milvus_type == DataType.ARRAY
        assert m.element_type == DataType.INT32

    def test_int64_collection_maps_to_array(self) -> None:
        m = resolve_type("Collection(Edm.Int64)")
        assert m.milvus_type == DataType.ARRAY
        assert m.element_type == DataType.INT64

    def test_double_collection_maps_to_array(self) -> None:
        m = resolve_type("Collection(Edm.Double)")
        assert m.milvus_type == DataType.ARRAY
        assert m.element_type == DataType.DOUBLE


class TestVectorTypeMappings:
    def test_float32_vector(self) -> None:
        m = resolve_type("Collection(Edm.Single)", is_vector_field=True)
        assert m.milvus_type == DataType.FLOAT_VECTOR
        assert m.is_vector is True
        assert m.confidence == MappingConfidence.EXACT

    def test_float16_vector(self) -> None:
        m = resolve_type("Collection(Edm.Half)", is_vector_field=True)
        assert m.milvus_type == DataType.FLOAT16_VECTOR
        assert m.is_vector is True
        assert m.confidence == MappingConfidence.EXACT

    def test_int16_vector_lossy(self) -> None:
        m = resolve_type("Collection(Edm.Int16)", is_vector_field=True)
        assert m.milvus_type == DataType.FLOAT_VECTOR
        assert m.confidence == MappingConfidence.LOSSY
        assert len(m.warnings) > 0

    def test_int8_vector_lossy(self) -> None:
        m = resolve_type("Collection(Edm.SByte)", is_vector_field=True)
        assert m.milvus_type == DataType.FLOAT_VECTOR
        assert m.confidence == MappingConfidence.LOSSY

    def test_binary_vector(self) -> None:
        m = resolve_type("Collection(Edm.Byte)", is_vector_field=True)
        assert m.milvus_type == DataType.BINARY_VECTOR
        assert m.is_vector is True
        assert m.confidence == MappingConfidence.EXACT

    def test_float32_vector_without_flag(self) -> None:
        """Collection(Edm.Single) should still map to FLOAT_VECTOR even
        without the explicit is_vector_field flag."""
        m = resolve_type("Collection(Edm.Single)")
        assert m.milvus_type == DataType.FLOAT_VECTOR
        assert m.is_vector is True


class TestUnknownType:
    def test_unknown_type_falls_back_to_json(self) -> None:
        m = resolve_type("Edm.Unknown")
        assert m.milvus_type == DataType.JSON
        assert m.confidence == MappingConfidence.UNSUPPORTED
        assert len(m.warnings) > 0


class TestGetAllMappings:
    def test_returns_all_known_mappings(self) -> None:
        all_mappings = get_all_mappings()
        expected_count = (
            len(SCALAR_TYPE_MAP) + len(COLLECTION_TYPE_MAP) + len(VECTOR_TYPE_MAP)
        )
        assert len(all_mappings) == expected_count

    def test_all_mappings_have_edm_type(self) -> None:
        for m in get_all_mappings():
            assert m.edm_type
            assert m.milvus_type is not None
