# Azure AI Search → Milvus Schema Mapping

This document provides a comprehensive mapping of all Azure AI Search field types to Milvus 2.6.x DataTypes.

## Scalar Type Mapping

| Azure AI Search (Edm) | Milvus DataType | Confidence | Notes |
|---|---|---|---|
| `Edm.String` | `VARCHAR` | EXACT | `max_length` required (default 65,535) |
| `Edm.Int32` | `INT32` | EXACT | Exact match |
| `Edm.Int64` | `INT64` | EXACT | Exact match. Can be used as Milvus primary key |
| `Edm.Double` | `DOUBLE` | EXACT | Exact match |
| `Edm.Single` | `FLOAT` | EXACT | Scalar float32. Not a vector |
| `Edm.Boolean` | `BOOL` | EXACT | Exact match |
| `Edm.Int16` | `INT16` | EXACT | Exact match |
| `Edm.SByte` | `INT8` | EXACT | Exact match |
| `Edm.Byte` | `INT16` | LOSSLESS | Upcast since Milvus has no UINT8 |
| `Edm.DateTimeOffset` | `VARCHAR` | SEMANTIC | Stored as ISO 8601 string. Date comparisons become string comparisons |
| `Edm.GeographyPoint` | `JSON` | SEMANTIC | `{"type":"Point","coordinates":[lon,lat]}` format |
| `Edm.ComplexType` | `JSON` | SEMANTIC | Nested complex types are flattened into JSON |

### Confidence Level Definitions

- **EXACT**: Types match exactly. No data conversion needed
- **LOSSLESS**: Safe upcast. No data loss
- **LOSSY**: Possible precision degradation
- **SEMANTIC**: Structurally different but semantically equivalent. Query rewriting may be needed

## Collection (Array) Type Mapping

| Azure AI Search (Edm) | Milvus DataType | Confidence | Notes |
|---|---|---|---|
| `Collection(Edm.String)` | `ARRAY(VARCHAR)` | EXACT | Filterable/facetable string arrays |
| `Collection(Edm.Int32)` | `ARRAY(INT32)` | EXACT | |
| `Collection(Edm.Int64)` | `ARRAY(INT64)` | EXACT | |
| `Collection(Edm.Double)` | `ARRAY(DOUBLE)` | EXACT | |

## Vector Type Mapping

| Azure AI Search (Edm) | Milvus DataType | Confidence | Notes |
|---|---|---|---|
| `Collection(Edm.Single)` | `FLOAT_VECTOR` | EXACT | float32 vectors. Most common |
| `Collection(Edm.Half)` | `FLOAT16_VECTOR` | EXACT | float16 vectors |
| `Collection(Edm.Int16)` | `FLOAT_VECTOR` | LOSSY | No INT16_VECTOR in Milvus; upcast to float32. 2x memory usage |
| `Collection(Edm.SByte)` | `FLOAT_VECTOR` | LOSSY | int8→float32 upcast. 4x memory usage. Re-encoding recommended |
| `Collection(Edm.Byte)` | `BINARY_VECTOR` | EXACT | Binary vectors (packed bits) |

### Notes on Vector Types

**Migrating Int16/SByte Vectors**

Azure AI Search's quantized vector types (`Collection(Edm.Int16)`, `Collection(Edm.SByte)`) have no direct equivalent in Milvus and are upcast to `FLOAT_VECTOR` (float32).

- Memory usage increases (Int16: 2x, SByte: 4x)
- Search accuracy remains nearly identical, but the original quantized representation is lost
- **Recommendation**: Consider re-encoding with the original model to generate float32 vectors directly

## Index Algorithm Mapping

| Azure AI Search | Milvus | Parameter Mapping |
|---|---|---|
| `hnsw` | `HNSW` | `m` → `M`, `efConstruction` → `efConstruction`, `efSearch` → `ef` (at search time) |
| `exhaustiveKnn` | `FLAT` | No parameters (brute force) |

### Distance Metric Mapping

| Azure AI Search | Milvus | Notes |
|---|---|---|
| `cosine` | `COSINE` | Exact match |
| `euclidean` | `L2` | Exact match |
| `dotProduct` | `IP` | Inner Product |
| `hamming` | `HAMMING` | For binary vectors |

## Field Attribute Migration

Many Azure AI Search field attributes are unnecessary in Milvus, but the following correspondences exist:

| Azure Attribute | Milvus Equivalent | Notes |
|---|---|---|
| `key: true` | `is_primary=True` | Primary key. VARCHAR or INT64 only |
| `filterable: true` | (automatic) | Milvus automatically indexes scalar fields |
| `sortable: true` | (automatic) | Scalar fields in Milvus are sortable |
| `searchable: true` | (N/A) | Use Milvus BM25 for full-text search |
| `facetable: true` | (N/A) | No direct equivalent. Use GROUP BY as alternative |
| `retrievable: true` | (automatic) | All fields are retrievable in Milvus |
| `dimensions` | `dim` parameter | Vector dimension count |
| `vectorSearchProfile` | Index definition | Specified separately via `create_index` |

## Unsupported Features (Azure AI Search Specific)

The following features are specific to Azure AI Search and cannot be migrated to Milvus.
The migration tool detects these features and outputs warnings.

| Feature | Description | Alternative |
|---|---|---|
| **Scoring Profiles** | Custom scoring | Implement ranking logic in application layer |
| **Semantic Ranker** | Semantic re-ranking | Integrate a reranker model such as Cross-Encoder |
| **Suggesters** | Autocomplete | Use prefix search or implement in application layer |
| **Skillsets** | AI enrichment pipeline | Build a separate data pipeline |
| **Indexers** | Auto-ingest from data source | Use Kafka/Spark connectors or Milvus CDC |
| **Synonym Maps** | Synonym expansion | Implement query expansion in application layer |
| **Encryption Key (CMK)** | Customer-managed encryption | Use LUKS/dm-crypt or cloud volume encryption |
| **geo.distance()** | Geospatial search | Consider using PostGIS or similar |
