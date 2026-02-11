🌐 [日本語](README.md) | [English](README.en.md) | [中文](README.zh.md)

# az-ai-search-to-milvus

**Azure AI Search to Milvus/Zilliz Migration Tool**

A Python tool that automates data migration from Azure AI Search (formerly Azure Cognitive Search) to self-hosted Milvus or Zilliz Cloud. Like AWS Schema Conversion Tool (SCT), it provides schema conversion, data migration, and validation in a single package.

## Features

- **Comprehensive Schema Conversion** — Mapping of all Azure AI Search Edm types to Milvus 2.6.x DataTypes
- **Vector Index Migration** — 1:1 parameter mapping for HNSW / Exhaustive KNN
- **Batch Data Migration** — Resumable batch processing with checkpoint support
- **Pre-Migration Assessment** — Compatibility analysis, unsupported feature detection, and Milvus benefit overview
- **Post-Migration Validation** — Document count and field value consistency verification
- **Zilliz Cloud Support** — Supports both self-hosted Milvus and Zilliz Cloud
- **CLI + Library** — Usable as both a command-line tool and a Python library

## Quick Start

### Installation

```bash
git clone https://github.com/Beginnersguide138/az-ai-search-to-milvus.git
cd az-ai-search-to-milvus
pip install -e .
```

### Configuration

```bash
cp examples/config.example.yaml config.yaml
# Edit config.yaml to set your connection details
```

```yaml
azure_search:
  endpoint: "https://your-service.search.windows.net"
  index_name: "your-index"
  api_key: ""  # Using the environment variable AZURE_SEARCH_API_KEY is recommended

milvus:
  uri: "http://localhost:19530"
  collection_name: "your_collection"

options:
  batch_size: 500
  enable_dynamic_field: true
```

### Execution

```bash
# Step 1: Assessment (schema analysis + compatibility report)
az-search-to-milvus assess --config config.yaml

# Step 2: Data migration
az-search-to-milvus migrate --config config.yaml

# Step 3: Validation
az-search-to-milvus validate --config config.yaml
```

## Schema Mapping

### Scalar Types

| Azure AI Search (Edm) | Milvus DataType | Confidence |
|---|---|---|
| `Edm.String` | `VARCHAR` | EXACT |
| `Edm.Int32` | `INT32` | EXACT |
| `Edm.Int64` | `INT64` | EXACT |
| `Edm.Double` | `DOUBLE` | EXACT |
| `Edm.Single` | `FLOAT` | EXACT |
| `Edm.Boolean` | `BOOL` | EXACT |
| `Edm.Int16` | `INT16` | EXACT |
| `Edm.SByte` | `INT8` | EXACT |
| `Edm.Byte` | `INT16` | LOSSLESS |
| `Edm.DateTimeOffset` | `VARCHAR` | SEMANTIC |
| `Edm.GeographyPoint` | `JSON` | SEMANTIC |
| `Edm.ComplexType` | `JSON` | SEMANTIC |

### Vector Types

| Azure AI Search (Edm) | Milvus DataType | Confidence |
|---|---|---|
| `Collection(Edm.Single)` | `FLOAT_VECTOR` | EXACT |
| `Collection(Edm.Half)` | `FLOAT16_VECTOR` | EXACT |
| `Collection(Edm.Byte)` | `BINARY_VECTOR` | EXACT |
| `Collection(Edm.Int16)` | `FLOAT_VECTOR` | LOSSY |
| `Collection(Edm.SByte)` | `FLOAT_VECTOR` | LOSSY |

### Collection (Array) Types

| Azure AI Search (Edm) | Milvus DataType | Confidence |
|---|---|---|
| `Collection(Edm.String)` | `ARRAY(VARCHAR)` | EXACT |
| `Collection(Edm.Int32)` | `ARRAY(INT32)` | EXACT |
| `Collection(Edm.Int64)` | `ARRAY(INT64)` | EXACT |
| `Collection(Edm.Double)` | `ARRAY(DOUBLE)` | EXACT |

### Index Algorithms

| Azure AI Search | Milvus | Metric |
|---|---|---|
| `hnsw` | `HNSW` | cosine→COSINE, euclidean→L2, dotProduct→IP |
| `exhaustiveKnn` | `FLAT` | Same as above |

> For details, see [docs/en/schema_mapping.md](docs/en/schema_mapping.md)

## Benefits of Migrating to Milvus

Milvus-specific features not available in Azure AI Search:

| Feature | Description |
|---|---|
| **12+ Index Types** | IVF_FLAT, IVF_SQ8, IVF_PQ, SCANN, DiskANN, GPU_CAGRA, etc. |
| **GPU Acceleration** | High-speed search with GPU_IVF_FLAT and GPU_CAGRA on Azure NC/ND VMs |
| **Hybrid Search** | Native dense + sparse vector search via SPARSE_FLOAT_VECTOR |
| **Partition Key** | Native multi-tenancy support |
| **Dynamic Schema** | Flexible storage of fields outside the schema |
| **Range Search** | Distance threshold-based search |
| **Grouping Search** | Search with grouping by field |
| **Iterator API** | Large-scale data retrieval without the $skip 100K limit |
| **CDC** | Data synchronization via Change Data Capture |
| **Cost Management** | VM costs only, no per-query charges |

> For details, see [docs/en/milvus_advantages.md](docs/en/milvus_advantages.md)

## Unsupported Features (Azure AI Search-Specific)

The following features are not included in the migration scope. The tool detects these and outputs warnings:

- Scoring Profiles — Implement ranking logic in the application layer
- Semantic Ranker — Integrate a reranker model such as Cross-Encoder
- Suggesters — Implement via prefix search or in the application layer
- Skillsets / Indexers — Build a separate data pipeline
- Synonym Maps — Implement query expansion in the application layer
- geo.distance() — Consider using PostGIS or a similar solution

## CLI Commands

```
az-search-to-milvus [OPTIONS] COMMAND [ARGS]...

Commands:
  assess    Run pre-migration assessment
  migrate   Run data migration
  validate  Verify post-migration data consistency
  schema    Run schema conversion only (no data migration)

Options:
  --version  Show the version and exit.
  -v         Enable verbose logging
```

### assess

```bash
az-search-to-milvus assess --config config.yaml [--output report.json]
```

### migrate

```bash
az-search-to-milvus migrate --config config.yaml [--dry-run] [--drop-existing] [--no-resume]
```

### validate

```bash
az-search-to-milvus validate --config config.yaml [--sample-size 100]
```

### schema

```bash
# Via SDK
az-search-to-milvus schema --config config.yaml [--output schema.json]

# From REST API JSON (no Azure connection required)
az-search-to-milvus schema --config config.yaml --from-json index.json
```

## Project Structure

```
az-ai-search-to-milvus/
├── src/az_search_to_milvus/
│   ├── type_mapping.py        # Edm → Milvus type mapping
│   ├── index_mapping.py       # Vector index mapping
│   ├── schema_converter.py    # Schema conversion engine
│   ├── data_migrator.py       # Data migration engine
│   ├── assessment.py          # Pre-migration assessment
│   ├── validation.py          # Post-migration validation
│   ├── config.py              # Configuration model
│   ├── cli.py                 # CLI interface
│   ├── clients/
│   │   ├── ai_search.py       # Azure AI Search client
│   │   └── milvus.py          # Milvus client
│   └── utils/
│       ├── logging.py         # Logging configuration
│       └── checkpoint.py      # Checkpoint management
├── examples/
│   ├── config.example.yaml    # Configuration file template
│   ├── 01_assess.py           # Assessment example
│   ├── 02_migrate.py          # Migration example
│   ├── 03_validate.py         # Validation example
│   └── 04_schema_from_json.py # Schema conversion from JSON example
├── docs/
│   ├── schema_mapping.md      # Comprehensive schema mapping (Japanese)
│   ├── migration_guide.md     # Step-by-step guide (Japanese)
│   ├── milvus_advantages.md   # Milvus benefits guide (Japanese)
│   ├── testing_guide.md       # Testing guide (Japanese)
│   ├── en/                    # English documentation
│   │   ├── schema_mapping.md
│   │   ├── migration_guide.md
│   │   ├── milvus_advantages.md
│   │   └── testing_guide.md
│   └── zh/                    # Chinese documentation
│       ├── schema_mapping.md
│       ├── migration_guide.md
│       ├── milvus_advantages.md
│       └── testing_guide.md
├── tests/                     # Unit tests (83 tests)
└── pyproject.toml
```

## Supported Environments

- **Python**: 3.10+
- **Azure AI Search**: SDK 11.6.0+
- **Milvus**: 2.5.x - 2.6.x
- **Recommended Runtime**: Azure VM (same VNet recommended)
- **Zilliz Cloud**: Supported

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## Documentation

- [Schema Mapping (Comprehensive)](docs/en/schema_mapping.md)
- [Migration Guide (Step-by-Step)](docs/en/migration_guide.md)
- [Milvus Benefits Guide](docs/en/milvus_advantages.md)
- [Testing Guide](docs/en/testing_guide.md)

## License

MIT
