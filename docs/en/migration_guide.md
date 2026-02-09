# Azure AI Search → Milvus Migration Guide

This document provides step-by-step instructions for migrating data from
Azure AI Search to Milvus on an Azure VM.

## Prerequisites

- Azure VM (recommended: Standard_D8s_v5 or higher)
- Python 3.10+
- Azure AI Search endpoint and API key (or Entra ID authentication)
- Milvus 2.6.x (same VNet recommended) or Zilliz Cloud account

## Step 1: Environment Setup

### 1.1 Deploy Milvus (Self-Hosted)

```bash
# Deploy Milvus with Docker Compose
wget https://github.com/milvus-io/milvus/releases/download/v2.6.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start
docker compose up -d

# Verify
docker compose ps
```

### 1.2 Install Migration Tool

```bash
# Clone repository
git clone https://github.com/Beginnersguide138/az-ai-search-to-milvus.git
cd az-ai-search-to-milvus

# Create virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 1.3 Prepare Configuration

```bash
cp examples/config.example.yaml config.yaml
```

Edit `config.yaml`:

```yaml
azure_search:
  endpoint: "https://your-search-service.search.windows.net"
  index_name: "your-index-name"
  api_key: ""  # Recommended: use env var AZURE_SEARCH_API_KEY
  use_entra_id: false

milvus:
  uri: "http://localhost:19530"
  collection_name: "your_collection_name"
  # For Zilliz Cloud
  # use_zilliz: true
  # zilliz_endpoint: "https://xxx.zillizcloud.com"
  # zilliz_api_key: ""

options:
  batch_size: 500
  drop_existing_collection: false
  enable_dynamic_field: true
  # partition_key_field: "tenant_id"  # For multi-tenancy
```

Set environment variables:

```bash
export AZURE_SEARCH_API_KEY="your-api-key"
# For Zilliz Cloud
# export ZILLIZ_API_KEY="your-zilliz-api-key"
```

## Step 2: Assessment (Pre-Migration Analysis)

```bash
az-search-to-milvus assess --config config.yaml --output assessment.json
```

This command outputs:

- Schema mapping table (type mapping for all fields)
- Migration feasibility assessment (FULL / PARTIAL / COMPLEX)
- Warnings for unsupported features (Scoring Profiles, Semantic Ranker, etc.)
- Benefits of migrating to Milvus
- JSON report (for use in CI/CD pipelines)

### Reading Assessment Results

| Assessment | Meaning |
|---|---|
| **FULL** | All fields can be converted without loss |
| **PARTIAL** | Some fields require lossy conversion (possible precision loss) |
| **COMPLEX** | Some fields require manual intervention |

## Step 3: Schema Conversion (Optional)

To review schema conversion before data migration:

```bash
# Via SDK
az-search-to-milvus schema --config config.yaml --output schema.json

# From REST API export JSON
az-search-to-milvus schema --config config.yaml --from-json index_definition.json
```

### Field Customization

Modify per-field settings via `field_overrides` in `config.yaml`:

```yaml
options:
  field_overrides:
    content:
      milvus_name: "text_content"  # Rename field
      max_length: 32768            # VARCHAR max length
    category:
      milvus_name: "category"
  exclude_fields:
    - "internal_field"  # Exclude unnecessary fields
```

## Step 4: Data Migration

```bash
# Dry run (no actual writes)
az-search-to-milvus migrate --config config.yaml --dry-run

# Production migration
az-search-to-milvus migrate --config config.yaml

# Drop existing collection and recreate
az-search-to-milvus migrate --config config.yaml --drop-existing
```

### Migration Workflow

1. Extract documents from Azure AI Search in batches
2. Transform each document to Milvus schema
3. Batch insert into Milvus
4. Save checkpoint (for resuming after failures)
5. Build vector indexes
6. Load collection into memory

### Checkpoint-Based Resumption

If migration is interrupted, re-running the same command automatically resumes from the last checkpoint:

```bash
# Re-run after interruption → resumes from checkpoint
az-search-to-milvus migrate --config config.yaml

# Ignore checkpoint and start from scratch
az-search-to-milvus migrate --config config.yaml --no-resume
```

## Step 5: Validation

```bash
az-search-to-milvus validate --config config.yaml
```

Validation checks:

- **Document count**: Whether Azure and Milvus document counts match
- **Field count**: Whether Milvus collection field count matches expectations
- **Sample data**: Whether field values in random samples were correctly migrated
- **Vector dimensions**: Whether vector field dimensions are correct

## Step 6: Application Switchover

### 6.1 Migrating to Milvus Client

Example of rewriting from Azure AI Search SDK to PyMilvus:

**Before (Azure AI Search)**:
```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

client = SearchClient(
    endpoint="https://xxx.search.windows.net",
    index_name="my-index",
    credential=AzureKeyCredential("key"),
)

results = client.search(
    search_text=None,
    vector_queries=[VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=10,
        fields="embedding",
    )],
    filter="category eq 'tech'",
)
```

**After (Milvus)**:
```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

results = client.search(
    collection_name="my_collection",
    data=[query_embedding],
    anns_field="embedding",
    limit=10,
    filter='category == "tech"',
    output_fields=["title", "content", "category"],
    search_params={"metric_type": "COSINE", "params": {"ef": 100}},
)
```

### 6.2 Filter Syntax Conversion

| Azure AI Search (OData) | Milvus | Notes |
|---|---|---|
| `field eq 'value'` | `field == "value"` | Equals |
| `field ne 'value'` | `field != "value"` | Not equals |
| `field gt 10` | `field > 10` | Greater than |
| `field ge 10` | `field >= 10` | Greater than or equal |
| `field lt 10` | `field < 10` | Less than |
| `field le 10` | `field <= 10` | Less than or equal |
| `field eq true` | `field == true` | Boolean |
| `search.in(field, 'a,b,c')` | `field in ["a", "b", "c"]` | IN clause |
| `field/any(t: t eq 'x')` | `array_contains(field, "x")` | Array search |
| `f1 eq 'a' and f2 gt 5` | `f1 == "a" and f2 > 5` | AND |
| `f1 eq 'a' or f2 gt 5` | `f1 == "a" or f2 > 5` | OR |

## Troubleshooting

### Q: Migration speed is slow

- Increase `batch_size` (max 1000)
- Verify Azure VM and Milvus are in the same VNet / region
- Check Milvus node resources (CPU/memory)

### Q: Out of memory error

- Decrease `batch_size`
- For high-dimensional vectors, consider using DiskANN index

### Q: Checkpoint is corrupted

```bash
# Delete checkpoint and start from scratch
rm -rf .checkpoints/
az-search-to-milvus migrate --config config.yaml --no-resume
```

### Q: Connection error to Zilliz Cloud

- Verify `zilliz_endpoint` is correct
- Check API key permissions
- Check network (firewall / NSG)
