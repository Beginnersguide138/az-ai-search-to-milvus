# E2E Testing Guide

Step-by-step instructions for testing the migration tool using the Azure AI Search free tier (F0).

## Prerequisites

| Item | Details |
|------|---------|
| Python | 3.10 or higher |
| Azure Subscription | Free tier is sufficient |
| Azure AI Search Service | F0 (Free) plan |
| Milvus | Docker locally or Zilliz Cloud free tier |

## 1. Create Azure AI Search Free Tier

### Steps in Azure Portal

1. Log in to [Azure Portal](https://portal.azure.com)
2. "Create a resource" → "AI + Machine Learning" → "Azure AI Search"
3. Create with the following settings:
   - **Resource group**: Create new or select existing
   - **Service name**: Any name (e.g., `my-search-test`)
   - **Location**: Choose your nearest region
   - **Pricing tier**: **Free (F0)**
4. "Review + create" → "Create"

### Get Connection Information

After creation, in Azure Portal:

1. Copy the endpoint URL from the AI Search service "Overview"
   - Example: `https://my-search-test.search.windows.net`
2. Copy the admin key from "Settings" → "Keys"

```bash
export AZURE_SEARCH_ENDPOINT="https://my-search-test.search.windows.net"
export AZURE_SEARCH_API_KEY="your-admin-key-here"
```

### Free Tier Limitations

| Item | Limit |
|------|-------|
| Storage | 50 MB |
| Index count | 3 |
| Document count | 10,000 |
| Scaling | Not available (1 partition, 1 replica) |

The test data is designed to fit within ~2 MB with approximately 200 documents and 128-dimensional vectors.

## 2. Milvus Setup

### Local Docker (Recommended)

```bash
# Start Milvus Standalone with Docker Compose
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

Starts by default at `http://localhost:19530`.

### Using Zilliz Cloud Free Tier

1. Create an account at [Zilliz Cloud](https://cloud.zilliz.com/)
2. Create a free cluster
3. Set connection info:

```bash
export MILVUS_URI="https://in01-xxxxxxx.api.gcp-us-west1.zillizcloud.com"
export MILVUS_TOKEN="your-zilliz-api-key"
```

## 3. Running Tests

### Step 1: Generate Test Data

```bash
python examples/test_data_generator.py -o test_data.json
```

Example output:
```
Generating test data... (200 documents)
Estimated data size: 1.53 MB
Complete: test_data.json (200 documents, 1.53 MB)

--- Field Configuration ---
  id: Edm.String (Key)
  title: Edm.String (Searchable)
  description: Edm.String (Searchable)
  category: Edm.String (Filterable, Facetable)
  brand: Edm.String (Filterable, Facetable)
  price: Edm.Double (Filterable, Sortable)
  rating: Edm.Int32 (Filterable, Sortable, Facetable)
  in_stock: Edm.Boolean (Filterable)
  created_at: Edm.DateTimeOffset (Filterable, Sortable)
  tags: Collection(Edm.String) (Filterable, Facetable)
  warehouse_location: Edm.GeographyPoint (Filterable)
  content_vector: Collection(Edm.Single) (128 dimensions, HNSW/cosine)
```

### Step 2: Create Index and Upload Data to Azure AI Search

```bash
python examples/setup_test_index.py -d test_data.json
```

### Step 3: E2E Migration Test

#### Dry Run (No Milvus Required)

```bash
python examples/e2e_migration_test.py --dry-run
```

No Milvus needed. Use this first to verify Azure AI Search connectivity.

#### Assessment Only

```bash
python examples/e2e_migration_test.py --assess-only
```

Review the schema compatibility report.

#### Full Migration Test

```bash
python examples/e2e_migration_test.py
```

Runs all steps: Assessment → Migration → Validation.

### Step 4: Cleanup

After testing, clean up Azure resources:

```bash
# Delete test index
python examples/setup_test_index.py --delete

# Delete checkpoint files
rm -rf .checkpoints_test/
rm -f test_data.json test_assessment_report.json
```

## 4. Test Data Field Type Mapping

List of type conversions verified by the test data:

| Azure AI Search Field | Edm Type | Milvus DataType | Conversion Accuracy |
|----------------------|----------|-----------------|-------------------|
| `id` | Edm.String (Key) | VARCHAR (Primary Key) | EXACT |
| `title` | Edm.String | VARCHAR | EXACT |
| `description` | Edm.String | VARCHAR | EXACT |
| `category` | Edm.String | VARCHAR | EXACT |
| `brand` | Edm.String | VARCHAR | EXACT |
| `price` | Edm.Double | DOUBLE | EXACT |
| `rating` | Edm.Int32 | INT32 | EXACT |
| `in_stock` | Edm.Boolean | BOOL | EXACT |
| `created_at` | Edm.DateTimeOffset | VARCHAR (ISO 8601) | SEMANTIC |
| `tags` | Collection(Edm.String) | ARRAY[VARCHAR] | LOSSLESS |
| `warehouse_location` | Edm.GeographyPoint | JSON | SEMANTIC |
| `content_vector` | Collection(Edm.Single) | FLOAT_VECTOR (128d) | EXACT |

## 5. Troubleshooting

### Azure AI Search Connection Error

```
azure.core.exceptions.HttpResponseError: (403) Forbidden
```

→ Verify your API key is correct. Use the admin key (not query key).

### Milvus Connection Error

```
pymilvus.exceptions.MilvusException: connection refused
```

→ Verify Milvus is running:
```bash
docker ps | grep milvus
```

### Storage Quota Error

```
The index is over the storage quota
```

→ You've exceeded the free tier 50MB limit. Reduce document count:
```bash
python examples/test_data_generator.py -n 100 -o test_data.json
```

### Vector Dimension Mismatch

Verify that the test data vector dimension (128) matches the index definition.
The `VECTOR_DIM` in the test data generator and `vector_search_dimensions` in the
index definition must be the same value.
