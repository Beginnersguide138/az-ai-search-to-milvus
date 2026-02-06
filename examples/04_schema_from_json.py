"""例: JSON エクスポートからのスキーマ変換。

REST API 経由でエクスポートした Azure AI Search インデックス定義を
Azure に接続せずに Milvus コレクションスキーマへ変換する。

以下の用途に便利:
- インデックススキーマのオフライン分析
- CI/CD パイプライン
- 本番サービスにアクセスする前のスキーマ変換テスト

使い方:
    python examples/04_schema_from_json.py
"""

import json

from rich.console import Console

from az_search_to_milvus.assessment import generate_assessment, print_assessment
from az_search_to_milvus.config import MigrationOptions
from az_search_to_milvus.schema_converter import SchemaConverter

console = Console()

# Azure AI Search インデックス定義の例 (JSON 形式)
SAMPLE_INDEX_JSON = {
    "name": "products-index",
    "fields": [
        {
            "name": "id",
            "type": "Edm.String",
            "key": True,
            "filterable": True,
        },
        {
            "name": "title",
            "type": "Edm.String",
            "searchable": True,
            "filterable": True,
            "sortable": True,
        },
        {
            "name": "description",
            "type": "Edm.String",
            "searchable": True,
        },
        {
            "name": "price",
            "type": "Edm.Double",
            "filterable": True,
            "sortable": True,
            "facetable": True,
        },
        {
            "name": "category",
            "type": "Edm.String",
            "filterable": True,
            "facetable": True,
        },
        {
            "name": "tags",
            "type": "Collection(Edm.String)",
            "filterable": True,
            "facetable": True,
        },
        {
            "name": "rating",
            "type": "Edm.Int32",
            "filterable": True,
            "sortable": True,
        },
        {
            "name": "created_at",
            "type": "Edm.DateTimeOffset",
            "filterable": True,
            "sortable": True,
        },
        {
            "name": "location",
            "type": "Edm.GeographyPoint",
            "filterable": True,
        },
        {
            "name": "embedding",
            "type": "Collection(Edm.Single)",
            "dimensions": 1536,
            "vectorSearchProfile": "embedding-profile",
        },
        {
            "name": "image_embedding",
            "type": "Collection(Edm.Single)",
            "dimensions": 512,
            "vectorSearchProfile": "image-profile",
        },
    ],
    "vectorSearch": {
        "algorithms": [
            {
                "name": "hnsw-algo",
                "kind": "hnsw",
                "hnswParameters": {
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine",
                },
            },
        ],
        "profiles": [
            {
                "name": "embedding-profile",
                "algorithmConfigurationName": "hnsw-algo",
            },
            {
                "name": "image-profile",
                "algorithmConfigurationName": "hnsw-algo",
            },
        ],
    },
    "scoringProfiles": [
        {
            "name": "boost-recent",
            "functions": [
                {
                    "type": "freshness",
                    "fieldName": "created_at",
                    "boost": 2,
                }
            ],
        }
    ],
    "suggesters": [
        {
            "name": "title-suggester",
            "searchMode": "analyzingInfixMatching",
            "sourceFields": ["title"],
        }
    ],
}


def main() -> None:
    console.print("[bold]Azure AI Search Index JSON → Milvus Schema Conversion[/bold]\n")

    # Convert from JSON (no Azure connection needed)
    options = MigrationOptions(
        enable_dynamic_field=True,
        partition_key_field="category",  # Use category as partition key
    )
    converter = SchemaConverter(options)
    conversion = converter.convert_from_json(SAMPLE_INDEX_JSON)

    # Print assessment
    report = generate_assessment(conversion, document_count=50_000)
    print_assessment(report, console)

    # Export Milvus schema as JSON
    schema_output = {
        "collection_name": conversion.milvus_collection_name,
        "fields": [],
        "indexes": [],
    }

    for fc in conversion.field_conversions:
        if fc.skipped or not fc.milvus_field:
            continue
        schema_output["fields"].append({
            "name": fc.milvus_field.name,
            "dtype": fc.milvus_field.dtype.name,
            "params": fc.milvus_field.params,
            "is_primary": getattr(fc.milvus_field, "is_primary", False),
        })

    for ic in conversion.index_conversions:
        schema_output["indexes"].append({
            "field": ic.target_field,
            "type": ic.milvus_config.index_type,
            "metric": ic.milvus_config.metric_type,
            "params": ic.milvus_config.params,
        })

    console.print("\n[bold]Milvus Schema (JSON):[/bold]")
    console.print(json.dumps(schema_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
