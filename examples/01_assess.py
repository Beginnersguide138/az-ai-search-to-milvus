"""例: 移行前アセスメント。

データに触れることなくスキーマ分析を実行し、互換性レポートを生成する。
これが推奨される最初のステップである。

使い方:
    python examples/01_assess.py
"""

from az_search_to_milvus.assessment import generate_assessment, print_assessment
from az_search_to_milvus.clients.ai_search import AzureSearchClientWrapper
from az_search_to_milvus.config import MigrationConfig
from az_search_to_milvus.schema_converter import SchemaConverter
from az_search_to_milvus.utils.logging import setup_logging


def main() -> None:
    setup_logging(verbose=True)

    # Load configuration
    config = MigrationConfig.from_yaml("config.yaml")

    # Connect to Azure AI Search and retrieve index schema
    az_client = AzureSearchClientWrapper(config.azure_search)
    index = az_client.get_index()
    doc_count = az_client.get_document_count()

    print(f"Index: {index.name}")
    print(f"Document count: {doc_count:,}")
    print(f"Fields: {len(index.fields)}")
    print()

    # Convert schema
    converter = SchemaConverter(config.options)
    conversion = converter.convert_from_index(index)

    # Generate and print assessment report
    report = generate_assessment(conversion, doc_count)
    print_assessment(report)

    # Save JSON report
    report.save_json("assessment_report.json")
    print("Assessment report saved to assessment_report.json")


if __name__ == "__main__":
    main()
