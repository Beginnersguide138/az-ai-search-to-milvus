"""例: 移行後バリデーション。

Azure AI Search と Milvus 間のドキュメント数、スキーマ構造、
サンプルデータを比較して移行後のデータ整合性を検証する。

使い方:
    python examples/03_validate.py
"""

from rich.console import Console

from az_search_to_milvus.clients.ai_search import AzureSearchClientWrapper
from az_search_to_milvus.clients.milvus import MilvusClientWrapper
from az_search_to_milvus.config import MigrationConfig
from az_search_to_milvus.schema_converter import SchemaConverter
from az_search_to_milvus.utils.logging import setup_logging
from az_search_to_milvus.validation import MigrationValidator

console = Console()


def main() -> None:
    setup_logging(verbose=True)

    config = MigrationConfig.from_yaml("config.yaml")

    # Connect to both sources
    az_client = AzureSearchClientWrapper(config.azure_search)
    index = az_client.get_index()

    converter = SchemaConverter(config.options)
    conversion = converter.convert_from_index(index)

    mv_client = MilvusClientWrapper(config.milvus)
    mv_client.connect()

    try:
        validator = MigrationValidator(az_client, mv_client, conversion)
        report = validator.validate(sample_size=100)

        console.print("[bold]Validation Results[/bold]\n")
        for check in report.checks:
            icon = "[green]✓[/green]" if check.passed else "[red]✗[/red]"
            console.print(f"  {icon} [bold]{check.name}[/bold]: {check.message}")
            if check.expected is not None:
                console.print(f"      Expected: {check.expected}  Actual: {check.actual}")

        console.print()
        console.print(report.summary())

    finally:
        mv_client.disconnect()


if __name__ == "__main__":
    main()
