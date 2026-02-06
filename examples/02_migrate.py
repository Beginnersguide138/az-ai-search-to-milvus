"""Example: Full data migration with progress tracking.

Migrates all data from Azure AI Search to Milvus, including:
- Schema conversion and collection creation
- Vector index creation
- Batch data transfer with checkpoint support
- Progress reporting

Usage:
    python examples/02_migrate.py
"""

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from az_search_to_milvus.clients.ai_search import AzureSearchClientWrapper
from az_search_to_milvus.clients.milvus import MilvusClientWrapper
from az_search_to_milvus.config import MigrationConfig
from az_search_to_milvus.data_migrator import DataMigrator
from az_search_to_milvus.schema_converter import SchemaConverter
from az_search_to_milvus.utils.logging import setup_logging

console = Console()


def main() -> None:
    setup_logging(verbose=True)

    # Load configuration
    config = MigrationConfig.from_yaml("config.yaml")

    # Connect to Azure AI Search
    az_client = AzureSearchClientWrapper(config.azure_search)
    index = az_client.get_index()

    # Convert schema
    converter = SchemaConverter(config.options)
    conversion = converter.convert_from_index(index)

    summary = conversion.summary()
    console.print(f"[bold]Index:[/bold] {summary['azure_index']}")
    console.print(f"[bold]Fields:[/bold] {summary['fields_converted']}/{summary['fields_total']}")
    console.print(f"[bold]Vectors:[/bold] {summary['vector_fields']}")
    console.print()

    # Print warnings
    for w in conversion.warnings:
        console.print(f"[yellow]⚠ {w.message}[/yellow]")
    console.print()

    # Connect to Milvus
    mv_client = MilvusClientWrapper(config.milvus)
    mv_client.connect()

    try:
        # Run migration with progress bar
        migrator = DataMigrator(config, conversion, az_client, mv_client)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed:,}/{task.total:,})"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Migrating...", total=None)

            def on_progress(migrated: int, total: int) -> None:
                progress.update(task, completed=migrated, total=total)

            migrator.set_progress_callback(on_progress)
            checkpoint = migrator.migrate()

        console.print()
        console.print(
            f"[green bold]Migration completed:[/green bold] "
            f"{checkpoint.migrated_documents:,} documents"
        )

    finally:
        mv_client.disconnect()


if __name__ == "__main__":
    main()
