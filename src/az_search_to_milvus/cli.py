"""CLI interface for the Azure AI Search → Milvus migration tool.

Usage:
    az-search-to-milvus assess   --config config.yaml
    az-search-to-milvus migrate  --config config.yaml
    az-search-to-milvus validate --config config.yaml
    az-search-to-milvus schema   --config config.yaml --output schema.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from az_search_to_milvus import __version__
from az_search_to_milvus.config import MigrationConfig
from az_search_to_milvus.utils.logging import setup_logging

console = Console()


def _load_config(config_path: str) -> MigrationConfig:
    """Load and validate configuration."""
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]設定ファイルが見つかりません: {path}[/red]")
        sys.exit(1)
    return MigrationConfig.from_yaml(path)


@click.group()
@click.version_option(__version__)
@click.option("--verbose", "-v", is_flag=True, help="詳細ログを出力")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Azure AI Search → Milvus/Zilliz 移行ツール"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose=verbose)


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(), help="設定ファイルパス (YAML)")
@click.option("--output", "-o", type=click.Path(), help="レポート出力先 (JSON)")
@click.pass_context
def assess(ctx: click.Context, config: str, output: str | None) -> None:
    """移行前アセスメントを実行

    Azure AI Search インデックスのスキーマを分析し、Milvus への移行可能性、
    型マッピング、非対応機能、Milvus のメリットをレポートします。
    """
    from az_search_to_milvus.assessment import generate_assessment, print_assessment
    from az_search_to_milvus.clients.ai_search import AzureSearchClientWrapper
    from az_search_to_milvus.schema_converter import SchemaConverter

    cfg = _load_config(config)

    console.print("[bold]アセスメント開始...[/bold]\n")

    # Connect to Azure AI Search
    az_client = AzureSearchClientWrapper(cfg.azure_search)
    index = az_client.get_index()
    doc_count = az_client.get_document_count()

    # Convert schema
    converter = SchemaConverter(cfg.options)
    conversion = converter.convert_from_index(index)

    # Generate report
    report = generate_assessment(conversion, doc_count)
    print_assessment(report, console)

    if output:
        path = report.save_json(output)
        console.print(f"[green]レポートを保存しました: {path}[/green]")


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(), help="設定ファイルパス (YAML)")
@click.option("--dry-run", is_flag=True, help="実際の書き込みを行わずにシミュレーション")
@click.option("--drop-existing", is_flag=True, help="既存のコレクションを削除して再作成")
@click.option("--resume/--no-resume", default=True, help="チェックポイントからの再開 (デフォルト: 有効)")
@click.pass_context
def migrate(
    ctx: click.Context,
    config: str,
    dry_run: bool,
    drop_existing: bool,
    resume: bool,
) -> None:
    """データ移行を実行

    Azure AI Search のインデックスデータを Milvus コレクションに移行します。
    バッチ処理とチェックポイントによる再開をサポートします。
    """
    from az_search_to_milvus.clients.ai_search import AzureSearchClientWrapper
    from az_search_to_milvus.clients.milvus import MilvusClientWrapper
    from az_search_to_milvus.data_migrator import DataMigrator
    from az_search_to_milvus.schema_converter import SchemaConverter
    from az_search_to_milvus.utils.checkpoint import CheckpointManager

    cfg = _load_config(config)
    if dry_run:
        cfg.options.dry_run = True
    if drop_existing:
        cfg.options.drop_existing_collection = True

    # Delete checkpoint if --no-resume
    if not resume:
        cm = CheckpointManager(cfg.options.checkpoint_dir)
        cm.delete(cfg.azure_search.index_name)

    console.print("[bold]移行開始...[/bold]\n")

    # Connect to Azure AI Search
    az_client = AzureSearchClientWrapper(cfg.azure_search)
    index = az_client.get_index()

    # Convert schema
    converter = SchemaConverter(cfg.options)
    conversion = converter.convert_from_index(index)

    # Print schema summary
    summary = conversion.summary()
    console.print(f"  インデックス: [cyan]{summary['azure_index']}[/cyan]")
    console.print(f"  フィールド: {summary['fields_converted']} 変換 / {summary['fields_total']} 合計")
    console.print(f"  ベクトルフィールド: {summary['vector_fields']}")
    if summary['warnings']:
        console.print(f"  [yellow]警告: {summary['warnings']} 件[/yellow]")
    console.print()

    # Connect to Milvus
    mv_client = MilvusClientWrapper(cfg.milvus)
    mv_client.connect()

    try:
        # Run migration with progress bar
        migrator = DataMigrator(cfg, conversion, az_client, mv_client)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("移行中...", total=None)

            def update_progress(migrated: int, total: int) -> None:
                progress.update(task_id, completed=migrated, total=total)

            migrator.set_progress_callback(update_progress)
            checkpoint = migrator.migrate()

        # Print result
        console.print()
        if checkpoint.status == "completed":
            console.print(
                f"[green bold]移行完了[/green bold]: "
                f"{checkpoint.migrated_documents:,} ドキュメント"
            )
        else:
            console.print(
                f"[red bold]移行失敗[/red bold]: {checkpoint.error_message}"
            )
            sys.exit(1)

    finally:
        mv_client.disconnect()


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(), help="設定ファイルパス (YAML)")
@click.option("--sample-size", default=100, help="サンプルチェックのドキュメント数")
@click.pass_context
def validate(ctx: click.Context, config: str, sample_size: int) -> None:
    """移行後のデータ整合性を検証

    ドキュメント数、フィールド数、サンプルデータの整合性をチェックします。
    """
    from az_search_to_milvus.clients.ai_search import AzureSearchClientWrapper
    from az_search_to_milvus.clients.milvus import MilvusClientWrapper
    from az_search_to_milvus.schema_converter import SchemaConverter
    from az_search_to_milvus.validation import MigrationValidator

    cfg = _load_config(config)

    console.print("[bold]バリデーション開始...[/bold]\n")

    az_client = AzureSearchClientWrapper(cfg.azure_search)
    index = az_client.get_index()

    converter = SchemaConverter(cfg.options)
    conversion = converter.convert_from_index(index)

    mv_client = MilvusClientWrapper(cfg.milvus)
    mv_client.connect()

    try:
        validator = MigrationValidator(az_client, mv_client, conversion)
        report = validator.validate(sample_size=sample_size)

        for check in report.checks:
            icon = "[green]✓[/green]" if check.passed else "[red]✗[/red]"
            console.print(f"  {icon} {check.name}: {check.message}")

        console.print()
        if report.all_passed:
            console.print("[green bold]全チェック合格[/green bold]")
        else:
            console.print(f"[red bold]{report.fail_count} 件のチェックが失敗[/red bold]")
            sys.exit(1)

    finally:
        mv_client.disconnect()


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(), help="設定ファイルパス (YAML)")
@click.option("--output", "-o", type=click.Path(), help="スキーマ出力先 (JSON)")
@click.option("--from-json", type=click.Path(), help="Azure REST API エクスポートの JSON から変換")
@click.pass_context
def schema(ctx: click.Context, config: str, output: str | None, from_json: str | None) -> None:
    """スキーマ変換のみ実行 (データ移行なし)

    Azure AI Search のスキーマを Milvus スキーマに変換し、結果を表示します。
    """
    from az_search_to_milvus.assessment import print_assessment, generate_assessment
    from az_search_to_milvus.schema_converter import SchemaConverter

    cfg = _load_config(config)
    converter = SchemaConverter(cfg.options)

    if from_json:
        with open(from_json) as f:
            index_json = json.load(f)
        conversion = converter.convert_from_json(index_json)
    else:
        from az_search_to_milvus.clients.ai_search import AzureSearchClientWrapper
        az_client = AzureSearchClientWrapper(cfg.azure_search)
        index = az_client.get_index()
        conversion = converter.convert_from_index(index)

    report = generate_assessment(conversion)
    print_assessment(report, console)

    if output:
        out_path = Path(output)
        schema_dict = {
            "azure_index": conversion.azure_index_name,
            "milvus_collection": conversion.milvus_collection_name,
            "fields": [
                {
                    "azure_name": fc.azure_name,
                    "azure_type": fc.azure_type,
                    "milvus_name": fc.milvus_field.name if fc.milvus_field else None,
                    "milvus_type": fc.milvus_field.dtype.name if fc.milvus_field else None,
                    "params": fc.milvus_field.params if fc.milvus_field else {},
                    "skipped": fc.skipped,
                    "confidence": fc.mapping.confidence.value,
                }
                for fc in conversion.field_conversions
            ],
            "indexes": [
                {
                    "field": ic.target_field,
                    "index_type": ic.milvus_config.index_type,
                    "metric_type": ic.milvus_config.metric_type,
                    "params": ic.milvus_config.params,
                }
                for ic in conversion.index_conversions
            ],
        }
        out_path.write_text(json.dumps(schema_dict, indent=2, ensure_ascii=False))
        console.print(f"\n[green]スキーマを保存しました: {out_path}[/green]")
