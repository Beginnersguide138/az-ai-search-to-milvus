"""Pre-migration assessment report generator.

Analyzes an Azure AI Search index and produces a detailed report showing:
- Schema compatibility analysis
- Estimated Milvus resource requirements
- Feature gap analysis (unsupported features)
- Milvus advantages applicable to this workload
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from az_search_to_milvus.index_mapping import MILVUS_EXCLUSIVE_INDEXES
from az_search_to_milvus.schema_converter import SchemaConversionResult, SchemaConverter
from az_search_to_milvus.type_mapping import (
    MappingConfidence,
    UNSUPPORTED_FEATURES,
    get_all_mappings,
)

logger = logging.getLogger("az_search_to_milvus.assessment")


@dataclass
class AssessmentReport:
    """Complete pre-migration assessment."""

    index_name: str
    generated_at: str = ""
    document_count: int = 0
    # Schema analysis
    total_fields: int = 0
    convertible_fields: int = 0
    lossy_fields: int = 0
    skipped_fields: int = 0
    vector_fields: int = 0
    # Feature gaps
    unsupported_features: list[str] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    # Schema detail
    field_details: list[dict[str, Any]] = field(default_factory=list)
    index_details: list[dict[str, Any]] = field(default_factory=list)
    # Milvus advantages
    applicable_advantages: list[dict[str, str]] = field(default_factory=list)
    # Overall verdict
    migration_feasibility: str = ""  # "full" | "partial" | "complex"

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
        return p


def generate_assessment(
    conversion: SchemaConversionResult,
    document_count: int = 0,
) -> AssessmentReport:
    """Generate an assessment report from a schema conversion result."""
    report = AssessmentReport(
        index_name=conversion.azure_index_name,
        document_count=document_count,
    )

    # Analyze fields
    for fc in conversion.field_conversions:
        detail: dict[str, Any] = {
            "azure_name": fc.azure_name,
            "azure_type": fc.azure_type,
            "milvus_type": fc.milvus_field.dtype.name if fc.milvus_field else "N/A",
            "milvus_name": fc.milvus_field.name if fc.milvus_field else "N/A",
            "confidence": fc.mapping.confidence.value,
            "is_vector": fc.mapping.is_vector,
            "is_primary_key": fc.is_primary_key,
            "skipped": fc.skipped,
            "notes": fc.mapping.notes,
        }
        if fc.renamed_to:
            detail["renamed_to"] = fc.renamed_to

        report.field_details.append(detail)
        report.total_fields += 1

        if fc.skipped:
            report.skipped_fields += 1
        elif fc.mapping.confidence == MappingConfidence.LOSSY:
            report.lossy_fields += 1
            report.convertible_fields += 1
        elif fc.mapping.confidence != MappingConfidence.UNSUPPORTED:
            report.convertible_fields += 1

        if fc.mapping.is_vector:
            report.vector_fields += 1

    # Analyze indexes
    for ic in conversion.index_conversions:
        report.index_details.append({
            "azure_profile": ic.azure_profile_name,
            "azure_algorithm": ic.azure_algorithm_kind,
            "azure_metric": ic.azure_metric,
            "milvus_index_type": ic.milvus_config.index_type,
            "milvus_metric_type": ic.milvus_config.metric_type,
            "milvus_params": ic.milvus_config.params,
            "target_field": ic.target_field,
        })

    # Warnings
    for w in conversion.warnings:
        report.warnings.append({
            "category": w.category,
            "message": w.message,
            "field": w.field_name,
        })

    # Unsupported features
    report.unsupported_features = list(conversion.unsupported_features)

    # Milvus advantages
    report.applicable_advantages = _identify_advantages(conversion)

    # Feasibility verdict
    if report.skipped_fields == 0 and report.lossy_fields == 0:
        report.migration_feasibility = "full"
    elif report.convertible_fields > 0:
        report.migration_feasibility = "partial"
    else:
        report.migration_feasibility = "complex"

    return report


def _identify_advantages(conversion: SchemaConversionResult) -> list[dict[str, str]]:
    """Identify Milvus advantages applicable to this migration."""
    advantages: list[dict[str, str]] = []

    has_vectors = any(fc.mapping.is_vector for fc in conversion.field_conversions)
    has_large_text = any(
        fc.azure_type == "Edm.String"
        and not fc.skipped
        for fc in conversion.field_conversions
    )

    if has_vectors:
        advantages.append({
            "feature": "豊富なインデックスタイプ",
            "description": (
                "Azure AI Search の HNSW/Exhaustive KNN に加えて、"
                "IVF_FLAT, IVF_SQ8, IVF_PQ, SCANN, DiskANN, "
                "GPU_IVF_FLAT, GPU_CAGRA 等を選択可能。"
                "ワークロードに最適なインデックスを選択できます"
            ),
        })
        advantages.append({
            "feature": "GPU アクセラレーション",
            "description": (
                "Azure VM の NC/ND シリーズ (NVIDIA GPU) を使用して "
                "GPU_IVF_FLAT, GPU_CAGRA 等のインデックスで高速検索を実現"
            ),
        })
        advantages.append({
            "feature": "スパースベクトル (ハイブリッド検索)",
            "description": (
                "SPARSE_FLOAT_VECTOR 型で BM25/SPLADE 等のスパース表現をネイティブサポート。"
                "密ベクトル + スパースベクトルのハイブリッド検索が可能"
            ),
        })
        advantages.append({
            "feature": "Range Search",
            "description": (
                "距離の範囲を指定した検索が可能。Top-K だけでなく、"
                "閾値ベースの検索ができます"
            ),
        })
        advantages.append({
            "feature": "Iterator API",
            "description": (
                "メモリ効率の良い大量結果の取得。"
                "Azure AI Search の $skip 100,000 件制限がありません"
            ),
        })

    advantages.append({
        "feature": "パーティションキー (マルチテナンシー)",
        "description": (
            "パーティションキーによるネイティブなマルチテナンシーサポート。"
            "テナントごとのデータ分離と効率的なクエリが可能"
        ),
    })
    advantages.append({
        "feature": "Dynamic Schema",
        "description": (
            "enable_dynamic_field=True により、スキーマに定義されていないフィールドも柔軟に格納。"
            "Azure AI Search のインデックス再構築なしにフィールドを追加可能"
        ),
    })
    advantages.append({
        "feature": "コスト管理",
        "description": (
            "セルフホスト Milvus は Azure VM のコストのみ。"
            "Azure AI Search のクエリ単価課金がなく、大量クエリでもコスト予測が容易"
        ),
    })
    advantages.append({
        "feature": "CDC (Change Data Capture)",
        "description": (
            "Milvus CDC によりデータ変更をリアルタイムにキャプチャ。"
            "DR 構成やデータ同期パイプラインに活用可能"
        ),
    })
    advantages.append({
        "feature": "Null/Default 値サポート (2.6.x)",
        "description": (
            "Milvus 2.6.x では NULL 値とデフォルト値をネイティブサポート。"
            "スキーマの柔軟性が大幅に向上"
        ),
    })

    return advantages


def print_assessment(report: AssessmentReport, console: Console | None = None) -> None:
    """Pretty-print an assessment report to the console."""
    c = console or Console()

    # Header
    c.print(Panel(
        f"[bold]Azure AI Search → Milvus 移行アセスメント[/bold]\n"
        f"インデックス: [cyan]{report.index_name}[/cyan]  |  "
        f"ドキュメント数: [cyan]{report.document_count:,}[/cyan]  |  "
        f"生成日時: {report.generated_at}",
        title="Assessment Report",
    ))

    # Feasibility
    feas_color = {"full": "green", "partial": "yellow", "complex": "red"}.get(
        report.migration_feasibility, "white"
    )
    c.print(f"\n移行可能性: [{feas_color} bold]{report.migration_feasibility.upper()}[/{feas_color} bold]")
    c.print(
        f"  フィールド: {report.convertible_fields}/{report.total_fields} 変換可能"
        f" ({report.lossy_fields} 損失あり, {report.skipped_fields} スキップ)"
    )

    # Field mapping table
    c.print("\n")
    field_table = Table(title="フィールドマッピング")
    field_table.add_column("Azure フィールド", style="cyan")
    field_table.add_column("Azure 型", style="blue")
    field_table.add_column("Milvus 型", style="green")
    field_table.add_column("信頼度", style="yellow")
    field_table.add_column("備考")

    for fd in report.field_details:
        conf = fd["confidence"]
        conf_style = {
            "exact": "[green]EXACT[/green]",
            "lossless": "[green]LOSSLESS[/green]",
            "lossy": "[yellow]LOSSY[/yellow]",
            "semantic": "[blue]SEMANTIC[/blue]",
            "unsupported": "[red]N/A[/red]",
        }.get(conf, conf)

        status = ""
        if fd.get("skipped"):
            status = "[dim]SKIP[/dim]"
        elif fd.get("is_primary_key"):
            status = "[bold]PK[/bold] "
        elif fd.get("is_vector"):
            status = "[magenta]VEC[/magenta] "

        field_table.add_row(
            f"{status}{fd['azure_name']}",
            fd["azure_type"],
            fd["milvus_type"],
            conf_style,
            fd.get("notes", "")[:60],
        )

    c.print(field_table)

    # Index mapping table
    if report.index_details:
        c.print("\n")
        idx_table = Table(title="インデックスマッピング")
        idx_table.add_column("Azure プロファイル")
        idx_table.add_column("Azure アルゴリズム")
        idx_table.add_column("メトリック")
        idx_table.add_column("→ Milvus インデックス", style="green")
        idx_table.add_column("パラメータ")

        for idx in report.index_details:
            idx_table.add_row(
                idx["azure_profile"],
                idx["azure_algorithm"],
                idx["azure_metric"],
                f"{idx['milvus_index_type']} ({idx['milvus_metric_type']})",
                str(idx["milvus_params"]),
            )
        c.print(idx_table)

    # Warnings
    if report.warnings:
        c.print("\n[yellow bold]⚠ 警告[/yellow bold]")
        for w in report.warnings:
            field_info = f" [{w['field']}]" if w.get("field") else ""
            c.print(f"  • {w['category']}{field_info}: {w['message']}")

    # Unsupported features
    if report.unsupported_features:
        c.print("\n[red bold]✗ 非対応機能 (Azure AI Search 固有)[/red bold]")
        for feat in report.unsupported_features:
            desc = UNSUPPORTED_FEATURES.get(feat, feat)
            c.print(f"  • [red]{feat}[/red]: {desc}")

    # Milvus advantages
    if report.applicable_advantages:
        c.print("\n[green bold]✓ Milvus への移行で得られるメリット[/green bold]")
        for adv in report.applicable_advantages:
            c.print(f"  • [green]{adv['feature']}[/green]: {adv['description']}")

    c.print()
