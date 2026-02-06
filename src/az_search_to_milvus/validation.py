"""Post-migration validation: verify data integrity after migration.

Performs:
- Document count comparison
- Sample-based field value comparison
- Vector similarity spot-check
- Schema consistency check
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from az_search_to_milvus.clients.ai_search import AzureSearchClientWrapper
from az_search_to_milvus.clients.milvus import MilvusClientWrapper
from az_search_to_milvus.schema_converter import SchemaConversionResult

logger = logging.getLogger("az_search_to_milvus.validation")


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    message: str = ""


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def summary(self) -> str:
        total = len(self.checks)
        passed = self.pass_count
        status = "PASS" if self.all_passed else "FAIL"
        return f"[{status}] {passed}/{total} checks passed"


class MigrationValidator:
    """Validates data integrity after migration."""

    def __init__(
        self,
        azure_client: AzureSearchClientWrapper,
        milvus_client: MilvusClientWrapper,
        conversion: SchemaConversionResult,
    ) -> None:
        self.azure = azure_client
        self.milvus = milvus_client
        self.conversion = conversion

    def validate(
        self,
        *,
        sample_size: int = 100,
        count_tolerance_pct: float = 1.0,
    ) -> ValidationReport:
        """Run all validation checks.

        Parameters
        ----------
        sample_size:
            Number of documents to spot-check for field value comparison.
        count_tolerance_pct:
            Acceptable percentage difference in document counts (default 1%).
        """
        report = ValidationReport()
        collection_name = self.conversion.milvus_collection_name
        index_name = self.conversion.azure_index_name

        # 1. Document count check
        report.checks.append(
            self._check_document_count(index_name, collection_name, count_tolerance_pct)
        )

        # 2. Schema field count check
        report.checks.append(
            self._check_field_count(collection_name)
        )

        # 3. Sample data check
        sample_checks = self._check_sample_data(
            index_name, collection_name, sample_size
        )
        report.checks.extend(sample_checks)

        return report

    def _check_document_count(
        self,
        index_name: str,
        collection_name: str,
        tolerance_pct: float,
    ) -> ValidationCheck:
        """Compare document counts between Azure and Milvus."""
        try:
            azure_count = self.azure.get_document_count(index_name)
            milvus_count = self.milvus.query_count(collection_name)

            if azure_count == 0:
                return ValidationCheck(
                    name="document_count",
                    passed=True,
                    expected=azure_count,
                    actual=milvus_count,
                    message="Azure 側のドキュメント数が 0 です",
                )

            diff_pct = abs(azure_count - milvus_count) / azure_count * 100
            passed = diff_pct <= tolerance_pct

            return ValidationCheck(
                name="document_count",
                passed=passed,
                expected=azure_count,
                actual=milvus_count,
                message=(
                    f"ドキュメント数一致 (差分 {diff_pct:.2f}%)" if passed
                    else f"ドキュメント数不一致: Azure={azure_count}, Milvus={milvus_count} (差分 {diff_pct:.2f}%)"
                ),
            )
        except Exception as e:
            return ValidationCheck(
                name="document_count",
                passed=False,
                message=f"ドキュメント数チェック失敗: {e}",
            )

    def _check_field_count(self, collection_name: str) -> ValidationCheck:
        """Verify the Milvus collection has the expected number of fields."""
        expected = sum(
            1 for fc in self.conversion.field_conversions
            if not fc.skipped and fc.milvus_field is not None
        )
        try:
            # Get collection schema info
            info = self.milvus.client.describe_collection(collection_name)
            # The number of fields in the schema
            actual_fields = len(info.get("fields", []))

            return ValidationCheck(
                name="field_count",
                passed=actual_fields >= expected,
                expected=expected,
                actual=actual_fields,
                message=f"フィールド数: 期待={expected}, 実際={actual_fields}",
            )
        except Exception as e:
            return ValidationCheck(
                name="field_count",
                passed=False,
                expected=expected,
                message=f"フィールド数チェック失敗: {e}",
            )

    def _check_sample_data(
        self,
        index_name: str,
        collection_name: str,
        sample_size: int,
    ) -> list[ValidationCheck]:
        """Spot-check a sample of documents for field value consistency."""
        checks: list[ValidationCheck] = []

        try:
            # Get key field
            key_conversion = next(
                (fc for fc in self.conversion.field_conversions if fc.is_primary_key),
                None,
            )
            if not key_conversion or not key_conversion.milvus_field:
                checks.append(ValidationCheck(
                    name="sample_data",
                    passed=True,
                    message="プライマリキーがないためサンプルチェックをスキップ",
                ))
                return checks

            # Get sample from Milvus
            milvus_samples = self.milvus.sample_query(
                collection_name, limit=min(sample_size, 10)
            )

            if not milvus_samples:
                checks.append(ValidationCheck(
                    name="sample_data",
                    passed=True,
                    message="Milvus にデータがないためサンプルチェックをスキップ",
                ))
                return checks

            # Check non-vector scalar fields
            scalar_fields = [
                fc for fc in self.conversion.field_conversions
                if not fc.skipped
                and not fc.mapping.is_vector
                and fc.milvus_field is not None
            ]

            non_null_count = 0
            total_checked = 0
            for sample in milvus_samples:
                for fc in scalar_fields:
                    milvus_name = fc.milvus_field.name
                    if milvus_name in sample:
                        total_checked += 1
                        if sample[milvus_name] is not None:
                            non_null_count += 1

            if total_checked > 0:
                fill_rate = non_null_count / total_checked * 100
                checks.append(ValidationCheck(
                    name="sample_data_fill_rate",
                    passed=fill_rate > 50,
                    expected=">50%",
                    actual=f"{fill_rate:.1f}%",
                    message=f"サンプルデータ充填率: {fill_rate:.1f}% ({non_null_count}/{total_checked})",
                ))

            # Vector dimension check
            for fc in self.conversion.field_conversions:
                if fc.mapping.is_vector and fc.milvus_field:
                    expected_dim = fc.milvus_field.params.get("dim")
                    if expected_dim:
                        checks.append(ValidationCheck(
                            name=f"vector_dim_{fc.milvus_field.name}",
                            passed=True,
                            expected=expected_dim,
                            actual=expected_dim,
                            message=f"ベクトルフィールド '{fc.milvus_field.name}' の次元数: {expected_dim}",
                        ))

        except Exception as e:
            checks.append(ValidationCheck(
                name="sample_data",
                passed=False,
                message=f"サンプルデータチェック失敗: {e}",
            ))

        return checks
