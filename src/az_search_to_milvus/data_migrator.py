"""Data migration engine: Azure AI Search → Milvus.

Handles batch extraction from Azure AI Search, data transformation to match
the Milvus schema, and insertion with checkpoint-based resumability.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from typing import Any

from pymilvus import DataType

from az_search_to_milvus.clients.ai_search import AzureSearchClientWrapper
from az_search_to_milvus.clients.milvus import MilvusClientWrapper
from az_search_to_milvus.config import MigrationConfig
from az_search_to_milvus.schema_converter import FieldConversion, SchemaConversionResult
from az_search_to_milvus.utils.checkpoint import CheckpointManager, MigrationCheckpoint

logger = logging.getLogger("az_search_to_milvus.data_migrator")


class DataTransformer:
    """Transforms Azure AI Search documents to Milvus-compatible format.

    Uses the field conversion results from :class:`SchemaConverter` to know
    how to map each field.
    """

    def __init__(self, field_conversions: list[FieldConversion]) -> None:
        self._field_map: dict[str, FieldConversion] = {}
        self._key_field: str | None = None

        for fc in field_conversions:
            if not fc.skipped and fc.milvus_field is not None:
                self._field_map[fc.azure_name] = fc
                if fc.is_primary_key:
                    self._key_field = fc.azure_name

    @property
    def key_field(self) -> str | None:
        return self._key_field

    def transform_document(self, doc: dict[str, Any]) -> dict[str, Any] | None:
        """Transform a single Azure document to a Milvus-compatible dict.

        Returns ``None`` if the document cannot be transformed (e.g. missing
        primary key).
        """
        result: dict[str, Any] = {}

        for azure_name, fc in self._field_map.items():
            value = doc.get(azure_name)
            milvus_name = fc.milvus_field.name
            dtype = fc.mapping.milvus_type

            if value is None:
                # Use type-appropriate defaults for required fields
                if fc.is_primary_key:
                    logger.warning("ドキュメントにキーフィールド '%s' がありません。スキップ", azure_name)
                    return None
                value = self._default_value(dtype, fc)
                if value is None:
                    continue
            else:
                value = self._coerce_value(value, dtype, fc)

            result[milvus_name] = value

        return result

    def transform_batch(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform a batch of documents, skipping those that fail."""
        results: list[dict[str, Any]] = []
        for doc in docs:
            transformed = self.transform_document(doc)
            if transformed is not None:
                results.append(transformed)
        return results

    def _coerce_value(self, value: Any, dtype: DataType, fc: FieldConversion) -> Any:
        """Coerce a value to the target Milvus DataType."""
        if dtype == DataType.VARCHAR:
            if isinstance(value, str):
                max_len = fc.milvus_field.params.get("max_length", 65_535)
                return value[:max_len]
            return str(value)

        if dtype == DataType.INT32:
            return int(value) if value is not None else 0

        if dtype == DataType.INT64:
            return int(value) if value is not None else 0

        if dtype == DataType.INT16:
            return int(value) if value is not None else 0

        if dtype == DataType.INT8:
            return int(value) if value is not None else 0

        if dtype == DataType.FLOAT:
            return float(value) if value is not None else 0.0

        if dtype == DataType.DOUBLE:
            return float(value) if value is not None else 0.0

        if dtype == DataType.BOOL:
            return bool(value)

        if dtype == DataType.JSON:
            if isinstance(value, (dict, list)):
                return value
            return json.loads(value) if isinstance(value, str) else {"value": value}

        if dtype == DataType.ARRAY:
            if isinstance(value, list):
                element_type = fc.mapping.element_type
                if element_type == DataType.VARCHAR:
                    return [str(v) for v in value]
                if element_type in (DataType.INT32, DataType.INT64):
                    return [int(v) for v in value]
                if element_type == DataType.DOUBLE:
                    return [float(v) for v in value]
                return value
            return [value] if value is not None else []

        if dtype == DataType.FLOAT_VECTOR:
            return self._coerce_float_vector(value, fc)

        if dtype == DataType.FLOAT16_VECTOR:
            return self._coerce_float16_vector(value, fc)

        if dtype == DataType.BINARY_VECTOR:
            return self._coerce_binary_vector(value, fc)

        return value

    def _coerce_float_vector(self, value: Any, fc: FieldConversion) -> list[float]:
        """Coerce to float32 vector, handling int8/int16 upcast."""
        if isinstance(value, list):
            return [float(v) for v in value]
        # Base64-encoded binary from Azure REST API
        if isinstance(value, (bytes, bytearray)):
            return [float(b) for b in value]
        return value

    def _coerce_float16_vector(self, value: Any, fc: FieldConversion) -> bytes:
        """Coerce to float16 vector bytes for Milvus FLOAT16_VECTOR.

        Milvus expects raw bytes for float16 vectors.
        """
        if isinstance(value, list):
            import struct
            buf = bytearray()
            for v in value:
                # Convert float to IEEE 754 half-precision
                buf.extend(struct.pack("<e", float(v)))
            return bytes(buf)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return value

    def _coerce_binary_vector(self, value: Any, fc: FieldConversion) -> bytes:
        """Coerce to binary vector for Milvus BINARY_VECTOR."""
        if isinstance(value, list):
            return bytes(value)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return value

    def _default_value(self, dtype: DataType, fc: FieldConversion) -> Any:
        """Return a sensible default for a given DataType, or None to skip."""
        defaults = {
            DataType.VARCHAR: "",
            DataType.INT32: 0,
            DataType.INT64: 0,
            DataType.INT16: 0,
            DataType.INT8: 0,
            DataType.FLOAT: 0.0,
            DataType.DOUBLE: 0.0,
            DataType.BOOL: False,
            DataType.JSON: {},
            DataType.ARRAY: [],
        }
        return defaults.get(dtype)


# ---------------------------------------------------------------------------
# Migration engine
# ---------------------------------------------------------------------------


class DataMigrator:
    """Orchestrates the full data migration from Azure AI Search to Milvus.

    Features:
    - Batch processing with configurable batch size
    - Checkpoint-based resumability
    - Progress reporting
    - Error handling with per-document skip
    """

    def __init__(
        self,
        config: MigrationConfig,
        conversion_result: SchemaConversionResult,
        azure_client: AzureSearchClientWrapper,
        milvus_client: MilvusClientWrapper,
    ) -> None:
        self.config = config
        self.conversion = conversion_result
        self.azure = azure_client
        self.milvus = milvus_client
        self.transformer = DataTransformer(conversion_result.field_conversions)
        self.checkpoint_mgr = CheckpointManager(config.options.checkpoint_dir)
        self._progress_callback: Any = None

    def set_progress_callback(self, callback: Any) -> None:
        """Set a callback ``fn(migrated, total)`` for progress reporting."""
        self._progress_callback = callback

    def migrate(self) -> MigrationCheckpoint:
        """Run the full data migration.

        Returns the final checkpoint with migration statistics.
        """
        index_name = self.config.azure_search.index_name
        collection_name = self.conversion.milvus_collection_name

        # Check for existing checkpoint (resume support)
        checkpoint = self.checkpoint_mgr.load(index_name)
        if checkpoint and checkpoint.status == "in_progress":
            logger.info(
                "前回のチェックポイントから再開: %d/%d ドキュメント完了",
                checkpoint.migrated_documents,
                checkpoint.total_documents,
            )
            skip_count = checkpoint.migrated_documents
        else:
            # Get total document count
            total = self.azure.get_document_count(index_name)
            checkpoint = MigrationCheckpoint(
                index_name=index_name,
                collection_name=collection_name,
                total_documents=total,
            )
            skip_count = 0

        checkpoint.mark_in_progress()
        self.checkpoint_mgr.save(checkpoint)

        if self.config.options.dry_run:
            logger.info("[DRY RUN] 実際のデータ挿入はスキップします")

        # Create collection and indexes
        self._setup_collection(collection_name)

        # Migrate data in batches
        try:
            for batch in self.azure.extract_documents(
                index_name,
                batch_size=self.config.options.batch_size,
                skip_count=skip_count,
            ):
                transformed = self.transformer.transform_batch(batch)

                if not transformed:
                    logger.warning("バッチの全ドキュメントがスキップされました")
                    continue

                if not self.config.options.dry_run:
                    inserted = self.milvus.insert_batch(collection_name, transformed)
                    logger.info("挿入: %d ドキュメント", inserted)
                else:
                    logger.info("[DRY RUN] 変換成功: %d ドキュメント", len(transformed))

                # Update checkpoint
                last_key = ""
                key_field = self.transformer.key_field
                if key_field and batch:
                    last_key = str(batch[-1].get(key_field, ""))

                checkpoint.advance(len(transformed), last_key)
                self.checkpoint_mgr.save(checkpoint)

                if self._progress_callback:
                    self._progress_callback(
                        checkpoint.migrated_documents, checkpoint.total_documents
                    )

        except Exception as e:
            checkpoint.mark_failed(str(e))
            self.checkpoint_mgr.save(checkpoint)
            logger.error("移行失敗: %s", e)
            raise

        # Finalize
        checkpoint.mark_completed()
        self.checkpoint_mgr.save(checkpoint)

        if not self.config.options.dry_run:
            # Flush and load for querying
            self.milvus.load_collection(collection_name)

        logger.info(
            "移行完了: %d/%d ドキュメント (%d 失敗)",
            checkpoint.migrated_documents,
            checkpoint.total_documents,
            len(checkpoint.failed_document_keys),
        )

        return checkpoint

    def _setup_collection(self, collection_name: str) -> None:
        """Create the Milvus collection and vector indexes."""
        self.milvus.create_collection(
            collection_name,
            self.conversion.collection_schema,
            drop_existing=self.config.options.drop_existing_collection,
        )

        # Create vector indexes
        for ic in self.conversion.index_conversions:
            self.milvus.create_index(
                collection_name,
                ic.target_field,
                ic.milvus_config,
            )
