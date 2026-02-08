"""E2E 移行テストスクリプト。

Azure AI Search → Milvus の移行を実際に実行して検証する。
全ステップ（アセスメント → スキーマ変換 → データ移行 → バリデーション）を順に実行する。

前提条件:
1. Azure AI Search にテストインデックスが作成済み（setup_test_index.py）
2. Milvus がローカルまたはリモートで起動中
3. 環境変数が設定済み:
   - AZURE_SEARCH_ENDPOINT
   - AZURE_SEARCH_API_KEY
   - MILVUS_URI（デフォルト: http://localhost:19530）

使い方:
    # フルテスト（全ステップ実行）
    python examples/e2e_migration_test.py

    # ドライラン（Milvus への書き込みなし）
    python examples/e2e_migration_test.py --dry-run

    # アセスメントのみ
    python examples/e2e_migration_test.py --assess-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# プロジェクトルートを PATH に追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from az_search_to_milvus.assessment import generate_assessment, print_assessment
from az_search_to_milvus.clients.ai_search import AzureSearchClientWrapper
from az_search_to_milvus.clients.milvus import MilvusClientWrapper
from az_search_to_milvus.config import (
    AzureSearchConfig,
    MigrationConfig,
    MigrationOptions,
    MilvusConfig,
)
from az_search_to_milvus.data_migrator import DataMigrator
from az_search_to_milvus.schema_converter import SchemaConverter
from az_search_to_milvus.utils.logging import setup_logging
from az_search_to_milvus.validation import validate_migration

INDEX_NAME = "test-migration-products"
COLLECTION_NAME = "test_migration_products"


def build_config(
    *,
    dry_run: bool = False,
    milvus_uri: str = "http://localhost:19530",
) -> MigrationConfig:
    """テスト用の設定を構築する。"""
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
    api_key = os.environ.get("AZURE_SEARCH_API_KEY", "")

    if not endpoint or not api_key:
        print("エラー: 環境変数が設定されていません")
        print("  export AZURE_SEARCH_ENDPOINT='https://your-service.search.windows.net'")
        print("  export AZURE_SEARCH_API_KEY='your-admin-key'")
        sys.exit(1)

    return MigrationConfig(
        azure_search=AzureSearchConfig(
            endpoint=endpoint,
            api_key=api_key,
            index_name=INDEX_NAME,
        ),
        milvus=MilvusConfig(
            uri=milvus_uri,
            collection_name=COLLECTION_NAME,
        ),
        options=MigrationOptions(
            batch_size=100,
            drop_existing_collection=True,
            dry_run=dry_run,
            enable_dynamic_field=True,
            varchar_max_length=65535,
            array_max_capacity=4096,
            checkpoint_dir=".checkpoints_test",
        ),
    )


def step_assess(config: MigrationConfig) -> None:
    """ステップ1: 移行前アセスメントを実行する。"""
    print("\n" + "=" * 60)
    print("ステップ 1: 移行前アセスメント")
    print("=" * 60)

    az_client = AzureSearchClientWrapper(config.azure_search)
    index = az_client.get_index()
    doc_count = az_client.get_document_count()

    print(f"インデックス: {index.name}")
    print(f"ドキュメント数: {doc_count:,}")
    print(f"フィールド数: {len(index.fields)}")

    # スキーマ変換
    converter = SchemaConverter(config.options)
    conversion = converter.convert_from_index(index)

    # アセスメントレポート
    report = generate_assessment(conversion, doc_count)
    print_assessment(report)

    # レポートを保存
    report.save_json("test_assessment_report.json")
    print("アセスメントレポート保存: test_assessment_report.json")

    return conversion


def step_migrate(config: MigrationConfig, conversion) -> None:
    """ステップ2: データ移行を実行する。"""
    print("\n" + "=" * 60)
    print("ステップ 2: データ移行")
    if config.options.dry_run:
        print("  [ドライランモード — Milvus への書き込みなし]")
    print("=" * 60)

    az_client = AzureSearchClientWrapper(config.azure_search)
    milvus_client = MilvusClientWrapper(config.milvus)

    if not config.options.dry_run:
        milvus_client.connect()

    migrator = DataMigrator(config, conversion, az_client, milvus_client)

    # プログレスコールバック
    def on_progress(migrated: int, total: int) -> None:
        pct = (migrated / total * 100) if total > 0 else 0
        print(f"  進捗: {migrated}/{total} ({pct:.1f}%)")

    migrator.set_progress_callback(on_progress)

    start_time = time.time()
    checkpoint = migrator.migrate()
    elapsed = time.time() - start_time

    print(f"\n移行結果:")
    print(f"  ステータス: {checkpoint.status}")
    print(f"  移行ドキュメント: {checkpoint.migrated_documents}/{checkpoint.total_documents}")
    print(f"  失敗: {len(checkpoint.failed_document_keys)} 件")
    print(f"  所要時間: {elapsed:.1f} 秒")

    if not config.options.dry_run:
        milvus_client.disconnect()


def step_validate(config: MigrationConfig) -> None:
    """ステップ3: 移行後バリデーションを実行する。"""
    print("\n" + "=" * 60)
    print("ステップ 3: 移行後バリデーション")
    print("=" * 60)

    az_client = AzureSearchClientWrapper(config.azure_search)
    milvus_client = MilvusClientWrapper(config.milvus)
    milvus_client.connect()

    converter = SchemaConverter(config.options)
    index = az_client.get_index()
    conversion = converter.convert_from_index(index)

    results = validate_migration(
        azure_client=az_client,
        milvus_client=milvus_client,
        conversion_result=conversion,
    )

    print("\nバリデーション結果:")
    all_passed = True
    for check_name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}: {detail}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n全チェック合格! 移行は正常に完了しました。")
    else:
        print("\n一部のチェックが失敗しました。詳細を確認してください。")

    milvus_client.disconnect()

    # Milvus のサンプルデータを確認
    print("\n--- Milvus サンプルデータ ---")
    milvus_client_2 = MilvusClientWrapper(config.milvus)
    milvus_client_2.connect()
    try:
        samples = milvus_client_2.sample_query(
            COLLECTION_NAME,
            limit=3,
            output_fields=["id", "title", "category", "price", "rating", "tags"],
        )
        for i, doc in enumerate(samples):
            print(f"  [{i+1}] id={doc.get('id')}, title={doc.get('title')}, "
                  f"category={doc.get('category')}, price={doc.get('price')}, "
                  f"rating={doc.get('rating')}")
    except Exception as e:
        print(f"  サンプル取得エラー: {e}")
    finally:
        milvus_client_2.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Azure AI Search → Milvus E2E 移行テスト"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ドライラン（Milvus 書き込みなし）",
    )
    parser.add_argument(
        "--assess-only",
        action="store_true",
        help="アセスメントのみ実行",
    )
    parser.add_argument(
        "--milvus-uri",
        default=os.environ.get("MILVUS_URI", "http://localhost:19530"),
        help="Milvus URI（デフォルト: http://localhost:19530）",
    )
    args = parser.parse_args()

    setup_logging(verbose=True)

    print("=" * 60)
    print("Azure AI Search → Milvus E2E 移行テスト")
    print("=" * 60)

    config = build_config(
        dry_run=args.dry_run,
        milvus_uri=args.milvus_uri,
    )

    # ステップ1: アセスメント
    conversion = step_assess(config)

    if args.assess_only:
        print("\n--assess-only: アセスメントのみ完了")
        return

    # ステップ2: データ移行
    step_migrate(config, conversion)

    # ステップ3: バリデーション（ドライランでない場合のみ）
    if not args.dry_run:
        step_validate(config)
    else:
        print("\n[ドライラン] バリデーションはスキップされました")

    print("\n" + "=" * 60)
    print("E2E テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
