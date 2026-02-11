"""Azure AI Search 無料枠にテストインデックスを作成・登録するスクリプト。

前提条件:
1. Azure ポータルで AI Search サービス（無料枠 F0）を作成済みであること
2. 環境変数に接続情報を設定:
   - AZURE_SEARCH_ENDPOINT: 例 "https://your-service.search.windows.net"
   - AZURE_SEARCH_API_KEY: 管理キー（Azure ポータルの「キー」から取得）
3. test_data.json が存在すること（test_data_generator.py で生成）

使い方:
    # 1. まずテストデータを生成
    python examples/test_data_generator.py -o test_data.json

    # 2. Azure AI Search にインデックスを作成してデータをアップロード
    python examples/setup_test_index.py

    # 3. （オプション）インデックスを削除
    python examples/setup_test_index.py --delete
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

# --- 定数 ---
INDEX_NAME = "test-migration-products"
VECTOR_DIM = 128
UPLOAD_BATCH_SIZE = 100


def get_credentials() -> tuple[str, str]:
    """環境変数から Azure AI Search の接続情報を取得する。"""
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
    api_key = os.environ.get("AZURE_SEARCH_API_KEY", "")

    if not endpoint:
        print("エラー: 環境変数 AZURE_SEARCH_ENDPOINT が設定されていません")
        print("例: export AZURE_SEARCH_ENDPOINT='https://your-service.search.windows.net'")
        sys.exit(1)

    if not api_key:
        print("エラー: 環境変数 AZURE_SEARCH_API_KEY が設定されていません")
        print("Azure ポータル → AI Search サービス → キー から管理キーを取得してください")
        sys.exit(1)

    return endpoint, api_key


def create_index_schema() -> SearchIndex:
    """テスト用のインデックススキーマを定義する。

    以下の Azure AI Search のフィールド型を網羅:
    - Edm.String（キー、検索可能、フィルタ可能）
    - Edm.Int32（フィルタ・ソート可能）
    - Edm.Double（フィルタ・ソート可能）
    - Edm.Boolean（フィルタ可能）
    - Edm.DateTimeOffset（フィルタ・ソート可能）
    - Edm.GeographyPoint（フィルタ可能）
    - Collection(Edm.String)（フィルタ可能）
    - Collection(Edm.Single)（ベクトル検索、128次元、HNSW/cosine）
    """
    fields = [
        # キーフィールド
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
            sortable=True,
        ),
        # テキストフィールド（検索可能）
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SearchableField(
            name="description",
            type=SearchFieldDataType.String,
            filterable=False,
        ),
        # カテゴリ・ブランド（ファセット対応）
        SimpleField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            sortable=True,
        ),
        SimpleField(
            name="brand",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            sortable=True,
        ),
        # 数値フィールド
        SimpleField(
            name="price",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SimpleField(
            name="rating",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        # 真偽値
        SimpleField(
            name="in_stock",
            type=SearchFieldDataType.Boolean,
            filterable=True,
            facetable=True,
        ),
        # 日時
        SimpleField(
            name="created_at",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
        # 文字列配列
        SimpleField(
            name="tags",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True,
        ),
        # 地理座標
        SimpleField(
            name="warehouse_location",
            type=SearchFieldDataType.GeographyPoint,
            filterable=True,
        ),
        # ベクトルフィールド（128次元、HNSW、コサイン類似度）
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_DIM,
            vector_search_profile_name="hnsw-cosine-profile",
        ),
    ]

    # ベクトル検索設定
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-cosine-algo",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-cosine-profile",
                algorithm_configuration_name="hnsw-cosine-algo",
            ),
        ],
    )

    return SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )


def create_or_update_index(index_client: SearchIndexClient) -> None:
    """インデックスを作成（既存の場合は更新）する。"""
    index_schema = create_index_schema()

    try:
        existing = index_client.get_index(INDEX_NAME)
        print(f"既存のインデックス '{INDEX_NAME}' を更新します...")
        result = index_client.create_or_update_index(index_schema)
    except Exception:
        print(f"インデックス '{INDEX_NAME}' を新規作成します...")
        result = index_client.create_or_update_index(index_schema)

    print(f"インデックス '{result.name}' 準備完了 (フィールド数: {len(result.fields)})")


def upload_documents(
    search_client: SearchClient,
    docs: list[dict],
) -> None:
    """ドキュメントをバッチでアップロードする。"""
    total = len(docs)
    uploaded = 0
    failed = 0

    for i in range(0, total, UPLOAD_BATCH_SIZE):
        batch = docs[i : i + UPLOAD_BATCH_SIZE]
        try:
            result = search_client.upload_documents(documents=batch)
            batch_success = sum(1 for r in result if r.succeeded)
            batch_failed = len(batch) - batch_success
            uploaded += batch_success
            failed += batch_failed

            print(f"  バッチ {i // UPLOAD_BATCH_SIZE + 1}: "
                  f"{batch_success}/{len(batch)} 成功 "
                  f"(累計: {uploaded}/{total})")

        except Exception as e:
            print(f"  バッチ {i // UPLOAD_BATCH_SIZE + 1}: エラー - {e}")
            failed += len(batch)

        # API レート制限対策
        time.sleep(0.5)

    print(f"\nアップロード完了: {uploaded} 成功, {failed} 失敗 / 合計 {total}")


def delete_index(index_client: SearchIndexClient) -> None:
    """テストインデックスを削除する。"""
    try:
        index_client.delete_index(INDEX_NAME)
        print(f"インデックス '{INDEX_NAME}' を削除しました")
    except Exception as e:
        print(f"削除エラー: {e}")


def verify_index(search_client: SearchClient) -> None:
    """アップロード後のインデックスを検証する。"""
    # インデックスの反映を待つ
    print("\nインデックス反映を待機中 (5秒)...")
    time.sleep(5)

    # ドキュメント数を確認
    results = search_client.search(
        search_text="*",
        include_total_count=True,
        top=0,
    )
    count = results.get_count()
    print(f"インデックス内ドキュメント数: {count}")

    # テキスト検索テスト
    results = search_client.search(search_text="プレミアム", top=3)
    print("\nテキスト検索テスト（クエリ: 'プレミアム'）:")
    for doc in results:
        print(f"  - {doc['id']}: {doc['title']} (rating={doc['rating']})")

    # フィルタテスト
    results = search_client.search(
        search_text="*",
        filter="rating ge 4 and in_stock eq true",
        top=3,
    )
    print("\nフィルタテスト（rating >= 4 かつ in_stock = true）:")
    for doc in results:
        print(f"  - {doc['id']}: {doc['title']} (price={doc['price']})")

    print("\n検証完了。移行テストの準備が整いました。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Azure AI Search テストインデックスのセットアップ"
    )
    parser.add_argument(
        "--data", "-d",
        default="test_data.json",
        help="テストデータファイル（デフォルト: test_data.json）",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="テストインデックスを削除する",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="既存インデックスの検証のみ行う",
    )
    args = parser.parse_args()

    endpoint, api_key = get_credentials()
    credential = AzureKeyCredential(api_key)
    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

    if args.delete:
        delete_index(index_client)
        return

    search_client = SearchClient(
        endpoint=endpoint,
        index_name=INDEX_NAME,
        credential=credential,
    )

    if args.verify_only:
        verify_index(search_client)
        return

    # インデックスの作成
    print("=" * 60)
    print("Azure AI Search テストインデックス セットアップ")
    print("=" * 60)
    print(f"エンドポイント: {endpoint}")
    print(f"インデックス名: {INDEX_NAME}")
    print()

    create_or_update_index(index_client)

    # テストデータの読み込みとアップロード
    if not os.path.exists(args.data):
        print(f"\nエラー: テストデータファイル '{args.data}' が見つかりません")
        print("先に test_data_generator.py を実行してください:")
        print(f"  python examples/test_data_generator.py -o {args.data}")
        sys.exit(1)

    with open(args.data, encoding="utf-8") as f:
        docs = json.load(f)

    print(f"\nテストデータ: {len(docs)} 件")
    print("アップロード開始...")
    upload_documents(search_client, docs)

    # 検証
    verify_index(search_client)

    print(f"\n--- 次のステップ ---")
    print(f"1. 設定ファイルを編集:")
    print(f"   cp examples/config.test.yaml config.yaml")
    print(f"   # AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY を確認")
    print(f"2. 移行テストを実行:")
    print(f"   python examples/e2e_migration_test.py")


if __name__ == "__main__":
    main()
