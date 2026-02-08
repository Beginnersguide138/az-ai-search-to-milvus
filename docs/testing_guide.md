# E2E テストガイド

Azure AI Search 無料枠（F0）を使用して、移行ツールの動作を実際にテストする手順です。

## 前提条件

| 項目 | 詳細 |
|------|------|
| Python | 3.10 以上 |
| Azure サブスクリプション | 無料枠でも可 |
| Azure AI Search サービス | F0（無料）プラン |
| Milvus | Docker でローカル起動 or Zilliz Cloud 無料枠 |

## 1. Azure AI Search 無料枠の作成

### Azure ポータルでの手順

1. [Azure ポータル](https://portal.azure.com) にログイン
2. 「リソースの作成」→「AI + Machine Learning」→「Azure AI Search」
3. 以下の設定で作成:
   - **リソースグループ**: 新規作成または既存を選択
   - **サービス名**: 任意（例: `my-search-test`）
   - **場所**: Japan East（東日本）推奨
   - **価格レベル**: **Free（F0）**
4. 「確認および作成」→「作成」

### 接続情報の取得

作成後、Azure ポータルで:

1. AI Search サービスの「概要」からエンドポイント URL をコピー
   - 例: `https://my-search-test.search.windows.net`
2. 「設定」→「キー」から管理キーをコピー

```bash
export AZURE_SEARCH_ENDPOINT="https://my-search-test.search.windows.net"
export AZURE_SEARCH_API_KEY="your-admin-key-here"
```

### 無料枠の制約

| 項目 | 制限 |
|------|------|
| ストレージ | 50 MB |
| インデックス数 | 3 |
| ドキュメント数 | 10,000 |
| スケーリング | 不可（1 パーティション, 1 レプリカ） |

テストデータは約 200 件 × 128 次元ベクトルで、約 2 MB に収まるよう設計されています。

## 2. Milvus のセットアップ

### Docker でのローカル起動（推奨）

```bash
# Milvus Standalone を Docker Compose で起動
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

デフォルトで `http://localhost:19530` で起動します。

### Zilliz Cloud 無料枠を使用する場合

1. [Zilliz Cloud](https://cloud.zilliz.com/) でアカウント作成
2. 無料クラスタを作成
3. 接続情報を設定:

```bash
export MILVUS_URI="https://in01-xxxxxxx.api.gcp-us-west1.zillizcloud.com"
export MILVUS_TOKEN="your-zilliz-api-key"
```

## 3. テストの実行

### ステップ 1: テストデータ生成

```bash
python examples/test_data_generator.py -o test_data.json
```

出力例:
```
テストデータ生成中... (200 件)
推定データサイズ: 1.53 MB
生成完了: test_data.json (200 件, 1.53 MB)

--- フィールド構成 ---
  id: Edm.String (Key)
  title: Edm.String (Searchable)
  description: Edm.String (Searchable)
  category: Edm.String (Filterable, Facetable)
  brand: Edm.String (Filterable, Facetable)
  price: Edm.Double (Filterable, Sortable)
  rating: Edm.Int32 (Filterable, Sortable, Facetable)
  in_stock: Edm.Boolean (Filterable)
  created_at: Edm.DateTimeOffset (Filterable, Sortable)
  tags: Collection(Edm.String) (Filterable, Facetable)
  warehouse_location: Edm.GeographyPoint (Filterable)
  content_vector: Collection(Edm.Single) (128次元, HNSW/cosine)
```

### ステップ 2: Azure AI Search へのインデックス作成・データ登録

```bash
python examples/setup_test_index.py -d test_data.json
```

出力例:
```
============================================================
Azure AI Search テストインデックス セットアップ
============================================================
エンドポイント: https://my-search-test.search.windows.net
インデックス名: test-migration-products

インデックス 'test-migration-products' を新規作成します...
インデックス 'test-migration-products' 準備完了 (フィールド数: 12)

テストデータ: 200 件
アップロード開始...
  バッチ 1: 100/100 成功 (累計: 100/200)
  バッチ 2: 100/100 成功 (累計: 200/200)

アップロード完了: 200 成功, 0 失敗 / 合計 200
```

### ステップ 3: E2E 移行テスト

#### ドライラン（Milvus なしで検証）

```bash
python examples/e2e_migration_test.py --dry-run
```

Milvus が不要なので、まずこちらで Azure AI Search 側の動作を確認できます。

#### アセスメントのみ

```bash
python examples/e2e_migration_test.py --assess-only
```

スキーマの互換性レポートを確認できます。

#### フル移行テスト

```bash
python examples/e2e_migration_test.py
```

全ステップ（アセスメント → 移行 → バリデーション）を実行します。

### ステップ 4: クリーンアップ

テスト完了後、Azure のリソースを削除:

```bash
# テストインデックスを削除
python examples/setup_test_index.py --delete

# チェックポイントファイルを削除
rm -rf .checkpoints_test/
rm -f test_data.json test_assessment_report.json
```

## 4. テストデータのフィールド型マッピング

テストデータで検証される型変換の一覧:

| Azure AI Search フィールド | Edm 型 | Milvus DataType | 変換精度 |
|---------------------------|--------|-----------------|---------|
| `id` | Edm.String (Key) | VARCHAR (Primary Key) | EXACT |
| `title` | Edm.String | VARCHAR | EXACT |
| `description` | Edm.String | VARCHAR | EXACT |
| `category` | Edm.String | VARCHAR | EXACT |
| `brand` | Edm.String | VARCHAR | EXACT |
| `price` | Edm.Double | DOUBLE | EXACT |
| `rating` | Edm.Int32 | INT32 | EXACT |
| `in_stock` | Edm.Boolean | BOOL | EXACT |
| `created_at` | Edm.DateTimeOffset | VARCHAR (ISO 8601) | SEMANTIC |
| `tags` | Collection(Edm.String) | ARRAY[VARCHAR] | LOSSLESS |
| `warehouse_location` | Edm.GeographyPoint | JSON | SEMANTIC |
| `content_vector` | Collection(Edm.Single) | FLOAT_VECTOR (128d) | EXACT |

## 5. トラブルシューティング

### Azure AI Search 接続エラー

```
azure.core.exceptions.HttpResponseError: (403) Forbidden
```

→ API キーが正しいか確認。管理キー（クエリキーではなく）を使用してください。

### Milvus 接続エラー

```
pymilvus.exceptions.MilvusException: connection refused
```

→ Milvus が起動しているか確認:
```bash
docker ps | grep milvus
```

### ストレージ上限エラー

```
The index is over the storage quota
```

→ 無料枠の 50MB 制限を超えています。ドキュメント数を減らしてください:
```bash
python examples/test_data_generator.py -n 100 -o test_data.json
```

### ベクトル次元のミスマッチ

テストデータのベクトル次元（128）がインデックス定義と一致していることを確認してください。
テストデータ生成時の `VECTOR_DIM` とインデックス定義の `vector_search_dimensions` が
同じ値である必要があります。
