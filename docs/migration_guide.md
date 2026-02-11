# Azure AI Search → Milvus 移行ガイド

本ドキュメントでは、Azure VM 上で Azure AI Search から Milvus への
データ移行を行う手順をステップバイステップで説明します。

## 前提条件

- Azure VM (推奨: Standard_D8s_v5 以上)
- Python 3.10+
- Azure AI Search のエンドポイントと API キー (または Entra ID 認証)
- Milvus 2.6.x (同一 VNet 内推奨) または Zilliz Cloud アカウント

## Step 1: 環境構築

### 1.1 Milvus のデプロイ (セルフホストの場合)

```bash
# Docker Compose で Milvus をデプロイ
wget https://github.com/milvus-io/milvus/releases/download/v2.6.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 起動
docker compose up -d

# 確認
docker compose ps
```

### 1.2 移行ツールのインストール

```bash
# リポジトリのクローン
git clone https://github.com/Beginnersguide138/az-ai-search-to-milvus.git
cd az-ai-search-to-milvus

# 仮想環境の作成とインストール
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 1.3 設定ファイルの準備

```bash
cp examples/config.example.yaml config.yaml
```

`config.yaml` を編集:

```yaml
azure_search:
  endpoint: "https://your-search-service.search.windows.net"
  index_name: "your-index-name"
  api_key: ""  # 環境変数 AZURE_SEARCH_API_KEY を推奨
  use_entra_id: false

milvus:
  uri: "http://localhost:19530"
  collection_name: "your_collection_name"
  # Zilliz Cloud の場合
  # use_zilliz: true
  # zilliz_endpoint: "https://xxx.zillizcloud.com"
  # zilliz_api_key: ""

options:
  batch_size: 500
  drop_existing_collection: false
  enable_dynamic_field: true
  # partition_key_field: "tenant_id"  # マルチテナンシーの場合
```

環境変数の設定:

```bash
export AZURE_SEARCH_API_KEY="your-api-key"
# Zilliz Cloud の場合
# export ZILLIZ_API_KEY="your-zilliz-api-key"
```

## Step 2: アセスメント (移行前分析)

```bash
az-search-to-milvus assess --config config.yaml --output assessment.json
```

このコマンドは以下を出力します:

- スキーマの対応表 (全フィールドの型マッピング)
- 移行可能性の判定 (FULL / PARTIAL / COMPLEX)
- 非対応機能の警告 (Scoring Profile、Semantic Ranker 等)
- Milvus への移行で得られるメリット
- JSON レポート (CI/CD パイプラインでの利用に)

### アセスメント結果の読み方

| 判定 | 意味 |
|---|---|
| **FULL** | 全フィールドが損失なしで変換可能 |
| **PARTIAL** | 一部のフィールドが損失あり変換 (精度低下の可能性) |
| **COMPLEX** | 手動での対応が必要なフィールドがある |

## Step 3: スキーマ変換 (オプション)

データ移行前にスキーマ変換のみ確認したい場合:

```bash
# SDK 経由
az-search-to-milvus schema --config config.yaml --output schema.json

# REST API エクスポートの JSON から変換
az-search-to-milvus schema --config config.yaml --from-json index_definition.json
```

### フィールドのカスタマイズ

`config.yaml` の `field_overrides` でフィールドごとの設定を変更できます:

```yaml
options:
  field_overrides:
    content:
      milvus_name: "text_content"  # フィールド名の変更
      max_length: 32768            # VARCHAR の最大長
    category:
      milvus_name: "category"
  exclude_fields:
    - "internal_field"  # 移行不要なフィールドを除外
```

## Step 4: データ移行

```bash
# ドライラン (実際の書き込みなし)
az-search-to-milvus migrate --config config.yaml --dry-run

# 本番移行
az-search-to-milvus migrate --config config.yaml

# 既存コレクションを削除して再作成
az-search-to-milvus migrate --config config.yaml --drop-existing
```

### 移行の動作

1. Azure AI Search からドキュメントをバッチで抽出
2. 各ドキュメントを Milvus スキーマに変換
3. Milvus にバッチ挿入
4. チェックポイントを保存 (障害時の再開用)
5. ベクトルインデックスを構築
6. コレクションをメモリにロード

### チェックポイントによる再開

移行が中断した場合、同じコマンドを再実行すると
最後のチェックポイントから自動的に再開します:

```bash
# 中断後に再実行 → チェックポイントから再開
az-search-to-milvus migrate --config config.yaml

# チェックポイントを無視して最初から
az-search-to-milvus migrate --config config.yaml --no-resume
```

## Step 5: バリデーション

```bash
az-search-to-milvus validate --config config.yaml
```

バリデーションでは以下をチェックします:

- **ドキュメント数**: Azure と Milvus のドキュメント数が一致するか
- **フィールド数**: Milvus コレクションのフィールド数が期待値と一致するか
- **サンプルデータ**: ランダムサンプルのフィールド値が正しく移行されているか
- **ベクトル次元数**: ベクトルフィールドの次元数が正しいか

## Step 6: アプリケーションの切り替え

### 6.1 Milvus クライアントへの移行

Azure AI Search SDK から PyMilvus への書き換え例:

**Before (Azure AI Search)**:
```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

client = SearchClient(
    endpoint="https://xxx.search.windows.net",
    index_name="my-index",
    credential=AzureKeyCredential("key"),
)

results = client.search(
    search_text=None,
    vector_queries=[VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=10,
        fields="embedding",
    )],
    filter="category eq 'tech'",
)
```

**After (Milvus)**:
```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

results = client.search(
    collection_name="my_collection",
    data=[query_embedding],
    anns_field="embedding",
    limit=10,
    filter='category == "tech"',
    output_fields=["title", "content", "category"],
    search_params={"metric_type": "COSINE", "params": {"ef": 100}},
)
```

### 6.2 フィルタ構文の変換

| Azure AI Search (OData) | Milvus | 備考 |
|---|---|---|
| `field eq 'value'` | `field == "value"` | 等号 |
| `field ne 'value'` | `field != "value"` | 不等号 |
| `field gt 10` | `field > 10` | 大なり |
| `field ge 10` | `field >= 10` | 以上 |
| `field lt 10` | `field < 10` | 小なり |
| `field le 10` | `field <= 10` | 以下 |
| `field eq true` | `field == true` | Boolean |
| `search.in(field, 'a,b,c')` | `field in ["a", "b", "c"]` | IN 句 |
| `field/any(t: t eq 'x')` | `array_contains(field, "x")` | 配列内検索 |
| `f1 eq 'a' and f2 gt 5` | `f1 == "a" and f2 > 5` | AND |
| `f1 eq 'a' or f2 gt 5` | `f1 == "a" or f2 > 5` | OR |

## トラブルシューティング

### Q: 移行速度が遅い

- `batch_size` を増やす (最大 1000)
- Azure VM と Milvus が同一 VNet / 同一リージョンにあることを確認
- Milvus ノードのリソース (CPU/メモリ) を確認

### Q: メモリ不足エラー

- `batch_size` を減らす
- ベクトルの次元数が大きい場合、DiskANN インデックスの使用を検討

### Q: チェックポイントが壊れた

```bash
# チェックポイントを削除して最初から
rm -rf .checkpoints/
az-search-to-milvus migrate --config config.yaml --no-resume
```

### Q: Zilliz Cloud への接続エラー

- `zilliz_endpoint` が正しいか確認
- API キーの権限を確認
- ネットワーク (ファイアウォール / NSG) を確認
