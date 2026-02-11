# Azure AI Search → Milvus 移行で得られるメリット

Milvus は Azure AI Search にはない多くの機能を提供しています。
本ドキュメントでは、移行により得られる具体的なメリットを解説します。

## 1. 豊富なベクトルインデックスタイプ

Azure AI Search は **HNSW** と **Exhaustive KNN** の2種類のみですが、
Milvus は **12種類以上**のインデックスをサポートしています。

### CPU インデックス

| インデックス | 説明 | ユースケース |
|---|---|---|
| **HNSW** | グラフベースANN | 高精度・低レイテンシ。Azure AI Search と同じ |
| **FLAT** | ブルートフォース | 100% 再現率が必要な場合 |
| **IVF_FLAT** | 転置ファイル + フラット | メモリ効率と速度のバランス |
| **IVF_SQ8** | IVF + スカラー量子化 | メモリ 70-75% 削減 |
| **IVF_PQ** | IVF + プロダクト量子化 | 大規模データセットでのメモリ大幅削減 |
| **SCANN** | Score-aware quantization | IVF_PQ より高精度 |
| **DiskANN** | ディスクベースANN | メモリに収まらない超大規模データ |

### GPU インデックス (Azure NC/ND シリーズ VM)

| インデックス | 説明 | ユースケース |
|---|---|---|
| **GPU_IVF_FLAT** | GPU 上の IVF_FLAT | ハイスループット検索 |
| **GPU_IVF_PQ** | GPU 上の IVF_PQ | GPU + 大規模データセット |
| **GPU_CAGRA** | NVIDIA RAFT ベース | GPU での最高性能 |
| **GPU_BRUTE_FORCE** | GPU ブルートフォース | 小〜中規模で 100% 再現率 |

### スパースベクトルインデックス

| インデックス | 説明 | ユースケース |
|---|---|---|
| **SPARSE_INVERTED_INDEX** | スパース転置インデックス | BM25/SPLADE ハイブリッド検索 |
| **SPARSE_WAND** | WAND アルゴリズム | 大規模スパースベクトル |

## 2. ネイティブ ハイブリッド検索

Milvus は `SPARSE_FLOAT_VECTOR` 型をネイティブサポートしており、
密ベクトル (embedding) とスパースベクトル (BM25/SPLADE) を
**同じコレクション内で**組み合わせたハイブリッド検索が可能です。

```python
from pymilvus import AnnSearchRequest, RRFRanker

# 密ベクトル検索
dense_req = AnnSearchRequest(
    data=[dense_embedding],
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=10,
)

# スパースベクトル検索
sparse_req = AnnSearchRequest(
    data=[sparse_embedding],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=10,
)

# RRF (Reciprocal Rank Fusion) で結合
results = collection.hybrid_search(
    [dense_req, sparse_req],
    ranker=RRFRanker(k=60),
    limit=10,
)
```

Azure AI Search のセマンティックランカーの代替として、このハイブリッド検索 + カスタムリランカーの組み合わせが有効です。

## 3. パーティションキー (マルチテナンシー)

```python
from pymilvus import FieldSchema, DataType

# テナント ID をパーティションキーに設定
tenant_field = FieldSchema(
    name="tenant_id",
    dtype=DataType.VARCHAR,
    max_length=64,
    is_partition_key=True,  # ← Milvus 固有の機能
)
```

- テナントごとにデータが物理的に分離される
- クエリ時に自動的にパーティションプルーニングが適用される
- Azure AI Search ではフィルタで疑似的に実現していた機能が、ネイティブにサポート

## 4. Dynamic Schema

```python
schema = CollectionSchema(
    fields=[...],
    enable_dynamic_field=True,  # スキーマにないフィールドも格納可能
)

# スキーマに定義されていないフィールドも挿入可能
collection.insert([{
    "id": 1,
    "vector": [0.1, 0.2, ...],
    "title": "Document",
    "custom_metadata": {"key": "value"},  # 動的フィールド
}])
```

Azure AI Search ではフィールド追加にインデックスの更新が必要でしたが、
Milvus ではスキーマに定義されていないフィールドも柔軟に格納できます。

## 5. コスト管理

| 項目 | Azure AI Search | Milvus (セルフホスト) |
|---|---|---|
| 課金モデル | Search Unit 単位 + ストレージ | VM コストのみ |
| クエリ課金 | SU 使用量に含まれる | なし |
| スケーリング | SU の追加購入 | VM のスケールアップ/アウト |
| 最低コスト | Standard S1: ~$250/月 | Azure VM D4s_v5: ~$140/月 |
| 大規模時 | コストが急増 | リニアなスケーリング |

### Zilliz Cloud (マネージド Milvus) の場合

- CU (Capacity Unit) ベースの課金
- サーバーレスオプションで使用量に応じた課金
- セルフホストの運用コストを削減

## 6. Range Search

```python
# 距離の閾値を指定した検索 (Azure AI Search にはない機能)
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={
        "metric_type": "L2",
        "params": {"ef": 100, "radius": 0.5, "range_filter": 0.1},
    },
    limit=100,
)
```

Top-K だけでなく、「距離 0.1〜0.5 の範囲にあるすべてのドキュメント」のような
閾値ベースの検索が可能です。

## 7. Grouping Search

```python
# フィールドでグルーピングした検索 (Azure AI Search にはない機能)
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=10,
    group_by_field="category",  # カテゴリごとに Top-K
)
```

## 8. Iterator API

```python
# メモリ効率の良い大量結果の取得
# Azure AI Search の $skip 100,000 件制限がない
iterator = collection.search_iterator(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE"},
    limit=1_000_000,
    batch_size=1000,
)
while True:
    batch = iterator.next()
    if not batch:
        break
    process(batch)
```

## 9. CDC (Change Data Capture)

Milvus CDC により、データ変更をリアルタイムにキャプチャして別のシステムに同期できます。
Azure AI Search にはこの機能がありません。

- DR (Disaster Recovery) 構成
- データパイプラインとの統合
- 監査ログの生成

## 10. Null/Default 値サポート (Milvus 2.6.x)

```python
# Milvus 2.6.x の新機能
field = FieldSchema(
    name="description",
    dtype=DataType.VARCHAR,
    max_length=1024,
    nullable=True,           # NULL 値を許可
    default_value="",        # デフォルト値
)
```

Azure AI Search でも null は格納できますが、Milvus 2.6.x ではスキーマレベルで
NULL 制約とデフォルト値を明示的に定義でき、データの整合性管理が向上しています。

## まとめ

Milvus への移行により、以下のメリットが得られます:

1. **インデックスの選択肢** — ワークロードに最適なインデックスを選択可能
2. **GPU 活用** — Azure GPU VM で桁違いの検索性能
3. **ハイブリッド検索** — 密+スパースベクトルのネイティブサポート
4. **コスト最適化** — VM コストのみで予測可能な価格体系
5. **スケーラビリティ** — $skip 制限なし、Iterator API による大量データ処理
6. **柔軟性** — Dynamic Schema、Partition Key、Range Search
7. **運用** — CDC、Null/Default サポート、豊富なスカラーインデックス
