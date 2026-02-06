# Azure AI Search → Milvus スキーマ対応表

本ドキュメントでは、Azure AI Search の全フィールド型と Milvus 2.6.x の DataType の
対応を網羅的に示します。

## スカラー型マッピング

| Azure AI Search (Edm) | Milvus DataType | 信頼度 | 備考 |
|---|---|---|---|
| `Edm.String` | `VARCHAR` | EXACT | `max_length` の指定が必要 (デフォルト 65,535) |
| `Edm.Int32` | `INT32` | EXACT | 完全一致 |
| `Edm.Int64` | `INT64` | EXACT | 完全一致。Milvus の主キーとして使用可能 |
| `Edm.Double` | `DOUBLE` | EXACT | 完全一致 |
| `Edm.Single` | `FLOAT` | EXACT | スカラーの float32。ベクトルではない |
| `Edm.Boolean` | `BOOL` | EXACT | 完全一致 |
| `Edm.Int16` | `INT16` | EXACT | 完全一致 |
| `Edm.SByte` | `INT8` | EXACT | 完全一致 |
| `Edm.Byte` | `INT16` | LOSSLESS | Milvus に UINT8 がないため INT16 にアップキャスト |
| `Edm.DateTimeOffset` | `VARCHAR` | SEMANTIC | ISO 8601 文字列として格納。日時比較は文字列比較になる |
| `Edm.GeographyPoint` | `JSON` | SEMANTIC | `{"type":"Point","coordinates":[lon,lat]}` 形式 |
| `Edm.ComplexType` | `JSON` | SEMANTIC | ネストされた複合型はフラット化して JSON に格納 |

### 信頼度の意味

- **EXACT**: 型が完全に一致。データの変換不要
- **LOSSLESS**: 安全なアップキャスト。データ損失なし
- **LOSSY**: 精度の低下の可能性あり
- **SEMANTIC**: 構造は異なるが意味的に等価。クエリの書き換えが必要な場合がある

## コレクション (配列) 型マッピング

| Azure AI Search (Edm) | Milvus DataType | 信頼度 | 備考 |
|---|---|---|---|
| `Collection(Edm.String)` | `ARRAY(VARCHAR)` | EXACT | filterable/facetable な文字列配列 |
| `Collection(Edm.Int32)` | `ARRAY(INT32)` | EXACT | |
| `Collection(Edm.Int64)` | `ARRAY(INT64)` | EXACT | |
| `Collection(Edm.Double)` | `ARRAY(DOUBLE)` | EXACT | |

## ベクトル型マッピング

| Azure AI Search (Edm) | Milvus DataType | 信頼度 | 備考 |
|---|---|---|---|
| `Collection(Edm.Single)` | `FLOAT_VECTOR` | EXACT | float32 ベクトル。最も一般的 |
| `Collection(Edm.Half)` | `FLOAT16_VECTOR` | EXACT | float16 ベクトル |
| `Collection(Edm.Int16)` | `FLOAT_VECTOR` | LOSSY | Milvus に INT16_VECTOR がないため float32 にアップキャスト。メモリ使用量 2倍 |
| `Collection(Edm.SByte)` | `FLOAT_VECTOR` | LOSSY | int8→float32 アップキャスト。メモリ使用量 4倍。再エンコード推奨 |
| `Collection(Edm.Byte)` | `BINARY_VECTOR` | EXACT | バイナリベクトル (パックドビット) |

### ベクトル型の注意点

**Int16/SByte ベクトルの移行について**

Azure AI Search の量子化ベクトル型 (`Collection(Edm.Int16)`, `Collection(Edm.SByte)`) は
Milvus に直接対応する型がないため、`FLOAT_VECTOR` (float32) にアップキャストされます。

- メモリ使用量が増加します (Int16: 2倍、SByte: 4倍)
- 検索精度はほぼ同等ですが、元の量子化された表現ではなくなります
- **推奨**: 元のモデルで再エンコードして float32 ベクトルを直接生成することを検討してください

## インデックスアルゴリズムマッピング

| Azure AI Search | Milvus | パラメータマッピング |
|---|---|---|
| `hnsw` | `HNSW` | `m` → `M`, `efConstruction` → `efConstruction`, `efSearch` → `ef` (検索時) |
| `exhaustiveKnn` | `FLAT` | パラメータなし (ブルートフォース) |

### 距離メトリックマッピング

| Azure AI Search | Milvus | 備考 |
|---|---|---|
| `cosine` | `COSINE` | 完全一致 |
| `euclidean` | `L2` | 完全一致 |
| `dotProduct` | `IP` | Inner Product |
| `hamming` | `HAMMING` | バイナリベクトル用 |

## フィールド属性の移行

Azure AI Search のフィールド属性の多くは Milvus では不要ですが、以下の対応関係があります:

| Azure 属性 | Milvus での対応 | 備考 |
|---|---|---|
| `key: true` | `is_primary=True` | 主キー。VARCHAR または INT64 のみ |
| `filterable: true` | (自動) | Milvus はスカラーフィールドに自動的にインデックスを作成 |
| `sortable: true` | (自動) | Milvus のスカラーフィールドはソート可能 |
| `searchable: true` | (N/A) | 全文検索は Milvus の BM25 機能を使用 |
| `facetable: true` | (N/A) | ファセットは直接対応なし。GROUP BY で代替可能 |
| `retrievable: true` | (自動) | Milvus は全フィールドが取得可能 |
| `dimensions` | `dim` パラメータ | ベクトル次元数 |
| `vectorSearchProfile` | インデックス定義 | 別途 `create_index` で指定 |

## 非対応機能 (Azure AI Search 固有)

以下の機能は Azure AI Search 固有のため、Milvus への移行対象外です。
移行ツールはこれらの機能を検出した場合、警告を出力します。

| 機能 | 説明 | 代替案 |
|---|---|---|
| **Scoring Profiles** | カスタムスコアリング | アプリケーション層でランキングロジックを実装 |
| **Semantic Ranker** | セマンティック再ランキング | Cross-Encoder 等のリランカーモデルを統合 |
| **Suggesters** | オートコンプリート | 前方一致検索 or アプリケーション層で実装 |
| **Skillsets** | AI エンリッチメントパイプライン | 別途データパイプラインを構築 |
| **Indexers** | データソース自動取り込み | Kafka/Spark コネクタ or Milvus CDC を使用 |
| **Synonym Maps** | 同義語展開 | クエリ拡張をアプリケーション層で実装 |
| **Encryption Key (CMK)** | カスタマーマネージド暗号化 | LUKS/dm-crypt or クラウドボリューム暗号化 |
| **geo.distance()** | 地理空間検索 | PostGIS 等との併用を検討 |
