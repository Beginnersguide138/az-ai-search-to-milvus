# az-ai-search-to-milvus

**Azure AI Search → Milvus/Zilliz 移行ツール**

Azure AI Search (旧 Azure Cognitive Search) から、セルフホスト Milvus または Zilliz Cloud へのデータ移行を自動化する Python ツールです。AWS Schema Conversion Tool (SCT) のようにスキーマ変換・データ移行・バリデーションをワンストップで提供します。

## 特徴

- **網羅的なスキーマ変換** — Azure AI Search の全 Edm 型 → Milvus 2.6.x DataType のマッピング
- **ベクトルインデックス移行** — HNSW / Exhaustive KNN パラメータの 1:1 マッピング
- **バッチデータ移行** — チェックポイントによる再開可能なバッチ処理
- **移行前アセスメント** — 互換性分析、非対応機能の検出、Milvus メリットの提示
- **移行後バリデーション** — ドキュメント数・フィールド値の整合性検証
- **Zilliz Cloud 対応** — セルフホスト Milvus と Zilliz Cloud の両方をサポート
- **CLI + ライブラリ** — コマンドラインツールとしても Python ライブラリとしても使用可能

## クイックスタート

### インストール

```bash
git clone https://github.com/Beginnersguide138/az-ai-search-to-milvus.git
cd az-ai-search-to-milvus
pip install -e .
```

### 設定

```bash
cp examples/config.example.yaml config.yaml
# config.yaml を編集して接続情報を設定
```

```yaml
azure_search:
  endpoint: "https://your-service.search.windows.net"
  index_name: "your-index"
  api_key: ""  # 環境変数 AZURE_SEARCH_API_KEY を推奨

milvus:
  uri: "http://localhost:19530"
  collection_name: "your_collection"

options:
  batch_size: 500
  enable_dynamic_field: true
```

### 実行

```bash
# Step 1: アセスメント (スキーマ分析 + 互換性レポート)
az-search-to-milvus assess --config config.yaml

# Step 2: データ移行
az-search-to-milvus migrate --config config.yaml

# Step 3: バリデーション
az-search-to-milvus validate --config config.yaml
```

## スキーマ対応表

### スカラー型

| Azure AI Search (Edm) | Milvus DataType | 信頼度 |
|---|---|---|
| `Edm.String` | `VARCHAR` | EXACT |
| `Edm.Int32` | `INT32` | EXACT |
| `Edm.Int64` | `INT64` | EXACT |
| `Edm.Double` | `DOUBLE` | EXACT |
| `Edm.Single` | `FLOAT` | EXACT |
| `Edm.Boolean` | `BOOL` | EXACT |
| `Edm.Int16` | `INT16` | EXACT |
| `Edm.SByte` | `INT8` | EXACT |
| `Edm.Byte` | `INT16` | LOSSLESS |
| `Edm.DateTimeOffset` | `VARCHAR` | SEMANTIC |
| `Edm.GeographyPoint` | `JSON` | SEMANTIC |
| `Edm.ComplexType` | `JSON` | SEMANTIC |

### ベクトル型

| Azure AI Search (Edm) | Milvus DataType | 信頼度 |
|---|---|---|
| `Collection(Edm.Single)` | `FLOAT_VECTOR` | EXACT |
| `Collection(Edm.Half)` | `FLOAT16_VECTOR` | EXACT |
| `Collection(Edm.Byte)` | `BINARY_VECTOR` | EXACT |
| `Collection(Edm.Int16)` | `FLOAT_VECTOR` | LOSSY |
| `Collection(Edm.SByte)` | `FLOAT_VECTOR` | LOSSY |

### コレクション (配列) 型

| Azure AI Search (Edm) | Milvus DataType | 信頼度 |
|---|---|---|
| `Collection(Edm.String)` | `ARRAY(VARCHAR)` | EXACT |
| `Collection(Edm.Int32)` | `ARRAY(INT32)` | EXACT |
| `Collection(Edm.Int64)` | `ARRAY(INT64)` | EXACT |
| `Collection(Edm.Double)` | `ARRAY(DOUBLE)` | EXACT |

### インデックスアルゴリズム

| Azure AI Search | Milvus | メトリック |
|---|---|---|
| `hnsw` | `HNSW` | cosine→COSINE, euclidean→L2, dotProduct→IP |
| `exhaustiveKnn` | `FLAT` | 同上 |

> 詳細は [docs/schema_mapping.md](docs/schema_mapping.md) を参照

## Milvus への移行で得られるメリット

Azure AI Search にはない Milvus 固有の機能:

| 機能 | 説明 |
|---|---|
| **12+ インデックスタイプ** | IVF_FLAT, IVF_SQ8, IVF_PQ, SCANN, DiskANN, GPU_CAGRA 等 |
| **GPU アクセラレーション** | Azure NC/ND VM で GPU_IVF_FLAT, GPU_CAGRA による高速検索 |
| **ハイブリッド検索** | SPARSE_FLOAT_VECTOR によるネイティブ密+スパースベクトル検索 |
| **パーティションキー** | ネイティブマルチテナンシーサポート |
| **Dynamic Schema** | スキーマ外フィールドの柔軟な格納 |
| **Range Search** | 距離閾値ベースの検索 |
| **Grouping Search** | フィールドによるグルーピング検索 |
| **Iterator API** | $skip 100K 制限なしの大量データ取得 |
| **CDC** | Change Data Capture によるデータ同期 |
| **コスト管理** | VM コストのみ、クエリ課金なし |

> 詳細は [docs/milvus_advantages.md](docs/milvus_advantages.md) を参照

## 非対応機能 (Azure AI Search 固有)

以下の機能は移行対象外です。ツールはこれらを検出して警告を出力します:

- Scoring Profiles → アプリケーション層でランキングロジックを実装
- Semantic Ranker → Cross-Encoder 等のリランカーモデルを統合
- Suggesters → 前方一致検索 or アプリケーション層で実装
- Skillsets / Indexers → データパイプラインを別途構築
- Synonym Maps → クエリ拡張をアプリケーション層で実装
- geo.distance() → PostGIS 等との併用を検討

## CLI コマンド

```
az-search-to-milvus [OPTIONS] COMMAND [ARGS]...

Commands:
  assess    移行前アセスメントを実行
  migrate   データ移行を実行
  validate  移行後のデータ整合性を検証
  schema    スキーマ変換のみ実行 (データ移行なし)

Options:
  --version  Show the version and exit.
  -v         詳細ログを出力
```

### assess

```bash
az-search-to-milvus assess --config config.yaml [--output report.json]
```

### migrate

```bash
az-search-to-milvus migrate --config config.yaml [--dry-run] [--drop-existing] [--no-resume]
```

### validate

```bash
az-search-to-milvus validate --config config.yaml [--sample-size 100]
```

### schema

```bash
# SDK 経由
az-search-to-milvus schema --config config.yaml [--output schema.json]

# REST API JSON から (Azure 接続不要)
az-search-to-milvus schema --config config.yaml --from-json index.json
```

## プロジェクト構成

```
az-ai-search-to-milvus/
├── src/az_search_to_milvus/
│   ├── type_mapping.py        # Edm → Milvus 型マッピング
│   ├── index_mapping.py       # ベクトルインデックスマッピング
│   ├── schema_converter.py    # スキーマ変換エンジン
│   ├── data_migrator.py       # データ移行エンジン
│   ├── assessment.py          # 移行前アセスメント
│   ├── validation.py          # 移行後バリデーション
│   ├── config.py              # 設定モデル
│   ├── cli.py                 # CLI インターフェース
│   ├── clients/
│   │   ├── ai_search.py       # Azure AI Search クライアント
│   │   └── milvus.py          # Milvus クライアント
│   └── utils/
│       ├── logging.py         # ログ設定
│       └── checkpoint.py      # チェックポイント管理
├── examples/
│   ├── config.example.yaml    # 設定ファイルテンプレート
│   ├── 01_assess.py           # アセスメント例
│   ├── 02_migrate.py          # 移行例
│   ├── 03_validate.py         # バリデーション例
│   └── 04_schema_from_json.py # JSON からのスキーマ変換例
├── docs/
│   ├── schema_mapping.md      # 網羅的スキーマ対応表
│   ├── migration_guide.md     # ステップバイステップガイド
│   └── milvus_advantages.md   # Milvus メリット解説
├── tests/                     # ユニットテスト (83 tests)
└── pyproject.toml
```

## 対象環境

- **Python**: 3.10+
- **Azure AI Search**: SDK 11.6.0+
- **Milvus**: 2.5.x〜2.6.x
- **想定実行環境**: Azure VM (同一 VNet 内推奨)
- **Zilliz Cloud**: サポート対象

## 開発

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## ドキュメント

- [スキーマ対応表 (網羅版)](docs/schema_mapping.md)
- [移行ガイド (ステップバイステップ)](docs/migration_guide.md)
- [Milvus メリット解説](docs/milvus_advantages.md)

## ライセンス

MIT
