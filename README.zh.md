🌐 [日本語](README.md) | [English](README.en.md) | [中文](README.zh.md)

# az-ai-search-to-milvus

**Azure AI Search → Milvus/Zilliz 迁移工具**

这是一款 Python 工具，用于将 Azure AI Search（原 Azure Cognitive Search）的数据自动迁移到自托管 Milvus 或 Zilliz Cloud。类似于 AWS Schema Conversion Tool (SCT)，本工具提供一站式的 Schema 转换、数据迁移和验证功能。

## 功能特性

- **全面的 Schema 转换** — Azure AI Search 全部 Edm 类型 → Milvus 2.6.x DataType 的映射
- **向量索引迁移** — HNSW / Exhaustive KNN 参数的 1:1 映射
- **批量数据迁移** — 基于检查点的可恢复批处理
- **迁移前评估** — 兼容性分析、不支持功能的检测、Milvus 优势展示
- **迁移后验证** — 文档数量和字段值的一致性校验
- **Zilliz Cloud 支持** — 同时支持自托管 Milvus 和 Zilliz Cloud
- **CLI + 库** — 既可作为命令行工具使用，也可作为 Python 库使用

## 快速入门

### 安装

```bash
git clone https://github.com/Beginnersguide138/az-ai-search-to-milvus.git
cd az-ai-search-to-milvus
pip install -e .
```

### 配置

```bash
cp examples/config.example.yaml config.yaml
# 编辑 config.yaml 设置连接信息
```

```yaml
azure_search:
  endpoint: "https://your-service.search.windows.net"
  index_name: "your-index"
  api_key: ""  # 推荐使用环境变量 AZURE_SEARCH_API_KEY

milvus:
  uri: "http://localhost:19530"
  collection_name: "your_collection"

options:
  batch_size: 500
  enable_dynamic_field: true
```

### 运行

```bash
# Step 1: 评估（Schema 分析 + 兼容性报告）
az-search-to-milvus assess --config config.yaml

# Step 2: 数据迁移
az-search-to-milvus migrate --config config.yaml

# Step 3: 验证
az-search-to-milvus validate --config config.yaml
```

## Schema 映射表

### 标量类型

| Azure AI Search (Edm) | Milvus DataType | 可信度 |
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

### 向量类型

| Azure AI Search (Edm) | Milvus DataType | 可信度 |
|---|---|---|
| `Collection(Edm.Single)` | `FLOAT_VECTOR` | EXACT |
| `Collection(Edm.Half)` | `FLOAT16_VECTOR` | EXACT |
| `Collection(Edm.Byte)` | `BINARY_VECTOR` | EXACT |
| `Collection(Edm.Int16)` | `FLOAT_VECTOR` | LOSSY |
| `Collection(Edm.SByte)` | `FLOAT_VECTOR` | LOSSY |

### 集合（数组）类型

| Azure AI Search (Edm) | Milvus DataType | 可信度 |
|---|---|---|
| `Collection(Edm.String)` | `ARRAY(VARCHAR)` | EXACT |
| `Collection(Edm.Int32)` | `ARRAY(INT32)` | EXACT |
| `Collection(Edm.Int64)` | `ARRAY(INT64)` | EXACT |
| `Collection(Edm.Double)` | `ARRAY(DOUBLE)` | EXACT |

### 索引算法

| Azure AI Search | Milvus | 度量方式 |
|---|---|---|
| `hnsw` | `HNSW` | cosine→COSINE, euclidean→L2, dotProduct→IP |
| `exhaustiveKnn` | `FLAT` | 同上 |

> 详细信息请参阅 [docs/zh/schema_mapping.md](docs/zh/schema_mapping.md)

## 迁移到 Milvus 的优势

Azure AI Search 不具备的 Milvus 特有功能：

| 功能 | 说明 |
|---|---|
| **12+ 索引类型** | IVF_FLAT, IVF_SQ8, IVF_PQ, SCANN, DiskANN, GPU_CAGRA 等 |
| **GPU 加速** | 在 Azure NC/ND VM 上使用 GPU_IVF_FLAT、GPU_CAGRA 进行高速搜索 |
| **混合搜索** | 通过 SPARSE_FLOAT_VECTOR 实现原生稠密+稀疏向量搜索 |
| **Partition Key** | 原生多租户支持 |
| **Dynamic Schema** | 灵活存储 Schema 外字段 |
| **Range Search** | 基于距离阈值的搜索 |
| **Grouping Search** | 按字段分组搜索 |
| **Iterator API** | 无 $skip 100K 限制的大批量数据获取 |
| **CDC** | 通过 Change Data Capture 进行数据同步 |
| **成本管控** | 仅需 VM 费用，无查询计费 |

> 详细信息请参阅 [docs/zh/milvus_advantages.md](docs/zh/milvus_advantages.md)

## 不支持的功能（Azure AI Search 特有）

以下功能不在迁移范围内。本工具会检测到这些功能并输出警告：

- Scoring Profiles → 在应用层实现排序逻辑
- Semantic Ranker → 集成 Cross-Encoder 等重排序模型
- Suggesters → 使用前缀匹配搜索或在应用层实现
- Skillsets / Indexers → 另行构建数据管道
- Synonym Maps → 在应用层实现查询扩展
- geo.distance() → 考虑与 PostGIS 等配合使用

## CLI 命令

```
az-search-to-milvus [OPTIONS] COMMAND [ARGS]...

Commands:
  assess    执行迁移前评估
  migrate   执行数据迁移
  validate  验证迁移后的数据一致性
  schema    仅执行 Schema 转换（不进行数据迁移）

Options:
  --version  Show the version and exit.
  -v         输出详细日志
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
# 通过 SDK
az-search-to-milvus schema --config config.yaml [--output schema.json]

# 从 REST API JSON 文件（无需连接 Azure）
az-search-to-milvus schema --config config.yaml --from-json index.json
```

## 项目结构

```
az-ai-search-to-milvus/
├── src/az_search_to_milvus/
│   ├── type_mapping.py        # Edm → Milvus 类型映射
│   ├── index_mapping.py       # 向量索引映射
│   ├── schema_converter.py    # Schema 转换引擎
│   ├── data_migrator.py       # 数据迁移引擎
│   ├── assessment.py          # 迁移前评估
│   ├── validation.py          # 迁移后验证
│   ├── config.py              # 配置模型
│   ├── cli.py                 # CLI 接口
│   ├── clients/
│   │   ├── ai_search.py       # Azure AI Search 客户端
│   │   └── milvus.py          # Milvus 客户端
│   └── utils/
│       ├── logging.py         # 日志配置
│       └── checkpoint.py      # 检查点管理
├── examples/
│   ├── config.example.yaml    # 配置文件模板
│   ├── 01_assess.py           # 评估示例
│   ├── 02_migrate.py          # 迁移示例
│   ├── 03_validate.py         # 验证示例
│   └── 04_schema_from_json.py # 从 JSON 进行 Schema 转换示例
├── docs/
│   ├── schema_mapping.md      # 完整 Schema 映射表
│   ├── migration_guide.md     # 分步迁移指南
│   └── milvus_advantages.md   # Milvus 优势详解
├── tests/                     # 单元测试（83 个测试）
└── pyproject.toml
```

## 运行环境

- **Python**: 3.10+
- **Azure AI Search**: SDK 11.6.0+
- **Milvus**: 2.5.x ~ 2.6.x
- **推荐运行环境**: Azure VM（建议在同一 VNet 内）
- **Zilliz Cloud**: 支持

## 开发

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## 文档

- [Schema 映射表（完整版）](docs/zh/schema_mapping.md)
- [迁移指南（分步说明）](docs/zh/migration_guide.md)
- [Milvus 优势详解](docs/zh/milvus_advantages.md)
- [测试指南](docs/zh/testing_guide.md)

## 许可证

MIT
