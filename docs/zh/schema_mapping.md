# Azure AI Search → Milvus Schema 映射表

本文档全面展示 Azure AI Search 所有字段类型与 Milvus 2.6.x DataType 的对应关系。

## 标量类型映射

| Azure AI Search (Edm) | Milvus DataType | 置信度 | 备注 |
|---|---|---|---|
| `Edm.String` | `VARCHAR` | EXACT | 需要指定 `max_length`（默认 65,535） |
| `Edm.Int32` | `INT32` | EXACT | 完全匹配 |
| `Edm.Int64` | `INT64` | EXACT | 完全匹配。可用作 Milvus 主键 |
| `Edm.Double` | `DOUBLE` | EXACT | 完全匹配 |
| `Edm.Single` | `FLOAT` | EXACT | 标量 float32，非向量 |
| `Edm.Boolean` | `BOOL` | EXACT | 完全匹配 |
| `Edm.Int16` | `INT16` | EXACT | 完全匹配 |
| `Edm.SByte` | `INT8` | EXACT | 完全匹配 |
| `Edm.Byte` | `INT16` | LOSSLESS | Milvus 无 UINT8，向上转换为 INT16 |
| `Edm.DateTimeOffset` | `VARCHAR` | SEMANTIC | 以 ISO 8601 字符串存储。日期比较变为字符串比较 |
| `Edm.GeographyPoint` | `JSON` | SEMANTIC | `{"type":"Point","coordinates":[lon,lat]}` 格式 |
| `Edm.ComplexType` | `JSON` | SEMANTIC | 嵌套复杂类型扁平化为 JSON 存储 |

### 置信度说明

- **EXACT**: 类型完全匹配，无需数据转换
- **LOSSLESS**: 安全的向上转换，无数据丢失
- **LOSSY**: 可能存在精度降低
- **SEMANTIC**: 结构不同但语义等价。可能需要重写查询

## 集合（数组）类型映射

| Azure AI Search (Edm) | Milvus DataType | 置信度 | 备注 |
|---|---|---|---|
| `Collection(Edm.String)` | `ARRAY(VARCHAR)` | EXACT | 可过滤/可分面的字符串数组 |
| `Collection(Edm.Int32)` | `ARRAY(INT32)` | EXACT | |
| `Collection(Edm.Int64)` | `ARRAY(INT64)` | EXACT | |
| `Collection(Edm.Double)` | `ARRAY(DOUBLE)` | EXACT | |

## 向量类型映射

| Azure AI Search (Edm) | Milvus DataType | 置信度 | 备注 |
|---|---|---|---|
| `Collection(Edm.Single)` | `FLOAT_VECTOR` | EXACT | float32 向量，最常见 |
| `Collection(Edm.Half)` | `FLOAT16_VECTOR` | EXACT | float16 向量 |
| `Collection(Edm.Int16)` | `FLOAT_VECTOR` | LOSSY | Milvus 无 INT16_VECTOR，向上转换为 float32。内存使用量翻倍 |
| `Collection(Edm.SByte)` | `FLOAT_VECTOR` | LOSSY | int8→float32 向上转换。内存使用量增加 4 倍。建议重新编码 |
| `Collection(Edm.Byte)` | `BINARY_VECTOR` | EXACT | 二进制向量（压缩位） |

### 向量类型注意事项

**Int16/SByte 向量的迁移**

Azure AI Search 的量化向量类型（`Collection(Edm.Int16)`、`Collection(Edm.SByte)`）
在 Milvus 中没有直接对应的类型，因此被向上转换为 `FLOAT_VECTOR`（float32）。

- 内存使用量会增加（Int16: 2倍，SByte: 4倍）
- 搜索精度几乎相同，但原始量化表示会丢失
- **建议**: 考虑使用原始模型重新编码，直接生成 float32 向量

## 索引算法映射

| Azure AI Search | Milvus | 参数映射 |
|---|---|---|
| `hnsw` | `HNSW` | `m` → `M`, `efConstruction` → `efConstruction`, `efSearch` → `ef`（搜索时） |
| `exhaustiveKnn` | `FLAT` | 无参数（暴力搜索） |

### 距离度量映射

| Azure AI Search | Milvus | 备注 |
|---|---|---|
| `cosine` | `COSINE` | 完全匹配 |
| `euclidean` | `L2` | 完全匹配 |
| `dotProduct` | `IP` | 内积 |
| `hamming` | `HAMMING` | 用于二进制向量 |

## 字段属性迁移

Azure AI Search 的许多字段属性在 Milvus 中不需要，但存在以下对应关系：

| Azure 属性 | Milvus 对应 | 备注 |
|---|---|---|
| `key: true` | `is_primary=True` | 主键。仅支持 VARCHAR 或 INT64 |
| `filterable: true` | （自动） | Milvus 自动为标量字段创建索引 |
| `sortable: true` | （自动） | Milvus 的标量字段可排序 |
| `searchable: true` | (N/A) | 全文搜索使用 Milvus 的 BM25 功能 |
| `facetable: true` | (N/A) | 无直接对应。可用 GROUP BY 替代 |
| `retrievable: true` | （自动） | Milvus 中所有字段都可检索 |
| `dimensions` | `dim` 参数 | 向量维度数 |
| `vectorSearchProfile` | 索引定义 | 通过 `create_index` 单独指定 |

## 不支持的功能（Azure AI Search 特有）

以下功能是 Azure AI Search 特有的，无法迁移到 Milvus。
迁移工具在检测到这些功能时会输出警告。

| 功能 | 说明 | 替代方案 |
|---|---|---|
| **Scoring Profiles** | 自定义评分 | 在应用层实现排序逻辑 |
| **Semantic Ranker** | 语义重排序 | 集成 Cross-Encoder 等重排序模型 |
| **Suggesters** | 自动补全 | 使用前缀搜索或在应用层实现 |
| **Skillsets** | AI 富化管道 | 另行构建数据管道 |
| **Indexers** | 数据源自动摄取 | 使用 Kafka/Spark 连接器或 Milvus CDC |
| **Synonym Maps** | 同义词扩展 | 在应用层实现查询扩展 |
| **Encryption Key (CMK)** | 客户管理加密 | 使用 LUKS/dm-crypt 或云卷加密 |
| **geo.distance()** | 地理空间搜索 | 考虑结合 PostGIS 等方案 |
