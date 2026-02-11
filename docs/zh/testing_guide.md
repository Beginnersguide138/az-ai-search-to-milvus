# E2E 测试指南

使用 Azure AI Search 免费层（F0）测试迁移工具的分步说明。

## 前提条件

| 项目 | 详情 |
|------|------|
| Python | 3.10 或更高 |
| Azure 订阅 | 免费层即可 |
| Azure AI Search 服务 | F0（免费）计划 |
| Milvus | Docker 本地运行或 Zilliz Cloud 免费层 |

## 1. 创建 Azure AI Search 免费层

### Azure 门户操作步骤

1. 登录 [Azure 门户](https://portal.azure.com)
2. "创建资源" → "AI + 机器学习" → "Azure AI Search"
3. 使用以下设置创建：
   - **资源组**: 新建或选择现有
   - **服务名称**: 任意名称（例如：`my-search-test`）
   - **位置**: 选择最近的区域
   - **定价层**: **Free（F0）**
4. "查看 + 创建" → "创建"

### 获取连接信息

创建后，在 Azure 门户中：

1. 从 AI Search 服务的"概述"复制端点 URL
   - 示例：`https://my-search-test.search.windows.net`
2. 从"设置" → "密钥"复制管理密钥

```bash
export AZURE_SEARCH_ENDPOINT="https://my-search-test.search.windows.net"
export AZURE_SEARCH_API_KEY="your-admin-key-here"
```

### 免费层限制

| 项目 | 限制 |
|------|------|
| 存储 | 50 MB |
| 索引数 | 3 |
| 文档数 | 10,000 |
| 扩展 | 不可用（1 个分区，1 个副本） |

测试数据设计为约 200 条文档 + 128 维向量，大约 2 MB，在免费层限制范围内。

## 2. Milvus 设置

### Docker 本地运行（推荐）

```bash
# 使用 Docker Compose 启动 Milvus Standalone
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

默认在 `http://localhost:19530` 启动。

### 使用 Zilliz Cloud 免费层

1. 在 [Zilliz Cloud](https://cloud.zilliz.com/) 创建账户
2. 创建免费集群
3. 设置连接信息：

```bash
export MILVUS_URI="https://in01-xxxxxxx.api.gcp-us-west1.zillizcloud.com"
export MILVUS_TOKEN="your-zilliz-api-key"
```

## 3. 运行测试

### 步骤 1: 生成测试数据

```bash
python examples/test_data_generator.py -o test_data.json
```

输出示例：
```
正在生成测试数据...（200 条）
预估数据大小：1.53 MB
完成：test_data.json（200 条，1.53 MB）

--- 字段配置 ---
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
  content_vector: Collection(Edm.Single)（128 维，HNSW/cosine）
```

### 步骤 2: 在 Azure AI Search 创建索引并上传数据

```bash
python examples/setup_test_index.py -d test_data.json
```

### 步骤 3: E2E 迁移测试

#### 试运行（无需 Milvus）

```bash
python examples/e2e_migration_test.py --dry-run
```

不需要 Milvus。先用此方式验证 Azure AI Search 端的连接。

#### 仅评估

```bash
python examples/e2e_migration_test.py --assess-only
```

查看 Schema 兼容性报告。

#### 完整迁移测试

```bash
python examples/e2e_migration_test.py
```

执行所有步骤：评估 → 迁移 → 验证。

### 步骤 4: 清理

测试完成后，清理 Azure 资源：

```bash
# 删除测试索引
python examples/setup_test_index.py --delete

# 删除检查点文件
rm -rf .checkpoints_test/
rm -f test_data.json test_assessment_report.json
```

## 4. 测试数据字段类型映射

测试数据验证的类型转换列表：

| Azure AI Search 字段 | Edm 类型 | Milvus DataType | 转换精度 |
|--------------------|----------|-----------------|---------|
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

## 5. 故障排除

### Azure AI Search 连接错误

```
azure.core.exceptions.HttpResponseError: (403) Forbidden
```

→ 确认 API 密钥是否正确。请使用管理密钥（而非查询密钥）。

### Milvus 连接错误

```
pymilvus.exceptions.MilvusException: connection refused
```

→ 确认 Milvus 是否正在运行：
```bash
docker ps | grep milvus
```

### 存储配额错误

```
The index is over the storage quota
```

→ 超出了免费层 50MB 限制。请减少文档数量：
```bash
python examples/test_data_generator.py -n 100 -o test_data.json
```

### 向量维度不匹配

确认测试数据的向量维度（128）与索引定义一致。
测试数据生成器中的 `VECTOR_DIM` 和索引定义中的 `vector_search_dimensions`
必须为相同的值。
