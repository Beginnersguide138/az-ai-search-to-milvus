# Azure AI Search → Milvus 迁移指南

本文档逐步说明如何在 Azure VM 上将数据从 Azure AI Search 迁移到 Milvus。

## 前提条件

- Azure VM（推荐：Standard_D8s_v5 或更高）
- Python 3.10+
- Azure AI Search 端点和 API 密钥（或 Entra ID 认证）
- Milvus 2.6.x（建议在同一 VNet 内）或 Zilliz Cloud 账户

## 步骤 1: 环境搭建

### 1.1 部署 Milvus（自托管）

```bash
# 使用 Docker Compose 部署 Milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.6.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动
docker compose up -d

# 验证
docker compose ps
```

### 1.2 安装迁移工具

```bash
# 克隆仓库
git clone https://github.com/Beginnersguide138/az-ai-search-to-milvus.git
cd az-ai-search-to-milvus

# 创建虚拟环境并安装
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 1.3 准备配置文件

```bash
cp examples/config.example.yaml config.yaml
```

编辑 `config.yaml`：

```yaml
azure_search:
  endpoint: "https://your-search-service.search.windows.net"
  index_name: "your-index-name"
  api_key: ""  # 建议使用环境变量 AZURE_SEARCH_API_KEY
  use_entra_id: false

milvus:
  uri: "http://localhost:19530"
  collection_name: "your_collection_name"
  # 使用 Zilliz Cloud 时
  # use_zilliz: true
  # zilliz_endpoint: "https://xxx.zillizcloud.com"
  # zilliz_api_key: ""

options:
  batch_size: 500
  drop_existing_collection: false
  enable_dynamic_field: true
  # partition_key_field: "tenant_id"  # 多租户场景
```

设置环境变量：

```bash
export AZURE_SEARCH_API_KEY="your-api-key"
# 使用 Zilliz Cloud 时
# export ZILLIZ_API_KEY="your-zilliz-api-key"
```

## 步骤 2: 评估（迁移前分析）

```bash
az-search-to-milvus assess --config config.yaml --output assessment.json
```

该命令输出以下内容：

- Schema 映射表（所有字段的类型映射）
- 迁移可行性判定（FULL / PARTIAL / COMPLEX）
- 不支持功能的警告（Scoring Profiles、Semantic Ranker 等）
- 迁移到 Milvus 可获得的优势
- JSON 报告（可用于 CI/CD 管道）

### 评估结果解读

| 判定 | 含义 |
|---|---|
| **FULL** | 所有字段均可无损转换 |
| **PARTIAL** | 部分字段需要有损转换（可能存在精度降低） |
| **COMPLEX** | 部分字段需要手动处理 |

## 步骤 3: Schema 转换（可选）

如需在数据迁移前预览 Schema 转换结果：

```bash
# 通过 SDK
az-search-to-milvus schema --config config.yaml --output schema.json

# 从 REST API 导出的 JSON 转换
az-search-to-milvus schema --config config.yaml --from-json index_definition.json
```

### 字段自定义

通过 `config.yaml` 中的 `field_overrides` 修改每个字段的设置：

```yaml
options:
  field_overrides:
    content:
      milvus_name: "text_content"  # 重命名字段
      max_length: 32768            # VARCHAR 最大长度
    category:
      milvus_name: "category"
  exclude_fields:
    - "internal_field"  # 排除不需要迁移的字段
```

## 步骤 4: 数据迁移

```bash
# 试运行（不实际写入）
az-search-to-milvus migrate --config config.yaml --dry-run

# 正式迁移
az-search-to-milvus migrate --config config.yaml

# 删除现有 Collection 并重新创建
az-search-to-milvus migrate --config config.yaml --drop-existing
```

### 迁移流程

1. 从 Azure AI Search 批量提取文档
2. 将每个文档转换为 Milvus Schema 格式
3. 批量插入到 Milvus
4. 保存检查点（用于故障恢复）
5. 构建向量索引
6. 将 Collection 加载到内存

### 基于检查点的恢复

如果迁移中断，重新运行相同命令将自动从最后的检查点恢复：

```bash
# 中断后重新运行 → 从检查点恢复
az-search-to-milvus migrate --config config.yaml

# 忽略检查点，从头开始
az-search-to-milvus migrate --config config.yaml --no-resume
```

## 步骤 5: 验证

```bash
az-search-to-milvus validate --config config.yaml
```

验证项目包括：

- **文档数量**: Azure 和 Milvus 的文档数量是否一致
- **字段数量**: Milvus Collection 的字段数量是否符合预期
- **样本数据**: 随机样本的字段值是否正确迁移
- **向量维度**: 向量字段的维度是否正确

## 步骤 6: 应用切换

### 6.1 迁移到 Milvus 客户端

从 Azure AI Search SDK 到 PyMilvus 的改写示例：

**改写前（Azure AI Search）**:
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

**改写后（Milvus）**:
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

### 6.2 过滤器语法转换

| Azure AI Search (OData) | Milvus | 说明 |
|---|---|---|
| `field eq 'value'` | `field == "value"` | 等于 |
| `field ne 'value'` | `field != "value"` | 不等于 |
| `field gt 10` | `field > 10` | 大于 |
| `field ge 10` | `field >= 10` | 大于等于 |
| `field lt 10` | `field < 10` | 小于 |
| `field le 10` | `field <= 10` | 小于等于 |
| `field eq true` | `field == true` | 布尔值 |
| `search.in(field, 'a,b,c')` | `field in ["a", "b", "c"]` | IN 子句 |
| `field/any(t: t eq 'x')` | `array_contains(field, "x")` | 数组搜索 |
| `f1 eq 'a' and f2 gt 5` | `f1 == "a" and f2 > 5` | AND |
| `f1 eq 'a' or f2 gt 5` | `f1 == "a" or f2 > 5` | OR |

## 常见问题

### Q: 迁移速度较慢

- 增大 `batch_size`（最大 1000）
- 确认 Azure VM 和 Milvus 在同一 VNet / 同一区域
- 检查 Milvus 节点资源（CPU/内存）

### Q: 内存不足错误

- 减小 `batch_size`
- 对于高维向量，考虑使用 DiskANN 索引

### Q: 检查点损坏

```bash
# 删除检查点，从头开始
rm -rf .checkpoints/
az-search-to-milvus migrate --config config.yaml --no-resume
```

### Q: 连接 Zilliz Cloud 出错

- 确认 `zilliz_endpoint` 是否正确
- 检查 API 密钥权限
- 检查网络（防火墙 / NSG）
