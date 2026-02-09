# 从 Azure AI Search 迁移到 Milvus 的优势

Milvus 提供了许多 Azure AI Search 所不具备的功能。
本文档详细介绍迁移后可获得的具体优势。

## 1. 丰富的向量索引类型

Azure AI Search 仅支持 **HNSW** 和 **Exhaustive KNN** 两种索引，
而 Milvus 支持 **12 种以上**的索引类型。

### CPU 索引

| 索引 | 说明 | 使用场景 |
|---|---|---|
| **HNSW** | 基于图的 ANN | 高精度、低延迟。与 Azure AI Search 相同 |
| **FLAT** | 暴力搜索 | 需要 100% 召回率时 |
| **IVF_FLAT** | 倒排文件 + 扁平 | 内存效率与速度的平衡 |
| **IVF_SQ8** | IVF + 标量量化 | 内存减少 70-75% |
| **IVF_PQ** | IVF + 乘积量化 | 大规模数据集的大幅内存优化 |
| **SCANN** | 分数感知量化 | 比 IVF_PQ 更高精度 |
| **DiskANN** | 基于磁盘的 ANN | 内存无法容纳的超大规模数据 |

### GPU 索引（Azure NC/ND 系列 VM）

| 索引 | 说明 | 使用场景 |
|---|---|---|
| **GPU_IVF_FLAT** | GPU 上的 IVF_FLAT | 高吞吐量搜索 |
| **GPU_IVF_PQ** | GPU 上的 IVF_PQ | GPU + 大规模数据集 |
| **GPU_CAGRA** | 基于 NVIDIA RAFT | GPU 上的最佳性能 |
| **GPU_BRUTE_FORCE** | GPU 暴力搜索 | 中小规模 100% 召回率 |

### 稀疏向量索引

| 索引 | 说明 | 使用场景 |
|---|---|---|
| **SPARSE_INVERTED_INDEX** | 稀疏倒排索引 | BM25/SPLADE 混合搜索 |
| **SPARSE_WAND** | WAND 算法 | 大规模稀疏向量 |

## 2. 原生混合搜索

Milvus 原生支持 `SPARSE_FLOAT_VECTOR` 类型，可以在**同一个 Collection 中**
组合稠密向量（embedding）和稀疏向量（BM25/SPLADE）进行混合搜索。

```python
from pymilvus import AnnSearchRequest, RRFRanker

# 稠密向量搜索
dense_req = AnnSearchRequest(
    data=[dense_embedding],
    anns_field="dense_vector",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=10,
)

# 稀疏向量搜索
sparse_req = AnnSearchRequest(
    data=[sparse_embedding],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=10,
)

# 使用 RRF（Reciprocal Rank Fusion）融合
results = collection.hybrid_search(
    [dense_req, sparse_req],
    ranker=RRFRanker(k=60),
    limit=10,
)
```

作为 Azure AI Search 语义排序器的替代方案，混合搜索 + 自定义重排序模型的组合非常有效。

## 3. 分区键（多租户）

```python
from pymilvus import FieldSchema, DataType

# 将租户 ID 设置为分区键
tenant_field = FieldSchema(
    name="tenant_id",
    dtype=DataType.VARCHAR,
    max_length=64,
    is_partition_key=True,  # ← Milvus 特有功能
)
```

- 数据按租户物理隔离
- 查询时自动应用分区裁剪
- Azure AI Search 中需要通过过滤器模拟的功能，这里原生支持

## 4. 动态 Schema

```python
schema = CollectionSchema(
    fields=[...],
    enable_dynamic_field=True,  # 可存储 Schema 中未定义的字段
)

# 可以插入 Schema 中未定义的字段
collection.insert([{
    "id": 1,
    "vector": [0.1, 0.2, ...],
    "title": "Document",
    "custom_metadata": {"key": "value"},  # 动态字段
}])
```

在 Azure AI Search 中，添加字段需要更新索引。
而 Milvus 可以灵活存储 Schema 中未定义的字段。

## 5. 成本管控

| 项目 | Azure AI Search | Milvus（自托管） |
|---|---|---|
| 计费模式 | 按搜索单位 + 存储 | 仅 VM 成本 |
| 查询费用 | 包含在 SU 使用量中 | 无 |
| 扩展方式 | 购买额外 SU | VM 纵向/横向扩展 |
| 最低成本 | Standard S1: 约 $250/月 | Azure VM D4s_v5: 约 $140/月 |
| 大规模时 | 成本急剧增长 | 线性扩展 |

### Zilliz Cloud（托管 Milvus）

- 基于 CU（Capacity Unit）的计费
- Serverless 选项提供按用量计费
- 降低自托管的运维成本

## 6. 范围搜索

```python
# 基于距离阈值的搜索（Azure AI Search 不具备此功能）
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

除了 Top-K 之外，还可以搜索"距离在 0.1 至 0.5 范围内的所有文档"，实现基于阈值的搜索。

## 7. 分组搜索

```python
# 按字段分组的搜索（Azure AI Search 不具备此功能）
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=10,
    group_by_field="category",  # 每个类别的 Top-K
)
```

## 8. 迭代器 API

```python
# 高效内存的大量结果检索
# 没有 Azure AI Search 的 $skip 100,000 条限制
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

## 9. CDC（变更数据捕获）

Milvus CDC 可以实时捕获数据变更并同步到其他系统。
Azure AI Search 不具备此功能。

- DR（灾难恢复）配置
- 与数据管道集成
- 审计日志生成

## 10. Null/默认值支持（Milvus 2.6.x）

```python
# Milvus 2.6.x 新功能
field = FieldSchema(
    name="description",
    dtype=DataType.VARCHAR,
    max_length=1024,
    nullable=True,           # 允许 NULL 值
    default_value="",        # 默认值
)
```

虽然 Azure AI Search 也可以存储 null 值，但 Milvus 2.6.x 允许在 Schema 级别
显式定义 NULL 约束和默认值，提升了数据完整性管理能力。

## 总结

迁移到 Milvus 可获得以下优势：

1. **索引选择** — 可以根据工作负载选择最优索引
2. **GPU 利用** — 在 Azure GPU VM 上实现数量级的搜索性能提升
3. **混合搜索** — 原生支持稠密 + 稀疏向量
4. **成本优化** — 仅 VM 成本，可预测的价格体系
5. **可扩展性** — 无 $skip 限制，迭代器 API 处理大规模数据
6. **灵活性** — 动态 Schema、分区键、范围搜索
7. **运维** — CDC、Null/默认值支持、丰富的标量索引
