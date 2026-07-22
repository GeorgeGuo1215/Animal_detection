# RAG 二级分类索引与专家限域检索

## 动机

原先四个 MoE 专家（clinical / nutrition / pharmacy / behavior）共享同一套全量知识库（`rag_index_e5`，约 16.7 万 chunks）。药学问题也常被内科大部头书目挤占 top-k，检索噪声大。

本次按《兽医医学资料分类》Excel 的**二级类**切分子索引，查询接口支持 `category` 限域；专家调用 `rag.search` 时强制注入本职类别。

## 修改文件清单（无遗漏核对）

### 新增

| 文件 | 作用 |
| --- | --- |
| [`agentAndRag/RAG/tools/split_index_by_category.py`](../agentAndRag/RAG/tools/split_index_by_category.py) | 从全量索引按 Excel 二级类切分子库 |
| [`agentAndRag/RAG/tools/__init__.py`](../agentAndRag/RAG/tools/__init__.py) | 包初始化 |
| [`agentAndRag/RAG/simple_rag/category_index.py`](../agentAndRag/RAG/simple_rag/category_index.py) | `category` → 子库目录解析（支持 `clinical.*`） |
| [`agentAndRag/RAG/data/category_taxonomy.json`](../agentAndRag/RAG/data/category_taxonomy.json) | 类别权威表（id / 中文 / 书目 / source_path / chunk_count） |
| [`agentAndRag/RAG/data/rag_index_e5_by_cat/<category_id>/`](../agentAndRag/RAG/data/rag_index_e5_by_cat/) | 46 个二级类子库（`embeddings.npy` + `meta.jsonl` + `store_config.json`） |
| [`agentAndRag/RAG/tests/__init__.py`](../agentAndRag/RAG/tests/__init__.py) | 测试包 |
| [`agentAndRag/RAG/tests/test_category_split.py`](../agentAndRag/RAG/tests/test_category_split.py) | 切分完整性 / 空库 / 一书多类 |
| [`agentAndRag/agent_api/tests/rag/__init__.py`](../agentAndRag/agent_api/tests/rag/__init__.py) | 测试包 |
| [`agentAndRag/agent_api/tests/rag/test_category_search.py`](../agentAndRag/agent_api/tests/rag/test_category_search.py) | 分类检索精准度指标 |
| [`agentAndRag/agent_api/tests/moe/test_expert_rag_categories.py`](../agentAndRag/agent_api/tests/moe/test_expert_rag_categories.py) | 专家强制注入 `category` |
| [`docs/rag_category_indexes.md`](rag_category_indexes.md) | 本文档 |

### 修改

| 文件 | 改动要点 |
| --- | --- |
| [`agentAndRag/agent_api/app/tools/rag_tools.py`](../agentAndRag/agent_api/app/tools/rag_tools.py) | `rag_search_tool(..., category=...)`；多子库召回后按 score 合并 |
| [`agentAndRag/agent_api/app/tools/tools_builtin.py`](../agentAndRag/agent_api/app/tools/tools_builtin.py) | `rag.search` input_schema 增加 `category` |
| [`agentAndRag/agent_api/app/schemas/tool_api.py`](../agentAndRag/agent_api/app/schemas/tool_api.py) | `RagSearchRequest.category` |
| [`agentAndRag/agent_api/app/services/moe/experts.py`](../agentAndRag/agent_api/app/services/moe/experts.py) | `ExpertConfig.rag_categories`；执行时强制写入 arguments |
| [`agentAndRag/agent_api/app/main.py`](../agentAndRag/agent_api/app/main.py) | 启动时可选预热专家子库（`AGENT_WARMUP_CATEGORIES`） |
| [`agentAndRag/RAG/query.py`](../agentAndRag/RAG/query.py) | CLI `--category` |

### 未改动（刻意保留）

| 路径 | 说明 |
| --- | --- |
| `agentAndRag/RAG/data/rag_index_e5/` | 全量索引保留；未传 `category` 时仍走全量库 |

### 测试产物（可选，本地生成）

| 文件 | 说明 |
| --- | --- |
| `agentAndRag/agent_api/tests/rag/_last_precision_metrics.json` | 精准度对比落盘 |
| `agentAndRag/agent_api/tests/rag/_live_category_api.json` | HTTP 联调落盘 |

## 目录与配置

| 路径 | 说明 |
| --- | --- |
| `RAG/data/rag_index_e5/` | 全量索引 |
| `RAG/data/rag_index_e5_by_cat/<category_id>/` | 二级类子库 |
| `RAG/data/category_taxonomy.json` | 类别权威表 |

重建子库（conda 环境 `RAG`）：

```bash
cd agentAndRag
python -m RAG.tools.split_index_by_category
# 可选指定 xlsx / 源索引 / 输出目录：
# python -m RAG.tools.split_index_by_category --xlsx PATH --src-index PATH --out-root PATH
```

当前切分结果：**46** 个二级类目录，其中 **35** 个非空；Excel 书目 **65/65** 对齐成功；无书类为空占位库（`embeddings` shape `(0,384)`）。

主要非空类 chunk 数示例：`clinical.internal_medicine` 27904、`clinical.surgery` 22837、`diagnostics.imaging` 14631、`pharmacy.papich` 3331、`behavior.dog_cat_problems` 3262、`exotic.default` 10135（完整表见 `category_taxonomy.json`）。

一书多类（如影像解剖 Atlas）会**复制**进多个子库，属预期行为。

## 查询接口

### `rag.search` / `POST /tools/rag/search`

新增参数 `category`：`string | string[] | null`。

- 省略：行为与改造前一致，检索全量 `rag_index_e5`
- 传 id：只在对应子库检索；多个 id / 通配（如 `pharmacy.*`）时分库召回后按 score 合并 top-k
- 空库：返回 `hits=[]`，不报错

CLI：

```bash
python -m RAG.query "Papich drug dosage" --category pharmacy.* --top-k 5
```

### MoE 专家绑定

实现位置：`agentAndRag/agent_api/app/services/moe/experts.py`（`ExpertConfig.rag_categories`，执行时**强制覆盖** planner 传入的 category）。

| 专家 | 绑定类别 |
| --- | --- |
| clinical | `basic.anatomy`, `basic.terminology`, `clinical.*`, `diagnostics.*`, `clinical_skills.*`, `integrative.general`, `anesthesia.default`, `immunology.default`, `reproduction.default`, `infectious.placeholder`, `exotic.default`, `zoonosis.toxoplasmosis`（不含马科大动物细类） |
| nutrition | `nutrition.placeholder`, `equine.nutrition`, `integrative.general` |
| pharmacy | `pharmacy.*`, `basic.pharmacology_fundamentals`, `anesthesia.default` |
| behavior | `behavior.*` |

环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `AGENT_WARMUP_CATEGORIES` | `1` | 启动时预热专家相关非空子库的 dense 向量（BM25 仍懒加载） |
| `AGENT_WARMUP_RAG` | `1` | 是否做 RAG warmup（含全量库） |

## 指标（测试与联调）

### Top-5 类内命中率（药学 query）

用例：`Papich veterinary drug dosage contraindication toxicity dog cat`，dense-only，`top_k=5`。

| 模式 | top-5 类内命中率（来源属于 `pharmacy.*`） |
| --- | --- |
| 全量库 | **0.20**（5 条中仅 1 条药学书，其余多为 Ettinger 内科） |
| `category=pharmacy.*` | **1.00**（全部来自 Papich / Applied Pharmacology） |
| 提升（lift） | **+0.80** |

测试：`agent_api/tests/rag/test_category_search.py`。

### 行为限域

`category=behavior.*` 时，命中来源均落在行为学书目（如 *Behavior Problems of the Dog and Cat*）。

### 空类占位

`category=nutrition.placeholder` → `hits=[]`，接口仍成功。

### 后端联调

```bash
set PYTHONPATH=agentAndRag;agentAndRag\agent_api
set AGENT_DISABLE_AUTH=1
set AGENT_WARMUP_CATEGORIES=0
python -m uvicorn agent_api.app.main:app --host 127.0.0.1 --port 8145
```

`POST /tools/rag/search`（客户端建议 `httpx.Client(trust_env=False)` 避开坏代理）：

- `pharmacy.*`：5 hits，category 为 `pharmacy.papich` / `pharmacy.applied_pharmacology`
- `behavior.*`：5 hits，全部行为学书目
- `nutrition.placeholder`：0 hits，`ok=true`

## 测试命令

```bash
# 切分完整性
pytest RAG/tests/test_category_split.py -q

# 检索精准度 + 限域
pytest agent_api/tests/rag/test_category_search.py -q

# 专家强制注入 category
pytest agent_api/tests/moe/test_expert_rag_categories.py -q

# 既有回归
pytest agent_api/tests/moe agent_api/tests/sql_search -q
```

## 占位与后续

- 营养学小动物专著目前为空（`nutrition.placeholder`）；营养专家暂时依赖 `equine.nutrition` + `integrative.general`，补书后重跑切分即可。
- 个体差异 / 中兽医 / 指南 / 职业 等注释型空类已占位，暂未绑定专家。
- 分类表来源：本地微信文件 `兽医医学资料分类(1).xlsx`（切分脚本默认路径；可用 `--xlsx` 覆盖）。
