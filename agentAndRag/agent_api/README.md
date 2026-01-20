## Agent Tools API（FastAPI，简易的agent及调用封装）

目标：把 `RAG/` 封装成稳定的 Tools（JSON I/O），供 n8n / 大模型 Function Calling 调用，并留出“评估闭环”的接口扩展点和其他工具的拓展点。

### 1) 安装

```bash
pip install -r agent_api/requirements.txt
pip install -r RAG/requirements.txt
pip install -r lora/requirements.txt
```

### 2) 启动

在仓库根目录运行：

```bash
python -m uvicorn agent_api.app.main:app --host 127.0.0.1 --port 8000
```

#### 启动时预热 RAG 缓存

默认不预热

Windows CMD 示例：

```bat
set AGENT_WARMUP_RAG=1
set AGENT_WARMUP_BM25=1
set AGENT_WARMUP_RERANKER=0
REM 如果你想让 embedding/reranker 上 GPU（你机器得有可用 CUDA）：
REM set AGENT_WARMUP_DEVICE=cuda
python -m uvicorn agent_api.app.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

### 3) Tool：RAG 检索

```bash
curl -X POST http://127.0.0.1:8000/tools/rag/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is BRDC?\",\"top_k\":5,\"multi_route\":true,\"rerank\":true,\"expand_neighbors\":1}"
```

### 3.1) Tool Registry

列出所有工具：

```bash
curl http://127.0.0.1:8000/tools
```

通用调用：

```bash
curl -X POST http://127.0.0.1:8000/tools/call ^
  -H "Content-Type: application/json" ^
  -d "{\"tool_name\":\"rag.search\",\"arguments\":{\"query\":\"What is BRDC?\",\"top_k\":5}}"
```

### 4) Tool：重建索引（入库）

```bash
curl -X POST http://127.0.0.1:8000/tools/rag/reindex ^
  -H "Content-Type: application/json" ^
  -d "{\"batch_size\":32}"
```

### 5) Tool：LoRA checkpoint 指标评估（后台 job）

```bash
curl -X POST http://127.0.0.1:8000/tools/lora/eval_checkpoints ^
  -H "Content-Type: application/json" ^
  -d "{\"model_id\":\"Qwen/Qwen2.5-7B-Instruct\",\"output_dir\":\"out/qwen2.5-7b-animals\",\"checkpoints\":[\"checkpoint-138\",\"checkpoint-204\"],\"device_map\":\"cuda\"}"
```

查询 job 状态：

```bash
curl http://127.0.0.1:8000/jobs/<job_id>
```

### 6) Trace（审计/论文截图）

默认写到仓库根目录：`agent_api_logs/trace.jsonl`  
可通过环境变量覆盖：

- `AGENT_TRACE_DIR=...`

### 7) Agent：Plan-and-Solve（一键：规划→调用 tools→回答）

你可以把这当成“最小可用的 Agent Core”，n8n 里用一个 HTTP Request 节点就能跑通。

#### 环境变量（推荐）

- `OPENAI_BASE_URL`：OpenAI-compatible 网关地址（例如 DeepSeek / 自建网关）
- `OPENAI_API_KEY`：API Key（也兼容 `DEEPSEEK_API_KEY`）
- `OPENAI_MODEL`：模型名（也兼容 `DEEPSEEK_MODEL`）

#### 调用示例

```bash
curl -X POST http://127.0.0.1:8000/agent/plan_and_solve ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"犬猫腹泻常见原因有哪些？需要注意哪些红旗症状？\",\"temperature\":0.2,\"max_tokens\":800}"
```

你也可以在请求里临时传入（方便在 n8n 里配置不同模型）：

```bash
curl -X POST http://127.0.0.1:8000/agent/plan_and_solve ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"猪群高热、咳嗽可能是什么？\",\"llm_base_url\":\"https://api.openai.com\",\"llm_api_key\":\"sk-...\",\"llm_model\":\"gpt-4o-mini\"}"
```


