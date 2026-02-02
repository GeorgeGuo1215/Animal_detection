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

#### 2.2）MCP（Model Context Protocol）工具扩展

本项目支持把 MCP 服务器中的工具“挂载”到本地 Tool Registry。

启用方式（默认开启）：

- 环境变量：`AGENT_ENABLE_MCP=1`
- 配置文件：`agent_api/mcp_servers.json`
- 或直接用环境变量注入：`MCP_SERVER_JSON`

`agent_api/mcp_servers.json` 示例：

```json
{
  "servers": [
    {
      "name": "sample",
      "transport": "stdio",
      "command": "python",
      "args": ["-m", "mcp_server_example"],
      "env": {},
      "enabled": false
    }
  ]
}
```

说明：

- 目前实现的是 **stdio transport**（本地启动 MCP server 进程）
- 会把 MCP tool 注册为 `mcp.{server}.{tool}`，例如 `mcp.sample.search`
- 如果 `enabled=false` 或配置缺失，启动时会跳过，不影响原有工具

#### 2.1）解决“HTTPS 页面调用 HTTP Agent 被浏览器拦截（Mixed Content）”

你的 `web/` 会通过 GitHub Pages 以 **HTTPS** 方式部署，而浏览器会 **强制拦截** 从 HTTPS 页面发往 HTTP 的请求（即使 CORS 全开也没用）。

因此生产环境要么：

- **让 Agent 也提供 HTTPS**（推荐：用域名 + 证书；或用 Nginx/Caddy 反代提供 HTTPS）
- 或者 **让 Web 与 Agent 同源（同域同协议）**，由反代把 `/agent/*` 转发到本机 HTTP 端口（浏览器只看到 HTTPS）

本项目提供了一个更方便的启动入口：当配置证书路径时自动走 HTTPS：

Windows CMD 示例：

```bat
set AGENT_HOST=0.0.0.0
set AGENT_PORT=9001
set AGENT_SSL_CERTFILE=fullchain.pem
set AGENT_SSL_KEYFILE=privkey.pem
python -m agent_api.app.serve
```

如果不设置 `AGENT_SSL_CERTFILE/AGENT_SSL_KEYFILE`，则默认以 HTTP 启动。

#### 启动时预热 RAG 缓存

默认预热（embedding + reranker）

Windows CMD 示例：

```bat
set AGENT_WARMUP_RAG=1
set AGENT_WARMUP_BM25=1
set AGENT_WARMUP_RERANKER=1
REM 如果你想关闭预热：
REM set AGENT_WARMUP_RAG=0
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


