## Integration（对接层）

作为 **n8n Webhook** 与业务 JSON（`Event`）之间的统一接入层：校验事件 → `httpx` 转发至 `N8N_WEBHOOK_URL`。

### 推荐：合并进 PetMind Agent（仅 8000）

仓库已将本包的 `routes_ingest` **挂载**到 Agent API：

- **`POST http://127.0.0.1:8000/integration/ingest`**：请求体为 `Event` JSON（与单独运行 integration 时相同）。
- 该路径在 Agent 上**免 API Key**（便于浏览器 / BLE 上报）；其余 `/tools/*`、`/v1/*` 仍按原鉴权。

在 **`agentAndRag`** 目录配置环境变量（可与 `agent_api` 共用 `.env`）：

```bash
set N8N_WEBHOOK_URL=http://127.0.0.1:5678/webhook/your-id
set N8N_TIMEOUT_SEC=60
```

复制示例：`integration/.env.example`。

启动 Agent：

```bash
cd agentAndRag
python -m uvicorn agent_api.app.main:app --host 127.0.0.1 --port 8000
```

### 可选：本包独立调试（9001）

不写 Agent 时仍可单独启动，便于只测转发：

```bash
python -m pip install -r integration/requirements.txt
python -m uvicorn integration.api.main:app --host 127.0.0.1 --port 9001
```

此时入口为 **`POST http://127.0.0.1:9001/ingest`**，无 Bearer；CORS 全开。

### 发送事件

向上述 **`/ingest`**（或合并后的 **`/integration/ingest`**）发送 JSON，字段需满足 `integration/schemas/event.py`（`event_id`、`ts`、`animal`、可选 `device` / `window` / `context` / `signals`）。

### 静态网页说明

浏览器直接打开 `file://` 可能受 CORS 限制；请用本地 HTTP 服务打开 `web/index.html`，或确保 Agent 已设置 `AGENT_ENABLE_CORS=1`。
