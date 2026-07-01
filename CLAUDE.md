# Claude.md · Animal_detection

使用中文作为主要回复语言，请遵循并维护当前文件。

## 项目概述

`Animal_detection` 是宠物健康 Agent 原型，包含：

- `agentAndRag/`：Python 后端，提供 OpenAI 兼容 `/v1/chat/completions`（SSE 流式），
  在每个 chunk 附带 `agent_status` / `agent_detail` 描述工具调用进度，并集成 RAG / SQL 检索 / MCP。
- `web/`：前端原型，**信号处理算法与 UI 未分离**（`src/processors/*.js`）。

> 该 Agent 已被 `PetHealth_Server` 以代理模式接入；信号处理算法已在 `RadarVital` 项目中前后端分离。

---

# 变更记录：修复"工具调用进度不显示"的前端交互 bug

## 一、整体
前端在工具调用阶段，上游会下发"仅有正文、`agent_status` 为 `null`"的明细 chunk
（如 `**Round 1 tool call**` 之类）。原逻辑把这类 chunk 直接丢弃，导致**工具进度区一片空白**，
用户看不到 Agent 正在做什么。修复方案：引入 `phase` 阶段标记，区分"工具阶段"与"作答阶段"，
工具阶段的无状态正文按 `tool_detail` 写入进度区而非丢弃。

## 二、功能 → 文件
| 功能 | 文件 |
| --- | --- |
| 流式回调阶段判定 | `web/src/health/health-chat.js` |
| 进度图标格式化 | `web/src/health/chat-formatter.js` |

## 三、函数级
### `web/src/health/health-chat.js`
- 在 `streamChatCompletions(...)` 回调外引入 `let phase = 'tool'`，跟踪流阶段。
- 当出现 `streaming` 状态（或进入作答阶段）时 `phase = 'answer'`，正文计入回答。
- 当 `agentStatus == null && phase === 'tool'` 时，把该 chunk 文本作为
  `{ icon: '·', text, status: 'tool_detail' }` 推入 `statusLogs`（原先被丢弃）。
- 作答阶段的无状态正文（`isGenerating && agentStatus == null && phase === 'answer'`）继续作为正文渲染。

### `web/src/health/chat-formatter.js`
- `formatAgentStatus(...)`：为新的 `tool_detail` 状态补充图标，使其在进度区正确显示。
