# LivestockMind 面试/演示指南

## 产品定位

LivestockMind 是面向**牛、猪、羊、马等大型畜牧**的兽医专家 Agent 系统，与 PetMind（犬猫）、PandaMind（大熊猫）共用同一套 **Plan-and-Solve + RAG + MCP + Reasoner** 流程。

## 端口与入口

| 服务 | 端口 | 聊天页 |
|------|------|--------|
| LivestockMind | 8000 | http://127.0.0.1:8000/chat |
| PandaMind | 8001 | http://127.0.0.1:8001/chat |
| PetMind | 8002 | http://127.0.0.1:8002/chat |

## 启动

```bash
cd /home/sam/Animal_Detection/Animal_detection/Livestock_agent
bash start_livestock_agent.sh
```

## 模型与角色

- 默认模型：`livestock-plan-solve`
- 复杂兽医推理：`REASONER_MODEL=deepseek-reasoner`
- 用户角色：`farmer`（牧场主）/ `veterinarian`（兽医）

## RAG 知识库

LivestockMind 与 **PetMind 共用同一套向量索引**（`agentAndRag/RAG/data/rag_index_e5`，约 16 万+ chunks，embedding: `intfloat/multilingual-e5-small`）。

环境变量（已在 `start_livestock_agent.sh` / `.env` 默认配置）：

```bash
RAG_INDEX_DIR=../agentAndRag/RAG/data/rag_index_e5
RAG_RAW_DIR=../agentAndRag/RAG/data/raw
```

重建索引请在 `agentAndRag` 目录执行 ingest；LivestockMind 会自动读取同一路径。

## MCP 工具

- `rag.search` — 与 PetMind 共用的兽医知识库检索
- `mcp.web_search.web_search` — 联网补充（需用户确认）
- `mcp.vital_signs_analyzer.analyze_vitals` — 牛/猪/羊/马 HR/RR/体温时序分析

## 演示问题示例

- 奶牛酮病的早期识别与牧场处置建议
- 育肥猪 PRRS 与继发感染的监测要点
- 羊群围产期常见代谢病预防
- 马匹绞痛（colic）的观察指标与何时必须请兽医
