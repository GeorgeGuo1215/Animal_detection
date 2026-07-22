# 兽医学生反馈对照评测报告（结果导向）

> 评测目的：验证本周改造是否覆盖组内兽医学生对**上一版 PetMind** 的批评点，并与 Deepseek / GPT 评估维度对齐。  
> 上一版基线来自评估文档记载（不重跑旧二进制）；本周结果来自本机 MoE HTTP live（`user_role=veterinarian`）。  
> **提示词修订（2026-07-22）**：病例工作流答风已改为「基本信息/主诉/现病史/既往史」分点、检查节内先紧急处理、文末强制「风险提示」，并禁止 AI 风字段标签；live 抽检见 `vet_eval_20260722_153502`（指令一/二均 PASS）。

## 1. 评测来源

| 材料 | 用途 |
| --- | --- |
| `各AI模型评估.docx` | PetMind / Deepseek / GPT 对照结论（上一版 PetMind 短板） |
| `具体指令和各AI回答.docx` | 指令一、指令二原文与参考回答结构 |
| 本周代码 | user_role、病例工作流、品种护栏、RAG 二级类限域 |

## 2. 上一版 PetMind 基线（评估文档）

摘自《各AI模型评估》中对 **petmind** 列的结论：

| 用例 | 上一版 PetMind |
| --- | --- |
| 指令一（金毛呕吐 + 整理与诊断） | **无病例整理**；逻辑偏「结论→依据→风险→建议行动→治疗→急诊」；有风险提示；有品种常发病提示 |
| 指令二（英短排尿困难 + 整理病例格式） | 有病理/病例整理；急症提示在最前（√）；**不够系统，无问题列表** |

Deepseek / GPT 在评估中普遍具备：病例整理、临床问题列表、初步诊断与检查方案。本周目标是：**在结构化维度上对齐「有整理 + 有问题列表」**，而非逐句复刻竞品文风。

## 3. 评测清单与对比表

评分说明：`有` / `无` / `部分`；本周列由 live 自动勾选（预热完成后结果）。

### 3.1 指令一 — 金毛 + 鸡骨头（整理和诊断）

| 检查项 | 上一版 PetMind | Deepseek（评估） | GPT（评估） | **本周 PetMind（live）** |
| --- | --- | --- | --- | --- |
| 病例整理 | 无 | 有 | 有 | **有** |
| 临床问题列表（POMR） | 无（未作为分节） | 有 | 有 | **有** |
| 初步鉴别 / 诊断方向 | 有（结论导向） | 有 | 有 | **有** |
| 风险 / 急诊提示 | 有 | 有 | 有 | **有** |
| 检查与治疗（轻重缓急） | 有（偏行动建议） | 有 | 更详尽 | **有** |
| 品种相关提示 | 有 | — | — | **有** |

### 3.2 指令二 — 英短排尿困难（标准病例格式）

| 检查项 | 上一版 PetMind | Deepseek（评估） | GPT（评估） | **本周 PetMind（live）** |
| --- | --- | --- | --- | --- |
| 急症提示前置 / 显著 | 有（√） | — | — | **有** |
| 病例整理 | 有 | 有（术语更严） | 有 | **有** |
| 问题列表 | **无** | 有 | 有 | **有** |
| 检查与治疗详尽度 | 不足 | 较详 | 更详 + 用药具体 | **有** |

### 3.3 附加 — 法斗剧烈运动（品种特异化）

| 检查项 | 期望 | **本周 PetMind（live）** |
| --- | --- | --- |
| 短吻 / 热耐受 / 运动或呼吸边界 | 应出现 | **有** |
| 不强制完整病例三件套 | 应满足 | **有** |

## 4. 客观指标（非主观文风）

### 4.1 RAG 二级类限域（已测）

| 指标 | 全量库 | 分类限域 `pharmacy.*` | lift |
| --- | --- | --- | --- |
| 药学 query top-5 类内命中率 | 0.20 | 1.00 | **+0.80** |

详见 [`rag_category_indexes.md`](rag_category_indexes.md)。

### 4.2 Live 分节命中率

报告目录：`C:/Users/ROG/Animal_detection/agentAndRag/agent_api/tests/moe/reports/vet_eval_20260722_172907`

| 用例 | 期望分节/关键词 | 结果 | 报告目录 |
| --- | --- | --- | --- |
| golden_vomit | 病例整理、问题列表、检查与治疗方案 | PASS | `vet_eval_20260722_172907` |
| flutd_organize | 病例整理、问题列表 | PASS | `vet_eval_20260722_172907` |
| bulldog | 短吻/热/运动/呼吸；非强制三件套 | PASS | `vet_eval_20260722_172907` |
| **合计 PASS** | — | **3/3** | — |

## 5. 结论

- Live 合计：**3/3 PASS**；详细产物见 `C:/Users/ROG/Animal_detection/agentAndRag/agent_api/tests/moe/reports/vet_eval_20260722_172907`。
- 相对上一版 PetMind：指令一病例整理 **有**、问题列表 **有**；指令二问题列表 **有**（上一版评估为「无」）。
- 法斗特异化关键词：**有**；未强制病例三件套：**有**。
- RAG 二级类限域客观指标：药学 top-5 类内命中率 0.20→1.00（lift +0.80），见 [`rag_category_indexes.md`](rag_category_indexes.md)。
- 结构化对齐评估：**已覆盖**兽医学生指出的「无病例整理 / 无问题列表」核心短板。

## 6. 附录

- Live 脚本：`agentAndRag/agent_api/tests/moe/run_moe_vet_eval_live.py`
- Live 输出：`agentAndRag/agent_api/tests/moe/reports/vet_eval_*`
- 周报解决方案：[`weekly_solutions_user_role_case_rag.md`](weekly_solutions_user_role_case_rag.md)
- RAG 分库文档：[`rag_category_indexes.md`](rag_category_indexes.md)
- 终答摘录：见各 `case_*_answer.txt`（指令一以 `vet_eval_20260721_150944_golden_retry` 为准）
