# 本周解决方案摘要（可直接贴周报）

> 对应上周四类问题：用户身份差异、临床结构化、品种特异化、RAG 知识库粒度。下列每条均为**已落地**方案。

---

## 一句话总览

本周围绕 MoE Agent 做了四条主线改造：用 `user_role` 区分宠主/兽医答风；兽医具体病例触发「病例整理 → POMR 问题列表 → 检查与治疗」结构化终答；注入物种/品种并强制专家特异化；按二级学科切分 RAG 子库并由专家限域检索，降低无关书目干扰。

---

## 问题 1：用户身份处理模糊，两类用户输出差异不足

**方案**

- 请求增加 `user_role`：`pet_owner` | `veterinarian`。
- 注入 Router / Aggregator / Critic：兽医侧偏病历与鉴别、检查方案；宠主侧偏可执行建议与就医提示，避免同一套「科普口吻」。

**落点**

- `agentAndRag/agent_api/app/services/plan_and_solve.py`（`build_solve_prompt`）
- `agentAndRag/agent_api/app/services/moe/orchestrator.py`、`router.py`、`critic.py`
- OpenAI 兼容入口透传 `user_role`（`routes_openai.py`）

**可观测效果**

- 同一病例在 `veterinarian` 下可触发病例工作流分节；`pet_owner` 下不强制 POMR 三件套，答风更偏宠主沟通。

---

## 问题 2：临床信息结构化差，兽医视角缺少问题列表与病历整理

**方案**

- 门控 `is_concrete_vet_case(query)`：需「有具体病例叙述」且意图为「整理病例/病例格式」或「诊断/鉴别」。
- 仅兽医角色 + 门控通过时，Aggregator 强制输出：
  1. **病例整理**
  2. **问题列表**（POMR：临床问题条目 + 鉴别，不是向宠主追问清单）
  3. **检查与治疗方案**（可含轻重缓急）
- 仅有病情但只问运动建议等 → **不**触发三件套（避免法斗运动评估被模板绑架）。

**落点**

- `plan_and_solve.py`（门控 + 兽医分节指令）
- `orchestrator.py`（病例工作流时提高终答 `max_tokens`）
- 验收：`tests/moe/test_vet_case_prompt.py`、`run_moe_vet_live.py` / `run_moe_vet_eval_live.py`

**可观测效果**

- 相对兽医学生评估中「上一版 PetMind：指令一无病例整理；指令二无问题列表」，本周在同类指令下 live **3/3 PASS**，稳定出现「病例整理 / 问题列表」分节（见 [`eval_report_vet_student_comparison.md`](eval_report_vet_student_comparison.md)）。
- 门控已覆盖评估原文「应该怎么整理和诊断」等表述。

---

## 问题 3：对特定品种缺乏敏感性（如法斗不宜剧烈运动）

**方案**

- 从 `animal_id` 画像或用户文本注入 `species` / `breed` 到 Router、专家 prompt 与 RAG query。
- 专家侧 `_SPECIES_BREED_GUARD`：要求在 risks/conclusion 中写明品种特异风险（短吻犬运动/热耐受/呼吸道等）。

**落点**

- `experts.py`、`router.py`、`orchestrator.py`
- 用例：法斗剧烈运动评估（live 关键词：短吻 / 热 / 运动 / 呼吸）

**可观测效果**

- 法斗场景终答应显式区分于普通中型犬的运动边界，而非泛化「多运动」。

---

## 问题 4：专家共享全量知识库，检索易被无关内容干扰

**方案**

- 按《兽医医学资料分类》Excel **二级类**从 `rag_index_e5` 切出 46 个子库（35 非空 + 空占位）。
- `rag.search` 增加 `category`（支持 `pharmacy.*` 等通配）；多类合并按 score 取 top-k。
- 四专家配置 `rag_categories`，调用时**强制注入**（不信任 planner 乱填）。

**落点**

- `RAG/tools/split_index_by_category.py`、`RAG/data/category_taxonomy.json`、`RAG/data/rag_index_e5_by_cat/`
- `RAG/simple_rag/category_index.py`、`agent_api/app/tools/rag_tools.py`、`experts.py`
- 文档：`docs/rag_category_indexes.md`

**可观测效果（客观指标）**

| 指标 | 全量库 | `category=pharmacy.*` | 提升 |
| --- | --- | --- | --- |
| 药学 query top-5 类内命中率 | 0.20 | 1.00 | +0.80 |

---

## 本周交付物索引

| 文档 | 内容 |
| --- | --- |
| 本文 | 周报解决方案摘要 |
| [`eval_report_vet_student_comparison.md`](eval_report_vet_student_comparison.md) | 对照兽医学生评估表的结果/评测报告 + live 对比 |
| [`rag_category_indexes.md`](rag_category_indexes.md) | RAG 分库改造说明与指标 |
