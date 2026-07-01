/**
 * JSON 数据健康分析（plan_and_solve 非流式）
 *
 * 修复：替换 process.env.OPENAI_API_KEY 为 getAgentApiKey()，缺失时引导用户配置。
 */

import state from '../app/state.js';
import { byId } from '../utils/dom.js';
import { showToast } from '../ui/toast.js';
import { planAndSolve, getAgentApiKey, getAgentEndpoint } from './agent-client.js';
import { showAIConfig } from '../ui/modals.js';
import { formatChatMessage } from './chat-formatter.js';

export async function performHealthAnalysis() {
  const apiKey = getAgentApiKey();
  if (!apiKey) {
    showToast('请先在 “AI 配置” 弹窗中填写 Agent API Key', 'warn');
    showAIConfig();
    return;
  }

  const target = byId('healthAnalysisResult') || byId('analysisResult');
  if (target) {
    target.innerHTML = `
      <div class="analysis-loading">
        <span class="spinner"></span>
        <span>正在调用 Agent 分析数据…（端点：${getAgentEndpoint()}）</span>
      </div>`;
  }

  // 取最近的 JSON 数据作为上下文
  const json = (state.processedResults || []).find(r => r?.dataType === 'json');
  const ctx = json ? JSON.stringify({
    animal: json.animal,
    heartRate: json.heartRate,
    respiratoryRate: json.respiratoryRate,
    temperature: json.temperature,
    activity: json.activity,
  }, null, 2) : '（无 JSON 数据）';

  const question = `请基于下面的宠物数据，给出健康评估、潜在风险和后续建议（中文输出）：\n\n\`\`\`json\n${ctx}\n\`\`\``;

  try {
    const r = await planAndSolve({ question });
    const answer = r?.answer || r?.final_answer || JSON.stringify(r, null, 2);
    if (target) {
      target.innerHTML = `
        <div class="analysis-report animate-fade-up">
          <div class="timestamp">${new Date().toLocaleString('zh-CN')}</div>
          ${formatChatMessage(answer)}
        </div>`;
    }
    showToast('健康分析完成', 'success');
  } catch (e) {
    if (target) {
      target.innerHTML = `<div class="text-muted">❌ 分析失败：${e.message}</div>`;
    }
    showToast(`分析失败：${e.message}`, 'error');
  }
}
