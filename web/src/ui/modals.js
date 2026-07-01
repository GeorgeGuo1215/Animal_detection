/**
 * Modal 控制模块：AI 配置 / Prompt 编辑 / RAG 编辑 / 通用关闭。
 *
 * 修复：补齐 11 个原本未实现的 onclick 函数：
 *   createNewPrompt / savePrompt / deletePrompt / previewPrompt
 *   addRAGEntry / saveRAGEntry / cancelRAGEdit / importRAGData / exportRAGData
 *   generateDiagnosticReport / exportReport
 */

import { byId, $, h } from '../utils/dom.js';
import { getItem, setItem } from '../utils/storage.js';
import { STORAGE_KEYS } from '../app/config.js';
import state from '../app/state.js';
import { showToast } from './toast.js';
import { setAgentEndpoint, setAgentApiKey, setAgentModel, getAgentEndpoint, getAgentApiKey, getAgentModel, checkAgentHealth } from '../health/agent-client.js';
import { formatChatMessage } from '../health/chat-formatter.js';

// ========== 通用 ==========
export function openModal(id) {
  const m = byId(id);
  if (!m) return;
  m.style.display = 'flex';
  m.classList.add('is-open');
}
export function closeModal(id) {
  const m = byId(id);
  if (!m) return;
  m.classList.remove('is-open');
  m.style.display = 'none';
}

// ========== AI 配置 ==========
export function showAIConfig() {
  const epEl = byId('azureEndpoint');
  const keyEl = byId('azureApiKey');
  const depEl = byId('azureDeployment');
  if (epEl) epEl.value = getAgentEndpoint();
  if (keyEl) keyEl.value = getAgentApiKey();
  if (depEl) depEl.value = getAgentModel();
  openModal('aiConfigModal');
}
export function saveAIConfig() {
  const ep = byId('azureEndpoint')?.value?.trim();
  const key = byId('azureApiKey')?.value?.trim();
  const dep = byId('azureDeployment')?.value?.trim();
  if (ep) setAgentEndpoint(ep);
  if (key) setAgentApiKey(key);
  if (dep) setAgentModel(dep);

  if (state.azureGPT && typeof state.azureGPT.configure === 'function') {
    try { state.azureGPT.configure(ep || '', key || '', dep || ''); } catch (_) { /* ignore */ }
  }

  showToast('AI 配置已保存', 'success');
  const btn = byId('generateReportBtn');
  if (btn) btn.disabled = !(ep && key);
  closeModal('aiConfigModal');
}
export async function testAIConnection() {
  showToast('正在连通 Agent…', 'info');
  try {
    const r = await checkAgentHealth();
    showToast(`连接成功 ✓ (${r?.status || 'ok'})`, 'success');
  } catch (e) {
    showToast(`连接失败：${e.message}`, 'error');
  }
}

// ========== Prompt 编辑 ==========
function ensureGPT() {
  if (!state.azureGPT) {
    showToast('AzureGPTAnalyzer 尚未初始化', 'error');
    return null;
  }
  return state.azureGPT;
}

export function refreshPromptSelector() {
  const gpt = ensureGPT();
  if (!gpt) return;
  const sel = byId('promptSelector');
  if (!sel) return;
  sel.innerHTML = '<option value="">-- 选择模板 --</option>';
  for (const p of gpt.getAvailablePrompts()) {
    sel.appendChild(h('option', { value: p.id }, [`${p.name}`]));
  }
}

export function showPromptEditor() {
  refreshPromptSelector();
  openModal('promptEditorModal');
}

export function loadPromptTemplate() {
  const gpt = ensureGPT();
  if (!gpt) return;
  const sel = byId('promptSelector');
  const id = sel?.value;
  if (!id) {
    setVal('promptName', '');
    setVal('promptDescription', '');
    setVal('promptTemplate', '');
    return;
  }
  const p = gpt.customPrompts?.get(id);
  if (!p) return;
  setVal('promptName', p.name || '');
  setVal('promptDescription', p.description || '');
  setVal('promptTemplate', p.template || '');
}

export function createNewPrompt() {
  setVal('promptSelector', '');
  setVal('promptName', '');
  setVal('promptDescription', '');
  setVal('promptTemplate', '');
  byId('promptName')?.focus();
}

export function savePrompt() {
  const gpt = ensureGPT();
  if (!gpt) return;
  const name = byId('promptName')?.value?.trim();
  const desc = byId('promptDescription')?.value?.trim() || '';
  const tpl = byId('promptTemplate')?.value || '';
  if (!name || !tpl) {
    showToast('请填写名称与模板内容', 'warn');
    return;
  }
  const id = (byId('promptSelector')?.value) || `custom_${Date.now()}`;
  gpt.addCustomPrompt(id, name, tpl, desc);
  persistPrompts();
  refreshPromptSelector();
  byId('promptSelector') && (byId('promptSelector').value = id);
  showToast('Prompt 已保存', 'success');
}

export function deletePrompt() {
  const gpt = ensureGPT();
  if (!gpt) return;
  const id = byId('promptSelector')?.value;
  if (!id) {
    showToast('请先选择要删除的模板', 'warn');
    return;
  }
  if (id === 'basic_analysis' || id === 'detailed_medical') {
    showToast('内置模板不能删除', 'warn');
    return;
  }
  gpt.customPrompts?.delete(id);
  persistPrompts();
  refreshPromptSelector();
  setVal('promptName', '');
  setVal('promptDescription', '');
  setVal('promptTemplate', '');
  showToast('Prompt 已删除', 'success');
}

export function previewPrompt() {
  const gpt = ensureGPT();
  if (!gpt) return;
  const tpl = byId('promptTemplate')?.value || '';
  const data = {
    avgHeartRate: 75, avgRespiratoryRate: 16,
    heartRateRange: '68-82 bpm', heartRateVariability: '正常',
    heartRateData: '示例数据', respiratoryData: '示例数据',
    ragContext: '（预览模式）',
    measurementTime: new Date().toLocaleString('zh-CN'),
    duration: '60s', dataQuality: '良好',
    respiratoryPattern: '规律', patientInfo: '未提供',
  };
  let rendered = tpl;
  for (const [k, v] of Object.entries(data)) rendered = rendered.replaceAll(`{${k}}`, String(v));

  const tx = byId('promptPreview') || byId('analysisResult');
  if (tx) {
    tx.innerHTML = `<div class="modal-card"><h4>Prompt 预览</h4><pre class="prompt-preview">${escapeHtml(rendered)}</pre></div>`;
  } else {
    showToast('已渲染预览（无显示容器，将以 alert 方式弹出）', 'info');
    alert(rendered.slice(0, 4000));
  }
}

function persistPrompts() {
  const gpt = ensureGPT();
  if (!gpt) return;
  const arr = [];
  for (const [id, p] of gpt.customPrompts.entries()) {
    if (id === 'basic_analysis' || id === 'detailed_medical') continue;
    arr.push({ id, ...p });
  }
  setItem(STORAGE_KEYS.CUSTOM_PROMPTS, JSON.stringify(arr));
}

export function loadPersistedPrompts() {
  const gpt = ensureGPT();
  if (!gpt) return;
  try {
    const raw = getItem(STORAGE_KEYS.CUSTOM_PROMPTS);
    if (!raw) return;
    const arr = JSON.parse(raw);
    for (const item of arr) {
      gpt.addCustomPrompt(item.id, item.name, item.template, item.description);
    }
  } catch (_) { /* ignore */ }
}

// ========== RAG 编辑 ==========
export function showRAGEditor() {
  renderRAGList();
  openModal('ragEditorModal');
}
function renderRAGList() {
  const gpt = ensureGPT();
  const list = byId('ragList');
  if (!gpt || !list) return;
  list.innerHTML = '';
  for (const [id, entry] of gpt.ragDatabase.entries()) {
    const card = h('div', { class: 'rag-entry-card' }, [
      h('div', { class: 'rag-entry-head' }, [
        h('div', { class: 'rag-entry-id mono' }, [id]),
        h('div', null, [
          h('button', { class: 'btn btn-sm', 'data-rag-edit': id }, ['编辑']),
          h('button', { class: 'btn btn-sm btn-secondary', 'data-rag-del': id }, ['删除']),
        ]),
      ]),
      h('div', { class: 'rag-entry-keywords' }, [
        h('span', { class: 'badge' }, [`关键词：${(entry.keywords || []).join(', ')}`]),
      ]),
      h('pre', { class: 'rag-entry-content' }, [String(entry.content).slice(0, 280) + ((entry.content.length > 280) ? '…' : '')]),
    ]);
    list.appendChild(card);
  }
  list.querySelectorAll('[data-rag-edit]').forEach(b => {
    b.addEventListener('click', () => editRAGEntry(b.getAttribute('data-rag-edit')));
  });
  list.querySelectorAll('[data-rag-del]').forEach(b => {
    b.addEventListener('click', () => deleteRAGEntry(b.getAttribute('data-rag-del')));
  });
}

export function addRAGEntry() {
  setVal('ragEntryId', '');
  setVal('ragKeywords', '');
  setVal('ragContent', '');
  byId('ragEditor').style.display = 'block';
  byId('ragEntryId')?.focus();
}

function editRAGEntry(id) {
  const gpt = ensureGPT();
  if (!gpt) return;
  const e = gpt.ragDatabase.get(id);
  if (!e) return;
  setVal('ragEntryId', id);
  setVal('ragKeywords', (e.keywords || []).join(','));
  setVal('ragContent', e.content || '');
  byId('ragEditor').style.display = 'block';
}

export function saveRAGEntry() {
  const gpt = ensureGPT();
  if (!gpt) return;
  const id = byId('ragEntryId')?.value?.trim();
  const kw = byId('ragKeywords')?.value?.trim();
  const ct = byId('ragContent')?.value || '';
  if (!id || !ct) {
    showToast('请填写条目 ID 和知识内容', 'warn');
    return;
  }
  const keywords = kw ? kw.split(',').map(s => s.trim()).filter(Boolean) : [];
  gpt.addRAGEntry(id, ct, keywords);
  persistRAG();
  renderRAGList();
  cancelRAGEdit();
  showToast('知识条目已保存', 'success');
}

function deleteRAGEntry(id) {
  const gpt = ensureGPT();
  if (!gpt) return;
  if (!confirm(`确定删除知识条目 "${id}" 吗？`)) return;
  gpt.ragDatabase.delete(id);
  persistRAG();
  renderRAGList();
}

export function cancelRAGEdit() {
  const e = byId('ragEditor');
  if (e) e.style.display = 'none';
}

export function importRAGData() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'application/json,.json';
  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result);
        const gpt = ensureGPT();
        if (!gpt) return;
        gpt.importRAGDatabase(data);
        persistRAG();
        renderRAGList();
        showToast('知识库导入完成', 'success');
      } catch (e) {
        showToast(`导入失败：${e.message}`, 'error');
      }
    };
    reader.readAsText(file);
  });
  input.click();
}

export function exportRAGData() {
  const gpt = ensureGPT();
  if (!gpt) return;
  const data = gpt.exportRAGDatabase();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `rag_database_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
  showToast('知识库已导出', 'success');
}

function persistRAG() {
  const gpt = ensureGPT();
  if (!gpt) return;
  setItem(STORAGE_KEYS.RAG_DATABASE, JSON.stringify(gpt.exportRAGDatabase()));
}
export function loadPersistedRAG() {
  const gpt = ensureGPT();
  if (!gpt) return;
  try {
    const raw = getItem(STORAGE_KEYS.RAG_DATABASE);
    if (!raw) return;
    gpt.importRAGDatabase(JSON.parse(raw));
  } catch (_) { /* ignore */ }
}

// ========== 诊断报告（基于 Azure GPT）==========
export async function generateDiagnosticReport() {
  const gpt = ensureGPT();
  if (!gpt) return;
  if (!gpt.isConfigured) {
    showAIConfig();
    showToast('请先配置 Azure OpenAI', 'warn');
    return;
  }
  const target = byId('diagnosticReport') || byId('analysisResult');
  if (target) {
    target.innerHTML = `<div class="analysis-loading"><span class="spinner"></span><span>正在生成诊断报告…</span></div>`;
  }
  try {
    const report = await gpt.generateDiagnosticReport(state.processedResults || []);
    state.lastDiagnosticReport = report;
    if (target) target.innerHTML = `<div class="analysis-report animate-fade-up">${formatChatMessage(report)}</div>`;
    showToast('诊断报告已生成', 'success');
  } catch (e) {
    if (target) target.innerHTML = `<div class="text-muted">❌ 生成失败：${e.message}</div>`;
    showToast(`生成失败：${e.message}`, 'error');
  }
}

export function exportReport() {
  const text = state.lastDiagnosticReport
    || byId('diagnosticReport')?.innerText
    || byId('analysisResult')?.innerText;
  if (!text || !text.trim()) {
    showToast('暂无可导出的报告', 'warn');
    return;
  }
  const md = `# 宠物健康诊断报告\n\n生成时间：${new Date().toLocaleString('zh-CN')}\n\n${text}`;
  const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `report_${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
  showToast('报告已导出', 'success');
}

export function exportHealthReport() {
  return exportReport();
}

// ========== Helpers ==========
function setVal(id, v) { const el = byId(id); if (el) el.value = v; }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
