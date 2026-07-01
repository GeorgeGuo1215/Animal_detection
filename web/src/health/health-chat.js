/**
 * 健康对话 UI：渲染消息、状态区折叠、流式 SSE 接入。
 *
 * 修复：
 *   - 移除 process.env 引用，全部通过 agent-client 读取
 *   - SSE 状态从 generating → streaming 切换时移除占位条目
 *   - 折叠按钮交互通过事件委托完成（不再依赖 onclick=app.xxx）
 */

import { byId, h } from '../utils/dom.js';
import {
  checkAgentHealth, streamChatCompletions, getAgentEndpoint,
} from './agent-client.js';
import { formatChatMessage, formatAgentStatus, stripAnswerPlaceholders } from './chat-formatter.js';
import { saveChatMessage, recentMessages, clearChatHistory, getChatHistory } from './chat-history.js';
import state from '../app/state.js';
import { showToast } from '../ui/toast.js';

/** 事件绑定只做一次；探活每次进入 / 点「重新连接」都会执行 */
let chatUiBound = false;

function bindHealthChatUiOnce() {
  if (chatUiBound) return;
  chatUiBound = true;

  const messages = byId('chatMessages');
  if (messages) {
    messages.addEventListener('click', (e) => {
      const target = e.target.closest('.message-status-area');
      if (target) target.classList.toggle('is-collapsed');
    });
  }

  const input = byId('chatInput');
  if (input) {
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        void sendChatMessage();
      }
    });
  }

  /**
   * 发送按钮不要用内联 onclick：部分环境下全局 sendChatMessage 未挂到 window 时，
   * 点击无任何请求；此处直接用模块内函数并 void 处理 async。
   */
  const sendBtn = byId('sendChatBtn');
  if (sendBtn) {
    sendBtn.type = 'button';
    sendBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      void sendChatMessage();
    });
  }
}

/**
 * 与 index.html 一致：#chatAgentStatusDot / #chatAgentStatusText / #chatAgentEndpointText
 * （旧代码误用 chatStatusIndicator 等 id，导致整段早退，界面永远停留在 HTML 默认「未连接」）
 */
export function setHealthChatStatus(status) {
  const dot = byId('chatAgentStatusDot');
  const text = byId('chatAgentStatusText');
  const ep = byId('chatAgentEndpointText');

  if (dot) {
    dot.classList.remove('disconnected', 'is-active', 'is-pulse', 'is-idle');
    if (status === 'connecting') {
      dot.classList.add('is-pulse');
    } else if (status === 'connected') {
      dot.classList.add('is-active');
    } else {
      dot.classList.add('is-idle', 'disconnected');
    }
  }

  let label = '未知';
  if (status === 'connecting') label = '连接中…';
  else if (status === 'connected') label = '已连接';
  else if (status === 'disconnected') label = '未连接';
  if (text) text.textContent = label;

  const origin = getAgentEndpoint();
  if (ep) {
    ep.textContent = origin;
    ep.style.display = origin ? '' : 'none';
  }
}

export function initializeHealthChat() {
  bindHealthChatUiOnce();
  setHealthChatStatus('connecting');

  const sendBtn = byId('sendChatBtn');
  if (sendBtn) {
    sendBtn.disabled = true;
    sendBtn.textContent = '连接中…';
  }

  checkAgentHealth()
    .then(() => {
      setHealthChatStatus('connected');
      if (sendBtn) {
        sendBtn.disabled = false;
        sendBtn.textContent = '发送';
      }
      loadChatHistory();
      showToast('宠物健康对话已就绪', 'success');
    })
    .catch((e) => {
      setHealthChatStatus('disconnected');
      if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.textContent = '发送';
      }
      showToast(`连接 Agent 失败：${e.message}`, 'warn');
    });
}

function buildSystemContext() {
  const jsonResults = state.processedResults?.filter(r => r?.dataType === 'json') || [];
  let context = '您是专业的宠物健康助手，可以解答关于宠物健康、护理、训练等方面的问题。';
  if (jsonResults.length > 0) {
    const r = jsonResults[0];
    const animal = r.animal || {};
    context += `

当前宠物信息:
- 类型: ${animal.species === 'dog' ? '狗狗' : animal.species === 'cat' ? '猫咪' : '其他'}
- 姓名: ${animal.name || '未命名'}
- 品种: ${animal.breed || '未知'}
- 年龄: ${animal.age_months ? Math.floor(animal.age_months / 12) + '岁' + (animal.age_months % 12) + '个月' : '未知'}
- 体重: ${animal.weight_kg || '未知'}kg
- 性别: ${animal.sex === 'male' ? '公' : animal.sex === 'female' ? '母' : '未知'}

最近的生理指标:
- 平均心率: ${r.heartRate ?? '--'} bpm
- 平均呼吸频率: ${r.respiratoryRate ?? '--'} bpm
- 体温: ${r.temperature ?? '--'}°C

请基于这些信息提供专业的建议。`;
  }
  return context;
}

function appendUserMessage(content) {
  const messagesEl = byId('chatMessages');
  if (!messagesEl) return;
  const node = h('div', { class: 'chat-message role-user' }, [
    h('div', { class: 'chat-message-meta' }, [new Date().toLocaleTimeString('zh-CN')]),
    h('div', { class: 'chat-message-bubble' }, [content]),
  ]);
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function createAssistantMessage() {
  const messagesEl = byId('chatMessages');
  if (!messagesEl) return null;
  const id = `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  const wrapper = h('div', { class: 'chat-message role-assistant', id }, []);

  const meta = h('div', { class: 'chat-message-meta' }, [new Date().toLocaleTimeString('zh-CN')]);
  const status = h('div', { class: 'message-status-area', id: `${id}_status` }, [
    h('div', { class: 'message-status-row is-summary' }, [
      h('span', { class: 'message-status-icon' }, ['⚙️']),
      h('span', null, ['Agent 思考过程（点击折叠/展开）']),
    ]),
  ]);
  const bubble = h('div', { class: 'chat-message-bubble', id: `${id}_answer` }, ['']);
  wrapper.append(meta, status, bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return id;
}

function updateStatusArea(id, statusLogs) {
  const el = byId(`${id}_status`);
  if (!el) return;
  el.innerHTML = '';
  el.appendChild(h('div', { class: 'message-status-row is-summary' }, [
    h('span', { class: 'message-status-icon' }, ['⚙️']),
    h('span', null, ['Agent 思考过程（点击折叠/展开）']),
  ]));
  for (const it of statusLogs) {
    el.appendChild(h('div', { class: 'message-status-row' }, [
      h('span', { class: 'message-status-icon' }, [it.icon]),
      h('span', null, [it.text]),
    ]));
  }
}

function updateAnswerArea(id, content) {
  const el = byId(`${id}_answer`);
  if (!el) return;
  const cleaned = stripAnswerPlaceholders(content);
  el.innerHTML = formatChatMessage(cleaned);
  const messagesEl = byId('chatMessages');
  if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function collapseStatusArea(id) {
  const el = byId(`${id}_status`);
  if (el) el.classList.add('is-collapsed');
}

export async function sendChatMessage() {
  const sendBtn = byId('sendChatBtn');
  if (sendBtn?.disabled) {
    showToast('尚未连接到 Agent，请先确认本机服务已启动，并点击「重新连接」', 'warn');
    return;
  }

  const input = byId('chatInput');
  if (!input) return;
  const message = input.value.trim();
  if (!message) {
    showToast('请输入问题内容', 'warn');
    return;
  }
  appendUserMessage(message);
  input.value = '';
  if (sendBtn) { sendBtn.disabled = true; sendBtn.textContent = '发送中…'; }

  const id = createAssistantMessage();
  let statusLogs = [];
  let fullContent = '';
  let isGenerating = false;
  let sawStreamToken = false;
  // 流分两个阶段：'tool'（计划/工具调用）→ 'answer'（最终回答流式）。
  // 工具阶段里后端会用「无 agent_status 的 content 块」下发子细节（如 "   Query: ..."、
  // "   Table: ..."、联网回退说明），过去这些行被直接丢弃，导致工具调用进度展示不完整。
  let phase = 'tool';

  try {
    const messages = [
      { role: 'system', content: buildSystemContext() },
      ...recentMessages(12).filter(m => m && (m.role === 'user' || m.role === 'assistant') && m.content)
        .map(m => ({ role: m.role, content: m.content })),
      { role: 'user', content: message },
    ];

    await streamChatCompletions(messages, (chunk) => {
      const delta = chunk.choices?.[0]?.delta;
      const content = delta?.content;
      const agentStatus = chunk.agent_status;
      const agentDetail = chunk.agent_detail ?? chunk.agentDetail ?? {};

      if (agentStatus) {
        if (agentStatus === 'generating' || agentStatus === 'streaming') isGenerating = true;
        if (agentStatus === 'streaming' && !sawStreamToken) {
          sawStreamToken = true;
          phase = 'answer';
          statusLogs = statusLogs.filter(l => l.status !== 'generating');
          updateStatusArea(id, statusLogs);
        }
        if (agentStatus !== 'streaming') {
          const info = formatAgentStatus(agentStatus, agentDetail);
          if (info) {
            statusLogs.push(info);
            updateStatusArea(id, statusLogs);
          }
        }
      }
      if (content) {
        if (agentStatus === 'streaming' || (isGenerating && agentStatus == null && phase === 'answer')) {
          // 最终回答正文
          fullContent += content;
          updateAnswerArea(id, fullContent);
        } else if (agentStatus == null && phase === 'tool') {
          // 工具阶段的子细节行（无状态的 content 块），并入工具进度展示，避免被丢弃
          const text = String(content).trim();
          if (text) {
            statusLogs.push({ icon: '·', text, status: 'tool_detail' });
            updateStatusArea(id, statusLogs);
          }
        }
      }
    });
    collapseStatusArea(id);
    saveChatMessage('user', message);
    saveChatMessage('assistant', fullContent || '暂无回复');
  } catch (e) {
    updateAnswerArea(id, `❌ 抱歉，回复失败：${e.message}`);
  } finally {
    if (sendBtn) { sendBtn.disabled = false; sendBtn.textContent = '发送'; }
  }
}

export function loadChatHistory() {
  const messagesEl = byId('chatMessages');
  if (!messagesEl) return;
  messagesEl.innerHTML = '';
  const history = getChatHistory();
  for (const msg of history) {
    if (!msg || !(msg.role === 'user' || msg.role === 'assistant') || !msg.content) continue;
    const role = msg.role === 'user' ? 'role-user' : 'role-assistant';
    const node = h('div', { class: `chat-message ${role}` }, [
      h('div', { class: 'chat-message-meta' }, [new Date(msg.timestamp).toLocaleTimeString('zh-CN')]),
      h('div', { class: 'chat-message-bubble' }, []),
    ]);
    node.querySelector('.chat-message-bubble').innerHTML = formatChatMessage(msg.content);
    messagesEl.appendChild(node);
  }
}

export function clearChat() {
  clearChatHistory();
  loadChatHistory();
  showToast('对话历史已清空', 'info');
}
