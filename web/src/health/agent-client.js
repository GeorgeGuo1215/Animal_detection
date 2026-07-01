/**
 * Agent API 客户端：/health, /v1/chat/completions, /agent/plan_and_solve
 *
 * 修复点：
 *   - 不再使用 process.env.OPENAI_API_KEY（浏览器无 process）
 *   - 统一通过 localStorage 读取 endpoint / apiKey / model
 */

import { byId, getStr } from '../utils/dom.js';
import { getItem, setItem } from '../utils/storage.js';
import { STORAGE_KEYS, URLS, DEFAULTS } from '../app/config.js';

export function getAgentEndpoint() {
  const v = getStr('agentEndpoint').trim() || getItem(STORAGE_KEYS.AGENT_ENDPOINT) || URLS.AGENT_DEFAULT;
  return v.replace(/\/+$/, '');
}

export function setAgentEndpoint(url) {
  if (url) setItem(STORAGE_KEYS.AGENT_ENDPOINT, url);
}

export function getAgentApiKey() {
  return getItem(STORAGE_KEYS.AGENT_API_KEY) || DEFAULTS.AGENT_API_KEY;
}

export function setAgentApiKey(key) {
  setItem(STORAGE_KEYS.AGENT_API_KEY, key || '');
}

export function getAgentModel() {
  return getItem(STORAGE_KEYS.AGENT_MODEL) || DEFAULTS.AGENT_MODEL;
}

export function setAgentModel(model) {
  const valid = ['agent-plan-solve', 'agent-multi-turn'];
  if (valid.includes(model)) setItem(STORAGE_KEYS.AGENT_MODEL, model);
}

export function getChatAnimalId() {
  // 页面无独立 chatAnimalId 输入时，沿用「蓝牙与上报」里的 bleAnimalId（index.html 仅有后者）
  return getStr('chatAnimalId').trim()
    || getStr('bleAnimalId').trim()
    || getItem(STORAGE_KEYS.CHAT_ANIMAL_ID)
    || 'demo_001';
}

export function setChatAnimalId(id) {
  if (id) setItem(STORAGE_KEYS.CHAT_ANIMAL_ID, id);
}

export async function fetchWithTimeout(url, opts = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...opts, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export async function checkAgentHealth() {
  const origin = getAgentEndpoint();
  const r = await fetchWithTimeout(`${origin}/health`, { method: 'GET' }, 6000);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json().catch(() => ({}));
}

/** OpenAI 兼容流式 chat completions（SSE），调用方以 callback 接收 chunk。 */
export async function streamChatCompletions(messages, callback) {
  const origin = getAgentEndpoint();
  const apiKey = getAgentApiKey();
  const animal = getChatAnimalId();
  const model = getAgentModel();

  const resp = await fetch(`${origin}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      'X-Animal-Id': animal,
    },
    body: JSON.stringify({
      model, messages, stream: true,
      temperature: 0.7, max_tokens: 1500, animal_id: animal,
    }),
  });
  if (!resp.ok) {
    const errData = await resp.json().catch(() => ({}));
    throw new Error(errData.error?.message || `HTTP ${resp.status}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') continue;
      try { callback(JSON.parse(data)); } catch {}
    }
  }
}

/** 非流式 plan_and_solve 调用，用于 JSON 数据健康分析 */
export async function planAndSolve(promptOrPayload) {
  const origin = getAgentEndpoint();
  const apiKey = getAgentApiKey();
  const animal = getChatAnimalId();
  const body = typeof promptOrPayload === 'string'
    ? { question: promptOrPayload, animal_id: animal }
    : { ...promptOrPayload, animal_id: promptOrPayload?.animal_id || animal };
  const r = await fetch(`${origin}/agent/plan_and_solve`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
