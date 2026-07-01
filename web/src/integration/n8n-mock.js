/**
 * n8n 模拟入站（POST /integration/ingest）
 *
 * 提供：
 *   integrationMockInit()       初始化输入框默认值（通过 legacy bridge 暴露）
 *   integrationMockLoadFixture()加载 fixtures/n8n_mock_event.json 模板
 *   integrationMockSend()       发送当前编辑器内容
 */

import { byId, setText } from '../utils/dom.js';
import { getItem, setItem } from '../utils/storage.js';
import { STORAGE_KEYS } from '../app/config.js';

const DEFAULT_BASE = 'http://127.0.0.1:8000/integration';
const FIXTURE_PATH = 'fixtures/n8n_mock_event.json';

function getBase() {
  const el = byId('integrationIngestBase');
  const v = (el && el.value && el.value.trim()) || getItem(STORAGE_KEYS.INTEGRATION_BASE) || DEFAULT_BASE;
  return v.replace(/\/+$/, '');
}

function ingestUrl() { return `${getBase()}/ingest`; }

function setStatus(msg, isError) {
  const el = byId('integrationMockStatus');
  if (!el) return;
  el.textContent = msg;
  el.classList.toggle('text-faint', !isError);
  el.style.color = isError ? '' : '';
}

function setResult(obj) {
  const el = byId('integrationMockResult');
  if (!el) return;
  el.textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
}

async function loadFixture() {
  const url = new URL(FIXTURE_PATH, window.location.href).href;
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`加载 fixture 失败: HTTP ${r.status}`);
  return r.json();
}

export function integrationMockInit() {
  const baseEl = byId('integrationIngestBase');
  if (baseEl && !baseEl.value) {
    baseEl.value = getItem(STORAGE_KEYS.INTEGRATION_BASE) || DEFAULT_BASE;
  }
  if (baseEl) {
    baseEl.addEventListener('change', () => {
      setItem(STORAGE_KEYS.INTEGRATION_BASE, baseEl.value.trim() || DEFAULT_BASE);
    });
  }
}

export async function integrationMockLoadFixture() {
  setStatus('加载中…');
  try {
    const data = await loadFixture();
    const ta = byId('integrationMockPayload');
    if (ta) ta.value = JSON.stringify(data, null, 2);
    setStatus('已加载模板（可编辑 event_id 后发送）');
    setResult('');
  } catch (e) {
    setStatus(String(e.message || e), true);
  }
}

export async function integrationMockSend() {
  const ta = byId('integrationMockPayload');
  let body;
  try {
    body = JSON.parse((ta && ta.value) || '{}');
  } catch (e) {
    setStatus('JSON 解析失败：' + e.message, true);
    return;
  }
  if (body.event_id) body.event_id = `${body.event_id}_${Date.now()}`;
  setStatus('发送中…');
  setResult('');
  try {
    const resp = await fetch(ingestUrl(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const text = await resp.text();
    let parsed;
    try { parsed = JSON.parse(text); } catch { parsed = text; }
    if (!resp.ok) {
      setStatus(`HTTP ${resp.status}`, true);
      setResult(parsed);
      return;
    }
    setStatus('成功');
    setResult(parsed);
  } catch (e) {
    setStatus('请求失败：' + (e.message || e), true);
    setResult(String(e));
  }
}
