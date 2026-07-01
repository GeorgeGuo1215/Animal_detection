/**
 * Workspace 路由：在多个 workspace-page 之间切换。
 *
 * 与 hash 同步：
 *   #ws-integration / #ws-chat / #ws-attitude / #ws-resting / #ws-activity /
 *   #ws-sleep / #ws-files / #ws-analytics / #ws-ble-charts
 *
 * 同时兼容旧锚点（sec-integration / healthChatSection 等）。
 */

import state from '../app/state.js';
import { bus } from '../app/event-bus.js';

const TITLES = {
  integration: 'n8n 模拟入站',
  chat: '宠物健康对话',
  attitude: 'IMU 姿态解算',
  resting: '静息心率 / 呼吸监测',
  activity: '活动量与步数',
  sleep: '睡眠质量监测',
  files: '数据文件上传',
  analytics: '处理状态与结果分析',
  'ble-charts': '蓝牙实时图表',
};

function setNavActive(kind, id) {
  document.querySelectorAll('.page-nav--demo .nav-pill').forEach((el) => {
    const matchWs = id && el.getAttribute('data-workspace') === id;
    const matchBle = kind === 'ble' && el.getAttribute('data-nav') === 'ble';
    el.classList.toggle('is-active', !!(matchWs || matchBle));
  });
}

export function showWorkspacePage(name, opts = {}) {
  const empty = document.getElementById('ws-empty');
  const pages = document.querySelectorAll('#workspacePages .workspace-page');
  pages.forEach((p) => p.classList.remove('is-active'));
  const label = document.getElementById('workspaceCurrentLabel');

  if (name === 'ble' || name === 'panel-ble') {
    const bt = document.getElementById('bluetoothChartsSection');
    if (bt) bt.style.display = 'none';
    const top = document.getElementById('panel-ble');
    if (top) top.scrollIntoView({ behavior: opts.instant ? 'auto' : 'smooth', block: 'start' });
    if (empty) empty.classList.add('is-active');
    if (label) label.textContent = '请选择导航中的功能模块';
    setNavActive('ble', null);
    bus.emit('workspace:change', { name: 'ble' });
    return;
  }
  if (name !== 'ble-charts') {
    const bt = document.getElementById('bluetoothChartsSection');
    if (bt) bt.style.display = 'none';
  }

  const target = document.getElementById('ws-' + name);
  if (!target) {
    if (empty) empty.classList.add('is-active');
    if (label) label.textContent = '模块未找到';
    return;
  }
  target.classList.add('is-active');
  if (label) label.textContent = TITLES[name] || name;
  setNavActive(null, name);

  const ws = document.querySelector('.demo-workspace');
  if (ws) ws.scrollIntoView({ behavior: opts.instant ? 'auto' : 'smooth', block: 'start' });

  if (name === 'ble-charts') {
    const section = document.getElementById('bluetoothChartsSection');
    if (section) section.style.display = 'block';
    setTimeout(() => {
      try {
        Object.values(state.bleCharts || {}).forEach((ch) => {
          if (ch && typeof ch.resize === 'function') ch.resize();
          if (ch && typeof ch.update === 'function') ch.update('none');
        });
      } catch (_) {}
    }, 80);
  }
  bus.emit('workspace:change', { name });
}

function routeFromHash() {
  const raw = (location.hash || '').replace(/^#/, '');
  if (!raw) return;
  const legacy = {
    'sec-integration': 'integration',
    'sec-attitude':    'attitude',
    'sec-files':       'files',
    'sec-resting':     'resting',
    'sec-activity':    'activity',
    'sec-sleep':       'sleep',
    healthChatSection: 'chat',
    'panel-ble':       'ble',
  };
  if (legacy[raw]) {
    showWorkspacePage(legacy[raw], { instant: true });
    return;
  }
  if (raw.startsWith('ws-')) {
    showWorkspacePage(raw.slice(3), { instant: true });
  }
}

function bindNav() {
  document.querySelectorAll('.page-nav--demo a[data-workspace]').forEach((a) => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      const w = a.getAttribute('data-workspace');
      showWorkspacePage(w);
      history.replaceState(null, '', '#ws-' + w);
    });
  });
  document.querySelectorAll('.page-nav--demo a[data-nav="ble"]').forEach((a) => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      showWorkspacePage('ble');
      history.replaceState(null, '', '#panel-ble');
    });
  });
}

export function initWorkspaceRouter() {
  bindNav();
  routeFromHash();
  window.addEventListener('hashchange', routeFromHash);
}

if (typeof window !== 'undefined') {
  window.showWorkspacePage = showWorkspacePage;
}
