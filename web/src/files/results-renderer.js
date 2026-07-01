/**
 * 处理结果渲染：统计卡片、表格、JSON 信息卡、CSV / PNG 导出。
 */

import { byId, h } from '../utils/dom.js';
import state from '../app/state.js';
import { showToast } from '../ui/toast.js';

export function renderResults(results) {
  renderStats(results);
  renderTable(results);
}

function renderStats(results) {
  const grid = byId('statisticsGrid');
  if (!grid) return;
  const ok = results.filter(r => r.status === 'success');
  const err = results.length - ok.length;
  const avg = (key) => {
    const arr = ok.map(r => r[key]).filter(Number.isFinite);
    return arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : '--';
  };

  grid.innerHTML = '';
  const cards = [
    ['处理文件', results.length],
    ['平均心率', `${avg('heartRate')} bpm`],
    ['平均呼吸', `${avg('respiratoryRate')} bpm`],
    ['失败文件', err],
  ];
  for (const [label, value] of cards) {
    grid.appendChild(h('div', { class: 'stat-card' }, [
      h('div', { class: 'label' }, [label]),
      h('div', { class: 'value' }, [String(value)]),
    ]));
  }
}

function renderTable(results) {
  const wrap = byId('resultsTableWrap');
  if (!wrap) return;
  wrap.innerHTML = '';
  const table = h('table', null, []);
  const head = h('thead', null, [h('tr', null, [
    h('th', null, ['文件']), h('th', null, ['类型']), h('th', null, ['心率']),
    h('th', null, ['呼吸']), h('th', null, ['数据点']), h('th', null, ['状态']),
  ])]);
  const body = h('tbody', null, []);
  for (const r of results) {
    body.appendChild(h('tr', null, [
      h('td', { class: 'mono' }, [r.fileName]),
      h('td', null, [r.dataType === 'json' ? 'JSON' : 'TXT']),
      h('td', null, [r.heartRate != null ? `${r.heartRate} bpm` : '--']),
      h('td', null, [r.respiratoryRate != null ? `${r.respiratoryRate} bpm` : '--']),
      h('td', null, [String(r.dataPoints || (r.heartRateTimeSeries?.length || 0))]),
      h('td', null, [r.status === 'success' ? '✓' : `✗ ${r.error || ''}`]),
    ]));
  }
  table.append(head, body);
  wrap.appendChild(table);
}

export function exportResultsCSV() {
  const r = state.processedResults || [];
  if (r.length === 0) {
    showToast('暂无可导出的结果', 'warn');
    return;
  }
  const header = ['file', 'type', 'heart_rate', 'resp_rate', 'data_points', 'status', 'error'];
  const rows = r.map(it => [
    quote(it.fileName), it.dataType === 'json' ? 'json' : 'txt',
    it.heartRate ?? '', it.respiratoryRate ?? '',
    it.dataPoints || (it.heartRateTimeSeries?.length || 0),
    it.status, quote(it.error || ''),
  ]);
  const csv = [header.join(','), ...rows.map(row => row.join(','))].join('\n');
  download(csv, `analysis_${Date.now()}.csv`, 'text/csv;charset=utf-8');
}

export function exportChartsPNG() {
  const charts = Object.values(state.charts || {});
  if (charts.length === 0) {
    showToast('请先处理文件以生成图表', 'warn');
    return;
  }
  for (const c of charts) {
    if (c?.canvas) {
      const a = document.createElement('a');
      a.href = c.canvas.toDataURL('image/png');
      a.download = `${c.options?.plugins?.title?.text || 'chart'}.png`;
      a.click();
    }
  }
}

function quote(s) { const v = String(s).replace(/"/g, '""'); return /[",\n]/.test(v) ? `"${v}"` : v; }
function download(content, filename, type) {
  const blob = new Blob([content], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
