/**
 * Chart.js 通用工厂：创建黑白风格的折线 / 柱状 / 散点 / 环形图，并安全销毁。
 */

import { SERIES_COLORS, COLORS } from './chart-theme.js';

export function destroyChart(chart) {
  try { chart && chart.destroy && chart.destroy(); } catch {}
}

export function lineChart(canvasId, opts = {}) {
  const Chart = window.Chart;
  const el = document.getElementById(canvasId);
  if (!el || !Chart) return null;
  return new Chart(el, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: mergeOptions({
      animation: false,
      plugins: { title: { display: !!opts.title, text: opts.title || '' } },
      scales: {
        x: { display: true, title: { display: !!opts.xLabel, text: opts.xLabel || '' } },
        y: { display: true, title: { display: !!opts.yLabel, text: opts.yLabel || '' } },
      },
    }, opts.options),
  });
}

export function barChart(canvasId, opts = {}) {
  const Chart = window.Chart;
  const el = document.getElementById(canvasId);
  if (!el || !Chart) return null;
  return new Chart(el, {
    type: 'bar',
    data: { labels: [], datasets: [] },
    options: mergeOptions({
      animation: false,
      plugins: { title: { display: !!opts.title, text: opts.title || '' } },
      scales: {
        x: { title: { display: !!opts.xLabel, text: opts.xLabel || '' } },
        y: { beginAtZero: true, title: { display: !!opts.yLabel, text: opts.yLabel || '' } },
      },
    }, opts.options),
  });
}

export function scatterChart(canvasId, opts = {}) {
  const Chart = window.Chart;
  const el = document.getElementById(canvasId);
  if (!el || !Chart) return null;
  return new Chart(el, {
    type: 'scatter',
    data: { datasets: [] },
    options: mergeOptions({
      animation: false,
      plugins: { title: { display: !!opts.title, text: opts.title || '' } },
      scales: {
        x: { title: { display: !!opts.xLabel, text: opts.xLabel || '' } },
        y: { title: { display: !!opts.yLabel, text: opts.yLabel || '' } },
      },
    }, opts.options),
  });
}

export function makeLineDataset(label, data, idx = 0) {
  const color = SERIES_COLORS[idx % SERIES_COLORS.length];
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: hexToAlpha(color, 0.08),
    tension: 0.18,
    pointRadius: 0,
    borderWidth: 1.4,
  };
}

function hexToAlpha(hex, a) {
  const m = hex.replace('#', '');
  const r = parseInt(m.length === 3 ? m[0] + m[0] : m.slice(0, 2), 16);
  const g = parseInt(m.length === 3 ? m[1] + m[1] : m.slice(2, 4), 16);
  const b = parseInt(m.length === 3 ? m[2] + m[2] : m.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

function mergeOptions(base, extra) {
  if (!extra) return base;
  const out = { ...base, ...extra };
  out.plugins = { ...(base.plugins || {}), ...(extra.plugins || {}) };
  out.scales = { ...(base.scales || {}), ...(extra.scales || {}) };
  return out;
}

export { COLORS };
