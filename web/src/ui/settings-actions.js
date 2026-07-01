/**
 * 通用设置 / 文件清理 / 图表与结果导出（小工具集合）
 */

import state from '../app/state.js';
import { byId } from '../utils/dom.js';
import { showToast } from './toast.js';
import { SAMPLING_RATE, VITAL } from '../app/config.js';
import { exportResultsCSV, exportChartsPNG } from '../files/results-renderer.js';

export function applySettings() {
  const sr = parseInt(byId('samplingRate')?.value || SAMPLING_RATE, 10) || SAMPLING_RATE;
  if (state.processor) state.processor.fs = sr;
  const smooth = parseInt(byId('heartRateSmoothing')?.value || VITAL.HEART_RATE_SMOOTHING, 10);
  if (Number.isFinite(smooth) && smooth >= 5 && smooth <= 60) state.historyMaxLength = smooth;
  const delta = parseInt(byId('heartRateDelta')?.value || VITAL.HEART_DELTA_BPM, 10);
  if (Number.isFinite(delta) && delta >= 5 && delta <= 30) state.heartRateDelta = delta;

  showToast(`已应用设置：${sr}Hz / 平滑${state.historyMaxLength || VITAL.HEART_RATE_SMOOTHING}次 / 阈值${state.heartRateDelta || VITAL.HEART_DELTA_BPM}bpm`, 'success');
  const details = byId('bleAlgoDetails');
  if (details) details.open = false;
}

export function clearFiles() {
  state.selectedFiles = [];
  state.processedResults = [];
  const list = byId('fileListUl');
  if (list) list.innerHTML = '';
  const wrap = byId('fileList');
  if (wrap) wrap.style.display = 'none';
  const grid = byId('statisticsGrid');
  if (grid) grid.innerHTML = '';
  const tbl = byId('resultsTableWrap');
  if (tbl) tbl.innerHTML = '';
  showToast('已清空文件与结果', 'info');
}

export function exportResults() { exportResultsCSV(); }
export function exportCharts() { exportChartsPNG(); }

export function toggleSettings() {
  const d = byId('bleAlgoDetails');
  if (d) { d.open = !d.open; return; }
  const p = byId('settingsPanel');
  if (p) p.classList.toggle('open');
}
