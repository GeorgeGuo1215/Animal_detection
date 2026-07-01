/**
 * 文件选择 / 拖拽 / TXT|JSON 处理
 */

import state from '../app/state.js';
import { byId, h } from '../utils/dom.js';
import { showToast } from '../ui/toast.js';
import { initFileCharts, updateFileCharts } from '../charts/file-charts.js';
import { renderResults } from './results-renderer.js';
import { child } from '../utils/logger.js';

const log = child('files');

export function bindFileHandlers() {
  const input = byId('fileInput');
  const dropArea = byId('uploadArea');
  if (input) {
    input.addEventListener('change', (e) => onFiles(Array.from(e.target.files || [])));
  }
  if (dropArea) {
    dropArea.addEventListener('dragover', (e) => { e.preventDefault(); dropArea.classList.add('is-drag'); });
    dropArea.addEventListener('dragleave', () => dropArea.classList.remove('is-drag'));
    dropArea.addEventListener('drop', (e) => {
      e.preventDefault();
      dropArea.classList.remove('is-drag');
      const files = Array.from(e.dataTransfer?.files || []);
      onFiles(files);
    });
  }
}

function onFiles(files) {
  if (!files.length) return;
  state.selectedFiles = files;
  renderFileList(files);
}

function renderFileList(files) {
  const list = byId('fileListUl');
  const wrap = byId('fileList');
  if (!list || !wrap) return;
  list.innerHTML = '';
  for (const f of files) {
    list.appendChild(h('li', null, [`${f.name}`, h('span', { class: 'text-faint' }, [` ${(f.size / 1024).toFixed(1)} KB`])]));
  }
  wrap.style.display = 'block';
}

export async function processFiles() {
  const files = state.selectedFiles || [];
  if (!files.length) {
    showToast('请先选择文件', 'warn');
    return;
  }
  showProgress(0, '准备处理…');
  state.processedResults = [];

  for (let i = 0; i < files.length; i++) {
    const f = files[i];
    try {
      const text = await readAsText(f);
      let result;
      if (f.name.toLowerCase().endsWith('.json') || text.trim().startsWith('{')) {
        result = parseJsonFile(f.name, text);
      } else {
        result = state.processor.processSingleFile(f.name, text);
      }
      state.processedResults.push(result);
    } catch (e) {
      log.error(`处理失败 ${f.name}`, e);
      state.processedResults.push({ fileName: f.name, status: 'error', error: e.message });
    }
    showProgress(((i + 1) / files.length) * 100, `已处理 ${i + 1}/${files.length}`);
  }

  showProgress(100, '完成');
  showToast('文件处理完成', 'success');

  initFileCharts();
  renderResults(state.processedResults);
  updateFileCharts(state.processedResults);
  // 显示结果区
  const sec = byId('resultsSection') || byId('ws-analytics');
  if (sec) sec.style.display = '';
  setTimeout(() => hideProgress(), 600);
}

function readAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

function parseJsonFile(name, content) {
  try {
    const obj = JSON.parse(content);
    const animal = obj.animal || {};
    const vitals = obj.signals?.vitals?.samples || [];
    const accel = obj.signals?.accel?.samples || [];
    const tempArr = obj.signals?.temperature?.samples || [];
    const hrArr = vitals.map(s => s.hr).filter(Number.isFinite);
    const rrArr = vitals.map(s => s.rr).filter(Number.isFinite);
    const tArr  = tempArr.map(s => s.value).filter(Number.isFinite);
    const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

    const heartRateTimeAxis = vitals.map(s => `${s.t_s ?? 0}s`);
    return {
      fileName: name, dataType: 'json',
      status: 'success',
      animal,
      heartRate: Math.round(avg(hrArr)),
      respiratoryRate: Math.round(avg(rrArr)),
      temperature: +avg(tArr).toFixed(2),
      activity: { samples: accel.length },
      heartRateTimeSeries: hrArr,
      respiratoryRateTimeSeries: rrArr,
      heartRateTimeAxis,
    };
  } catch (e) {
    return { fileName: name, status: 'error', error: e.message };
  }
}

function showProgress(percent, text) {
  const wrap = byId('processingStatus') || byId('progressContainer');
  if (wrap) wrap.style.display = 'block';
  const fill = byId('progressFill');
  const tx = byId('progressText');
  if (fill) fill.style.width = `${percent}%`;
  if (tx) tx.textContent = text;
}
function hideProgress() {
  const wrap = byId('processingStatus') || byId('progressContainer');
  if (wrap) wrap.style.display = 'none';
}
