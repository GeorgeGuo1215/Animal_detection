/**
 * 文件分析图表（I/Q 信号、星座、呼吸/心跳波形、心率/呼吸分布与时间序列）
 */

import { lineChart, barChart, scatterChart, makeLineDataset, destroyChart, COLORS } from './chart-factory.js';
import state from '../app/state.js';
import { byId } from '../utils/dom.js';

export function initFileCharts() {
  if (typeof window.Chart === 'undefined') return;
  const required = ['iSignalChart','qSignalChart','constellationChart','respiratoryChart','heartbeatChart','heartRateChart','respRateChart','heartRateTimeChart','respRateTimeChart'];
  const missing = required.filter(id => !byId(id));
  if (missing.length) return;

  for (const k of Object.keys(state.charts)) destroyChart(state.charts[k]);
  state.charts = {};

  state.charts.iSignal = lineChart('iSignalChart', { title: 'I 通道信号 (放大显示)', xLabel: '采样点', yLabel: '幅度 (V)' });
  state.charts.qSignal = lineChart('qSignalChart', { title: 'Q 通道信号 (放大显示)', xLabel: '采样点', yLabel: '幅度 (V)' });
  state.charts.constellation = scatterChart('constellationChart', { title: 'I/Q 星座图', xLabel: 'I 通道', yLabel: 'Q 通道' });
  state.charts.respiratory   = lineChart('respiratoryChart', { title: '呼吸波形' });
  state.charts.heartbeat     = lineChart('heartbeatChart',   { title: '心跳波形' });
  state.charts.heartRate     = barChart('heartRateChart',    { title: '心率分布', xLabel: '文件', yLabel: '心率 (bpm)' });
  state.charts.respRate      = barChart('respRateChart',     { title: '呼吸频率分布', xLabel: '文件', yLabel: '呼吸 (bpm)' });
  state.charts.heartRateTime = lineChart('heartRateTimeChart', { title: '心率随时间变化', xLabel: '时间', yLabel: '心率 (bpm)' });
  state.charts.respRateTime  = lineChart('respRateTimeChart',  { title: '呼吸频率随时间变化', xLabel: '时间', yLabel: '呼吸 (bpm)' });
}

export function updateFileCharts(results) {
  if (results.length === 0) return;
  const first = results[0];
  if (first.dataType === 'json') return updateJsonCharts(results);

  const sampleSize = Math.min(1000, first.iData.length);
  const idx = Array.from({ length: sampleSize }, (_, i) => i);

  const iSlice = first.iData.slice(0, sampleSize);
  const qSlice = first.qData.slice(0, sampleSize);

  setLineRange(state.charts.iSignal, iSlice, 'I 通道');
  setLineRange(state.charts.qSignal, qSlice, 'Q 通道');

  const con = [];
  const step = Math.max(1, Math.floor(first.iData.length / 500));
  for (let i = 0; i < first.iData.length; i += step) con.push({ x: first.iData[i], y: first.qData[i] });
  if (state.charts.constellation) {
    state.charts.constellation.data = {
      datasets: [
        { label: 'I/Q 数据点', data: con, backgroundColor: 'rgba(29,29,31,0.5)', pointRadius: 1.6 },
        { label: '圆心', data: [{ x: first.circleCenter[0], y: first.circleCenter[1] }], backgroundColor: COLORS.text, pointRadius: 6 },
      ],
    };
    state.charts.constellation.update();
  }
  if (state.charts.respiratory && first.respiratoryWave) {
    state.charts.respiratory.data = {
      labels: idx,
      datasets: [makeLineDataset(`呼吸波形 (${first.respiratoryRate} bpm)`, Array.from(first.respiratoryWave.slice(0, sampleSize)), 0)],
    };
    state.charts.respiratory.update();
  }
  if (state.charts.heartbeat && first.heartbeatWave) {
    state.charts.heartbeat.data = {
      labels: idx,
      datasets: [makeLineDataset(`心跳波形 (${first.heartRate} bpm)`, Array.from(first.heartbeatWave.slice(0, sampleSize)), 1)],
    };
    state.charts.heartbeat.update();
  }

  const fileNames = results.map(r => r.fileName.length > 12 ? r.fileName.substring(0, 10) + '…' : r.fileName);
  if (state.charts.heartRate) {
    state.charts.heartRate.data = {
      labels: fileNames,
      datasets: [{ label: '心率 (bpm)', data: results.map(r => r.heartRate), backgroundColor: COLORS.text, borderRadius: 4 }],
    };
    state.charts.heartRate.update();
  }
  if (state.charts.respRate) {
    state.charts.respRate.data = {
      labels: fileNames,
      datasets: [{ label: '呼吸 (bpm)', data: results.map(r => r.respiratoryRate), backgroundColor: '#6E6E73', borderRadius: 4 }],
    };
    state.charts.respRate.update();
  }
  updateTimeSeries(results);
}

function setLineRange(chart, data, label) {
  if (!chart) return;
  const min = Math.min(...data), max = Math.max(...data);
  const pad = (max - min) * 0.05;
  chart.options.scales.y.min = min - pad;
  chart.options.scales.y.max = max + pad;
  chart.data = {
    labels: Array.from({ length: data.length }, (_, i) => i),
    datasets: [makeLineDataset(label, Array.from(data), 0)],
  };
  chart.update();
}

function updateTimeSeries(results) {
  let allHr = [], allRr = [], allLabels = [], cur = 0;
  for (const r of results) {
    if (r.heartRateTimeSeries && r.respiratoryRateTimeSeries && r.timeAxis) {
      const off = cur;
      r.timeAxis.forEach((t, i) => {
        const at = off + t;
        allLabels.push(`${Math.floor(at / 60)}:${String(Math.floor(at % 60)).padStart(2, '0')}`);
        allHr.push(r.heartRateTimeSeries[i]);
        allRr.push(r.respiratoryRateTimeSeries[i]);
      });
      cur += r.dataPoints / (state.processor?.fs || 50);
    }
  }
  if (allHr.length === 0) {
    allLabels = results.map((_, i) => `文件${i + 1}`);
    allHr = results.map(r => r.heartRate);
    allRr = results.map(r => r.respiratoryRate);
  }
  if (state.charts.heartRateTime) {
    state.charts.heartRateTime.data = { labels: allLabels, datasets: [makeLineDataset('心率', allHr, 0)] };
    state.charts.heartRateTime.update();
  }
  if (state.charts.respRateTime) {
    state.charts.respRateTime.data = { labels: allLabels, datasets: [makeLineDataset('呼吸', allRr, 1)] };
    state.charts.respRateTime.update();
  }
}

function updateJsonCharts(results) {
  const r = results[0];
  if (!r) return;
  const labels = r.heartRateTimeAxis || [];
  if (state.charts.heartRateTime) {
    state.charts.heartRateTime.data = { labels, datasets: [makeLineDataset('心率', r.heartRateTimeSeries || [], 0)] };
    state.charts.heartRateTime.update();
  }
  if (state.charts.respRateTime) {
    state.charts.respRateTime.data = { labels, datasets: [makeLineDataset('呼吸', r.respiratoryRateTimeSeries || [], 1)] };
    state.charts.respRateTime.update();
  }
}
