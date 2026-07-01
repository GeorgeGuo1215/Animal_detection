/**
 * 蓝牙实时图表（I/Q/星座/IMU/ACC/温度/呼吸/心跳）+ 自适应 Y 轴。
 */

import { lineChart, scatterChart, makeLineDataset, destroyChart, COLORS } from './chart-factory.js';
import state, { resetAdaptiveYAxis as resetAdaptive } from '../app/state.js';
import { rafThrottle } from '../utils/throttle.js';
import { byId } from '../utils/dom.js';
import { bus } from '../app/event-bus.js';
import { child } from '../utils/logger.js';

const log = child('bleChart');

export function initBluetoothCharts() {
  if (typeof window.Chart === 'undefined') {
    log.warn('Chart.js 未加载，跳过 BLE 图表初始化');
    return;
  }
  const required = ['bleISignalChart', 'bleQSignalChart', 'bleConstellationChart', 'bleRespiratoryChart', 'bleHeartbeatChart'];
  const missing = required.filter(id => !byId(id));
  if (missing.length) {
    log.warn('BLE 图表 DOM 缺失：', missing);
    return;
  }
  // 销毁旧实例，避免 “Canvas is already in use”
  for (const k of Object.keys(state.bleCharts)) destroyChart(state.bleCharts[k]);
  state.bleCharts = {};

  state.bleCharts.iSignal = lineChart('bleISignalChart', {
    title: '蓝牙 I 通道实时信号 (自适应放大)',
    xLabel: '采样点', yLabel: '幅度 (V)',
    options: { scales: { y: { min: 1.2, max: 2.8, beginAtZero: false } } },
  });
  state.bleCharts.qSignal = lineChart('bleQSignalChart', {
    title: '蓝牙 Q 通道实时信号 (自适应放大)',
    xLabel: '采样点', yLabel: '幅度 (V)',
    options: { scales: { y: { min: 1.2, max: 2.8, beginAtZero: false } } },
  });
  state.bleCharts.constellation = scatterChart('bleConstellationChart', {
    title: '蓝牙 I/Q 星座图', xLabel: 'I 通道', yLabel: 'Q 通道',
  });
  state.bleCharts.respiratory = lineChart('bleRespiratoryChart', { title: '蓝牙呼吸波形' });
  state.bleCharts.heartbeat   = lineChart('bleHeartbeatChart',   { title: '蓝牙心跳波形' });

  if (byId('bleIMUChart'))         state.bleCharts.imu          = lineChart('bleIMUChart', { title: '蓝牙 Gx/Gy/Gz 三轴变化' });
  if (byId('bleACCChart'))         state.bleCharts.acc          = lineChart('bleACCChart', { title: '蓝牙 Ax/Ay/Az 三轴变化' });
  if (byId('bleTemperatureChart')) state.bleCharts.temperature  = lineChart('bleTemperatureChart', {
    title: '蓝牙 温度变化 (°C)', yLabel: '温度 (°C)',
    options: { scales: { y: { min: 15, max: 45 } } },
  });
}

const updateLive = rafThrottle(() => {
  const { iSignal, qSignal, constellation, imu, acc, temperature } = state.bleCharts;
  if (!iSignal || !qSignal || !constellation) return;
  const len = state.bleBufferI.length;
  if (len < 10) return;

  // 自适应 Y 轴
  if (state.adaptiveYAxisEnabled && state.bleDataCount % 2 === 0) {
    state.adaptiveSampleCount++;
    const recentSize = Math.min(len, state.adaptiveStabilizeWindow);
    const startIdx = len - recentSize;
    const recentI = state.bleBufferI.slice(startIdx);
    const recentQ = state.bleBufferQ.slice(startIdx);
    const minI = Math.min(...recentI), maxI = Math.max(...recentI);
    const minQ = Math.min(...recentQ), maxQ = Math.max(...recentQ);

    if (state.adaptiveStabilized) {
      const cMinI = iSignal.options.scales.y.min, cMaxI = iSignal.options.scales.y.max;
      const cMinQ = qSignal.options.scales.y.min, cMaxQ = qSignal.options.scales.y.max;
      if (minI < cMinI || maxI > cMaxI || minQ < cMinQ || maxQ > cMaxQ) {
        resetAdaptive();
      }
    }

    if (!state.adaptiveStabilized) {
      state.adaptiveLastMinI = Math.min(state.adaptiveLastMinI, minI);
      state.adaptiveLastMaxI = Math.max(state.adaptiveLastMaxI, maxI);
      state.adaptiveLastMinQ = Math.min(state.adaptiveLastMinQ, minQ);
      state.adaptiveLastMaxQ = Math.max(state.adaptiveLastMaxQ, maxQ);

      if (state.adaptiveSampleCount >= state.adaptiveStabilizeThreshold) {
        const rangeI = state.adaptiveLastMaxI - state.adaptiveLastMinI;
        const rangeQ = state.adaptiveLastMaxQ - state.adaptiveLastMinQ;
        const stdI = rangeI * 0.1, stdQ = rangeQ * 0.1;
        let nMinI, nMaxI, nMinQ, nMaxQ;
        if (rangeI <= 0.2 || rangeQ <= 0.2) {
          const cI = (state.adaptiveLastMinI + state.adaptiveLastMaxI) / 2;
          const cQ = (state.adaptiveLastMinQ + state.adaptiveLastMaxQ) / 2;
          nMinI = Math.max(0, cI - 0.05); nMaxI = cI + 0.05;
          nMinQ = Math.max(0, cQ - 0.05); nMaxQ = cQ + 0.05;
        } else {
          const padI = Math.max(0.01, Math.min(stdI * 3, rangeI * 0.20));
          const padQ = Math.max(0.01, Math.min(stdQ * 3, rangeQ * 0.20));
          nMinI = Math.max(0, state.adaptiveLastMinI - padI); nMaxI = state.adaptiveLastMaxI + padI;
          nMinQ = Math.max(0, state.adaptiveLastMinQ - padQ); nMaxQ = state.adaptiveLastMaxQ + padQ;
        }
        iSignal.options.scales.y.min = nMinI; iSignal.options.scales.y.max = nMaxI;
        qSignal.options.scales.y.min = nMinQ; qSignal.options.scales.y.max = nMaxQ;
        state.adaptiveStabilized = true;
      }
    }
  }

  const sampleSize = Math.min(1000, len);
  const start = len - sampleSize;
  const indices = Array.from({ length: sampleSize }, (_, i) => i);

  iSignal.data = { labels: indices, datasets: [makeLineDataset('I 通道', state.bleBufferI.slice(start), 0)] };
  qSignal.data = { labels: indices, datasets: [makeLineDataset('Q 通道', state.bleBufferQ.slice(start), 1)] };
  iSignal.update('none');
  qSignal.update('none');

  const conSize = Math.min(500, len);
  const step = Math.max(1, Math.floor(len / conSize));
  const points = [];
  for (let i = start; i < len; i += step) points.push({ x: state.bleBufferI[i], y: state.bleBufferQ[i] });
  constellation.data = { datasets: [{ label: 'I/Q 数据点', data: points, backgroundColor: 'rgba(29,29,31,0.45)', pointRadius: 1.6 }] };
  constellation.update('none');

  if (imu && state.bleBufferIMU_X.length > 0) {
    imu.data = {
      labels: indices,
      datasets: [
        makeLineDataset('gx', state.bleBufferIMU_X.slice(start), 0),
        makeLineDataset('gy', state.bleBufferIMU_Y.slice(start), 1),
        makeLineDataset('gz', state.bleBufferIMU_Z.slice(start), 2),
      ],
    };
    imu.update('none');
  }
  if (acc && state.bleBufferACC_X.length > 0) {
    acc.data = {
      labels: indices,
      datasets: [
        makeLineDataset('ax', state.bleBufferACC_X.slice(start), 0),
        makeLineDataset('ay', state.bleBufferACC_Y.slice(start), 1),
        makeLineDataset('az', state.bleBufferACC_Z.slice(start), 2),
      ],
    };
    acc.update('none');
  }
  if (temperature && state.bleBufferTemperature.length > 0) {
    const data = state.bleBufferTemperature.slice(start);
    const valid = data.filter(v => v !== null);
    const cur = valid.length ? valid[valid.length - 1] : null;
    temperature.data = {
      labels: indices,
      datasets: [{
        label: cur != null ? `温度 (°C) - 最新 ${cur.toFixed(1)}°C` : '温度 (°C) - 无数据',
        data, borderColor: COLORS.text, backgroundColor: 'rgba(29,29,31,0.04)',
        tension: 0.3, pointRadius: 0, fill: true, spanGaps: false,
      }],
    };
    temperature.update('none');

    const tempEl = byId('bleCurrentTemp');
    const avgTempEl = byId('bleAvgTemp');
    if (cur != null) {
      if (tempEl) tempEl.textContent = `${cur.toFixed(1)} °C`;
      if (avgTempEl) avgTempEl.textContent = `${cur.toFixed(1)} °C`;
    }
  }

  // 当前 Gyr/Acc 文本
  if (state.bleBufferIMU_X.length) {
    const gx = state.bleBufferIMU_X.at(-1), gy = state.bleBufferIMU_Y.at(-1), gz = state.bleBufferIMU_Z.at(-1);
    const el = byId('bleCurrentGyr');
    if (el) el.textContent = `gx:${gx.toFixed(1)} gy:${gy.toFixed(1)} gz:${gz.toFixed(1)}`;
  }
  if (state.bleBufferACC_X.length) {
    const ax = state.bleBufferACC_X.at(-1), ay = state.bleBufferACC_Y.at(-1), az = state.bleBufferACC_Z.at(-1);
    const el = byId('bleCurrentAcc');
    if (el) el.textContent = `ax:${ax.toFixed(3)} ay:${ay.toFixed(3)} az:${az.toFixed(3)}`;
  }
});

bus.on('ble:tick', updateLive);
bus.on('chart:reset', () => {
  for (const k of Object.keys(state.bleCharts)) destroyChart(state.bleCharts[k]);
  state.bleCharts = {};
  resetAdaptive();
});

export function resetAdaptiveYAxis() {
  resetAdaptive();
  if (state.bleCharts.iSignal) {
    state.bleCharts.iSignal.options.scales.y.min = 0;
    state.bleCharts.iSignal.options.scales.y.max = 4.0;
    state.bleCharts.iSignal.update();
  }
  if (state.bleCharts.qSignal) {
    state.bleCharts.qSignal.options.scales.y.min = 0;
    state.bleCharts.qSignal.options.scales.y.max = 4.0;
    state.bleCharts.qSignal.update();
  }
}

export function forceDetailMode() {
  if (state.bleBufferI.length < 50) return;
  const size = Math.min(state.bleBufferI.length, 50);
  const startIdx = state.bleBufferI.length - size;
  const dI = state.bleBufferI.slice(startIdx);
  const dQ = state.bleBufferQ.slice(startIdx);
  const minI = Math.min(...dI), maxI = Math.max(...dI);
  const minQ = Math.min(...dQ), maxQ = Math.max(...dQ);
  const padI = Math.max(0.02, (maxI - minI) * 0.02);
  const padQ = Math.max(0.02, (maxQ - minQ) * 0.02);
  if (state.bleCharts.iSignal) {
    state.bleCharts.iSignal.options.scales.y.min = Math.max(0, minI - padI);
    state.bleCharts.iSignal.options.scales.y.max = maxI + padI;
    state.bleCharts.iSignal.update();
  }
  if (state.bleCharts.qSignal) {
    state.bleCharts.qSignal.options.scales.y.min = Math.max(0, minQ - padQ);
    state.bleCharts.qSignal.options.scales.y.max = maxQ + padQ;
    state.bleCharts.qSignal.update();
  }
  state.adaptiveStabilized = false;
}
