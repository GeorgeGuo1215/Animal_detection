/**
 * BLE 实时生理参数提取与稳定化
 *
 * 1. 取最近窗口（默认 30s）I/Q 数据
 * 2. 调用 RadarDataProcessor.extractVitalSignsMainPy 得到 HR/RR + 波形
 * 3. 维护循环平滑数组 (HR/RR History) + 心率最大变化限幅
 */

import state from '../app/state.js';
import { VITAL } from '../app/config.js';
import { byId, setText } from '../utils/dom.js';

export function updateBluetoothVitalSigns() {
  const fs = state.processor?.fs ?? 50;
  const win = Math.min(state.bleBufferI.length, fs * 30);
  if (win < fs * 5) return;
  const iData = new Float64Array(state.bleBufferI.slice(-win));
  const qData = new Float64Array(state.bleBufferQ.slice(-win));

  if (!state.processor || typeof state.processor.extractVitalSignsMainPy !== 'function') return;

  let result;
  try {
    result = state.processor.extractVitalSignsMainPy(iData, qData);
  } catch (e) {
    return;
  }
  const { heartRate, respiratoryRate, respiratoryWave, heartbeatWave } = result;

  // ====== 心率/呼吸平滑（参考 main.py 332-360）======
  state.heartRateHistory[state.historyIndex] = heartRate;
  state.respiratoryHistory[state.historyIndex] = respiratoryRate;
  state.historyIndex = (state.historyIndex + 1) % state.historyMaxLength;

  const avgHR = Math.round(state.heartRateHistory.reduce((a, b) => a + b, 0) / state.historyMaxLength);
  const avgRR = Math.round(state.respiratoryHistory.reduce((a, b) => a + b, 0) / state.historyMaxLength);

  let displayHR = avgHR;
  const delta = avgHR - state.lastStableHeartRate;
  if (Math.abs(delta) > state.heartRateDelta) {
    displayHR = state.lastStableHeartRate + Math.sign(delta) * state.heartRateDelta;
  }
  state.lastStableHeartRate = displayHR;
  state.lastStableRespRate = avgRR;
  state.currentHeartRate = displayHR;
  state.currentRespiratoryRate = avgRR;

  // ====== 更新 DOM 显示 ======
  setText('bleCurrentHR', `${displayHR} bpm`);
  setText('bleCurrentResp', `${avgRR} bpm`);
  setText('bleAvgHeartRate', `${displayHR} bpm`);
  setText('bleAvgRespRate', `${avgRR} bpm`);
  setText('bleCurrentHeartRate', `${displayHR} bpm`);
  setText('bleCurrentRespRate', `${avgRR} bpm`);
  setText('currentHeartRate', `${displayHR} bpm`);
  setText('currentRespRate', `${avgRR} bpm`);

  // ====== 推到 ECG 播放器 ======
  if (state.bleECG) {
    const norm = (arr) => {
      if (!arr.length) return arr;
      const m = arr.reduce((a, b) => a + b, 0) / arr.length;
      const s = Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length) || 1;
      return arr.map(v => (v - m) / (s * 3));
    };
    const len = Math.min(50, respiratoryWave.length);
    const resSeg = Array.from(respiratoryWave.slice(-len));
    const hbSeg  = Array.from(heartbeatWave.slice(-len));
    norm(resSeg).forEach(v => state.bleECG.res.data.push(v));
    norm(hbSeg).forEach(v => state.bleECG.hb.data.push(v));
    if (state.bleECG.res.data.length > 5000) state.bleECG.res.data.splice(0, state.bleECG.res.data.length - 5000);
    if (state.bleECG.hb.data.length > 5000)  state.bleECG.hb.data.splice(0,  state.bleECG.hb.data.length  - 5000);
    if (state.bleECG.draw) state.bleECG.draw();
  }

  // ====== 更新呼吸/心跳波形图表 ======
  const ss = Math.min(1000, iData.length);
  const idx = Array.from({ length: ss }, (_, i) => i);
  if (state.bleCharts.respiratory) {
    state.bleCharts.respiratory.data = {
      labels: idx,
      datasets: [{ label: '呼吸波形 (实时)', data: Array.from(respiratoryWave.slice(-ss)), borderColor: '#1D1D1F', backgroundColor: 'rgba(29,29,31,0.04)', tension: 0.1, pointRadius: 0, borderWidth: 1.4 }],
    };
    state.bleCharts.respiratory.update();
  }
  if (state.bleCharts.heartbeat) {
    state.bleCharts.heartbeat.data = {
      labels: idx,
      datasets: [{ label: '心跳波形 (实时)', data: Array.from(heartbeatWave.slice(-ss)), borderColor: '#1D1D1F', backgroundColor: 'rgba(29,29,31,0.04)', tension: 0.1, pointRadius: 0, borderWidth: 1.4 }],
    };
    state.bleCharts.heartbeat.update();
  }
}
