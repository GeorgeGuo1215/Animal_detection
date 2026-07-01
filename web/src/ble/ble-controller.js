/**
 * BLE 控制器：封装连接 / 断开 / 录制 / 行处理 / 缓冲管理。
 *
 * 此模块从原 app.js 的 initializeBLE / handleBLELine / addBLELog / 录制相关 等抽出。
 */

import ble from './ble-manager.js';
import { parseBleLine } from './protocol-parser.js';
import { updateLossStats } from './ble-stats.js';
import state, { resetBleBuffers } from '../app/state.js';
import { bus } from '../app/event-bus.js';
import { byId, setText } from '../utils/dom.js';
import { rafThrottle, throttle } from '../utils/throttle.js';
import { showToast } from '../ui/toast.js';
import { child } from '../utils/logger.js';
import { stopBleUpload } from './ble-uploader.js';
import { stopSimulation } from './ble-simulator.js';
import { setItem, getItem, removeItem } from '../utils/storage.js';
import { STORAGE_KEYS } from '../app/config.js';
import { toLocalISOString } from '../utils/timezone.js';

const log = child('ble');

const carry = { pendingFloat: null };

let logLines = [];
const renderLogThrottled = throttle(() => {
  const el = byId('bleLog');
  if (!el) return;
  el.style.whiteSpace = 'pre-line';
  el.textContent = logLines.join('\n');
  el.scrollTop = el.scrollHeight;
}, 200);

export function logChat(msg) {
  const ts = new Date().toLocaleTimeString();
  logLines.push(`[${ts}] ${msg}`);
  if (logLines.length > 120) logLines.splice(0, logLines.length - 120);
  renderLogThrottled();
}

let chartUpdateScheduled = false;
const requestChartUpdate = () => {
  if (chartUpdateScheduled) return;
  chartUpdateScheduled = true;
  requestAnimationFrame(() => {
    chartUpdateScheduled = false;
    bus.emit('ble:tick');
  });
};

function setRealtimeVisibility(visible) {
  const rt = byId('bleRealTimeData');
  if (rt) rt.style.display = visible ? 'block' : 'none';
}

function updateConnectButtons() {
  const c = byId('bleConnectBtn');
  if (c) c.style.display = state.bleConnected ? 'none' : 'inline-flex';
  const stopBtn = byId('bleStopRecordBtn');
  if (stopBtn) stopBtn.style.display = state.bleRecordingFlag === 1 ? 'inline-flex' : 'none';
}

/** 处理一行（用于真实 BLE + 模拟器） */
export function handleBleLine(line) {
  if (state.bleRecordingFlag === 1) {
    state.bleRecordingRawData.push(line);
  }

  state.lastBleRxTs = Date.now();

  const parsed = parseBleLine(line, carry);
  if (!parsed) return;

  const { ts, iVal, qVal, accX, accY, accZ, gyrX, gyrY, gyrZ, temperature, seq } = parsed;
  updateLossStats(seq);

  state.bleBufferTimestamps.push(ts);
  state.bleBufferI.push(iVal);
  state.bleBufferQ.push(qVal);
  state.bleBufferIMU_X.push(gyrX);
  state.bleBufferIMU_Y.push(gyrY);
  state.bleBufferIMU_Z.push(gyrZ);
  state.bleBufferACC_X.push(accX);
  state.bleBufferACC_Y.push(accY);
  state.bleBufferACC_Z.push(accZ);
  state.bleBufferTemperature.push(Number.isFinite(temperature) ? temperature : null);

  // 转发给监测模块
  bus.emit('ble:sample', { iVal, qVal, accX, accY, accZ, gyrX, gyrY, gyrZ, temperature, ts });

  // 录制时落盘格式（兼容原 main.py 风格）
  if (state.bleRecordingFlag === 1) {
    const stamp = new Date().toISOString().replace('T', ' ').slice(0, 19);
    const dataLine = `${stamp}  ${parsed.adcI}  ${parsed.adcQ}  ${accX.toFixed(3)}  ${accY.toFixed(3)}  ${accZ.toFixed(3)}  ${iVal.toFixed(6)}  ${qVal.toFixed(6)}  ${gyrX.toFixed(3)}  ${gyrY.toFixed(3)}  ${gyrZ.toFixed(3)}  ${temperature !== null ? temperature.toFixed(2) : 'N/A'}`;
    state.bleRecordingData.push(dataLine);
  }

  state.bleDataCount++;
  setText('bleDataCount', state.bleDataCount);
  setText('bleTotalDataPoints', state.bleDataCount);

  // 首次几条数据时确保 UI 区域显示
  if (state.bleDataCount <= 5) setRealtimeVisibility(true);

  trimBuffersIfNeeded();
  requestChartUpdate();

  // 节流：每 fs 个数据点（约 1s）做一次完整生理参数估计
  const fs = state.processor?.fs ?? 50;
  if (state.bleBufferI.length % fs === 0 && state.bleBufferI.length >= fs * 5) {
    bus.emit('ble:estimate-vitals');
  }
}

/* 用于在 BLE（蓝牙低功耗）数据缓冲区超过阈值时进行裁剪 */
function trimBuffersIfNeeded() {
  const len = state.bleBufferI.length;
  if (len <= state.bleMaxBufferHard) return;
  const removeCount = len - state.bleMaxBuffer;
  if (removeCount <= 0) return;
  const arrays = [
    state.bleBufferTimestamps,
    state.bleBufferI, state.bleBufferQ,
    state.bleBufferIMU_X, state.bleBufferIMU_Y, state.bleBufferIMU_Z,
    state.bleBufferACC_X, state.bleBufferACC_Y, state.bleBufferACC_Z,
    state.bleBufferTemperature,
  ];
  for (const a of arrays) {
    if (a.length >= removeCount) a.splice(0, removeCount);
  }
}

/** 初始化 BLE 事件回调，绑定 onLine / onConnect / onDisconnect */
export function initializeBleEvents() {
  if (!ble) return;

  ble.onConnect = (device) => {
    state.bleConnected = true;
    logChat(`已连接：${device?.name || '未知设备'} (${device?.id || ''})`);
    setRealtimeVisibility(true);
    state.bleConnectStartTime = Date.now();
    updateConnectButtons();

    // 保存设备信息
    if (device) {
      try {
        const info = { id: device.id, name: device.name || '', t: Date.now() };
        setItem(STORAGE_KEYS.BLE_SAVED_DEVICE, JSON.stringify(info));
      } catch {}
    }

    // 显示蓝牙图表区，触发 resize
    const cs = byId('bluetoothChartsSection');
    if (cs) cs.style.display = 'block';
    setTimeout(() => bus.emit('ble:state-change', { connected: true }), 80);
  };

  ble.onDisconnect = () => {
    state.bleConnected = false;
    logChat('已断开连接');
    setRealtimeVisibility(false);
    stopSimulation();
    stopBleUpload();
    updateConnectButtons();
    bus.emit('ble:state-change', { connected: false });
  };

  ble.onError = (err) => logChat(`错误：${err?.message || err}`);
  ble.onServiceDiscovered = (info) => logChat(info);
  ble.onLine = (line) => handleBleLine(line);

  updateConnectButtons();
  showSavedDeviceUi();
}

export async function bleConnect() {
  resetBleBuffers();
  return ble.connect();
}

export async function bleDisconnect() {
  stopSimulation();
  return ble.disconnect();
}

/** 一键重连：从 localStorage 恢复 + 提示 + 重新调用浏览器选择对话框 */
export async function bleReconnect() {
  showToast('正在尝试重连…', 'info');
  return bleConnect();
}

export function bleClearSaved() {
  removeItem(STORAGE_KEYS.BLE_SAVED_DEVICE);
  showSavedDeviceUi();
  showToast('已清除保存的设备', 'info');
}

export function showSavedDeviceUi() {
  const wrap = byId('bleSavedDevice');
  if (!wrap) return;
  try {
    const raw = getItem(STORAGE_KEYS.BLE_SAVED_DEVICE);
    if (!raw) {
      wrap.style.display = 'none';
      const reBtn = byId('bleReconnectBtn');
      const clearBtn = byId('bleClearSavedBtn');
      if (reBtn) reBtn.style.display = 'none';
      if (clearBtn) clearBtn.style.display = 'none';
      return;
    }
    const info = JSON.parse(raw);
    wrap.style.display = 'block';
    wrap.innerHTML = `<strong>已保存设备：</strong> ${info.name || '未命名'} <span class="text-faint">(${info.id})</span>`;
    const reBtn = byId('bleReconnectBtn');
    const clearBtn = byId('bleClearSavedBtn');
    if (reBtn) reBtn.style.display = 'inline-flex';
    if (clearBtn) clearBtn.style.display = 'inline-flex';
  } catch {
    wrap.style.display = 'none';
  }
}

/** 录制控制 */
export function bleStartRecording() {
  if (state.bleRecordingFlag === 1) return;
  state.bleRecordingFlag = 1;
  state.bleRecordingData.length = 0;
  state.bleRecordingRawData.length = 0;
  state.bleRecordingStartTime = Date.now();
  logChat('开始记录蓝牙数据…');
  bus.emit('recording:start');
  updateConnectButtons();
}

export function bleStopRecording() {
  if (state.bleRecordingFlag !== 1) return;
  state.bleRecordingFlag = 0;
  const duration = Math.round((Date.now() - (state.bleRecordingStartTime || Date.now())) / 1000);
  state._lastSessionStats = {
    points: state.bleRecordingData.length,
    duration,
    avgHR: state.lastStableHeartRate,
    avgRR: state.lastStableRespRate,
  };
  logChat(`记录结束：共 ${state.bleRecordingData.length} 行 / ${duration}s`);
  bus.emit('recording:stop');
  updateConnectButtons();
}

export function clearBluetoothData() {
  resetBleBuffers();
  state.bleRecordingData.length = 0;
  state.bleRecordingRawData.length = 0;
  setText('bleDataCount', 0);
  setText('bleTotalDataPoints', 0);
  bus.emit('chart:reset');
  showToast('蓝牙缓冲区已清空', 'info');
}

export function saveBluetoothData() {
  if (state.bleRecordingData.length === 0 && state.bleBufferI.length === 0) {
    showToast('暂无蓝牙数据可保存', 'warn');
    return;
  }
  const lines = state.bleRecordingData.length > 0
    ? state.bleRecordingData
    : state.bleBufferI.map((v, i) => `${i}\t${v}\t${state.bleBufferQ[i] ?? ''}`);
  const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `ble_data_${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(a.href);
  showToast('数据已下载', 'success');
}
