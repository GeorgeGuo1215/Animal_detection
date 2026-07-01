/**
 * 自适应图表的全局动作：重置 Y 轴 / 强制细节模式 / 显示/隐藏蓝牙图表 / ECG 控制
 */

import { bus } from '../app/event-bus.js';
import { byId } from '../utils/dom.js';
import state from '../app/state.js';
import {
  resetAdaptiveYAxis as chartsResetY,
  forceDetailMode as chartsForceDetail,
  initBluetoothCharts,
} from '../charts/bluetooth-charts.js';

export function resetAdaptiveYAxis() {
  chartsResetY();
  bus.emit('chart:reset');
}

export function forceDetailMode() {
  chartsForceDetail();
}

export function showBluetoothCharts() {
  const wrap = byId('bluetoothCharts');
  if (wrap) wrap.style.display = '';
  initBluetoothCharts();
}

export function hideBluetoothCharts() {
  const wrap = byId('bluetoothCharts');
  if (wrap) wrap.style.display = 'none';
}

export function forceReinitializeCharts() {
  initBluetoothCharts();
}

// ====== ECG / 文件 ECG 简化播放控制 ======
export function toggleECGPlayback() {
  const player = state.ecgPlayer;
  if (!player) return;
  if (player.isPlaying?.()) {
    player.pause?.();
    setEcgUI(false);
  } else {
    player.play?.();
    setEcgUI(true);
  }
}
export function resetECG() { state.ecgPlayer?.reset?.(); setEcgUI(false); }
export function testECG() { state.ecgPlayer?.test?.(); }

function setEcgUI(playing) {
  const play = byId('playBtn');
  const pause = byId('pauseBtn');
  if (play && pause) {
    play.style.display = playing ? 'none' : '';
    pause.style.display = playing ? '' : 'none';
  }
}

// ====== BLE ECG ======
export function toggleBLEECGPlayback() {
  const p = state.bleEcgPlayer;
  if (!p) return;
  if (p.isPlaying?.()) {
    p.pause?.();
    setBleEcgUI(false);
  } else {
    p.play?.();
    setBleEcgUI(true);
  }
}
export function resetBLEECG() { state.bleEcgPlayer?.reset?.(); setBleEcgUI(false); }

function setBleEcgUI(playing) {
  const play = byId('blePlayBtn');
  const pause = byId('blePauseBtn');
  if (play && pause) {
    play.style.display = playing ? 'none' : '';
    pause.style.display = playing ? '' : 'none';
  }
}
