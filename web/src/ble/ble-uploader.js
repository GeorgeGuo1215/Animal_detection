/**
 * BLE 周期上报到 /integration/ingest
 *
 * 与原 app.js 的 startBleUpload / _sendBleUploadOnce / _buildBleEventPayload 等价。
 */

import state from '../app/state.js';
import { byId, getNum, getStr } from '../utils/dom.js';
import { setItem } from '../utils/storage.js';
import { toLocalISOString, toEpochMs, formatTimezoneOffset } from '../utils/timezone.js';
import { STORAGE_KEYS, URLS } from '../app/config.js';
import { showToast } from '../ui/toast.js';
import { child } from '../utils/logger.js';

const log = child('upload');

let timer = null;

function setStatus(text) {
  const el = byId('bleUploadStatus');
  if (el) el.textContent = text;
}

function getConfig() {
  return {
    url: getStr('bleUploadUrl').trim() || URLS.INTEGRATION_DEFAULT,
    animalId: getStr('bleAnimalId').trim(),
    deviceId: getStr('bleDeviceId').trim(),
    intervalSec: Math.max(5, Math.floor(getNum('bleUploadInterval', state.bleUploadIntervalSec))),
  };
}

function buildPayload() {
  const cfg = getConfig();
  const fs = state.processor?.fs ?? 50;
  const len = state.bleBufferI.length;
  if (len < Math.max(10, fs * 2)) {
    log.warn('数据不足，跳过本次上报');
    return null;
  }

  const windowSize = Math.min(len, Math.max(10, fs * state.bleUploadWindowSec));
  const startIdx = len - windowSize;
  const endIdx = len - 1;

  const startTsMs = toEpochMs(state.bleBufferTimestamps[startIdx]);
  const endTsMs = toEpochMs(state.bleBufferTimestamps[endIdx]);
  const timezone = formatTimezoneOffset();

  const accelSamples = [];
  const tempSamples = [];
  let lastTempSecond = -1;

  for (let i = startIdx; i <= endIdx; i++) {
    const tMs = Math.round(((i - startIdx) / fs) * 1000);
    const tS = Math.floor((i - startIdx) / fs);
    accelSamples.push({
      t_ms: tMs,
      x: Number(state.bleBufferIMU_X[i] || 0),
      y: Number(state.bleBufferIMU_Y[i] || 0),
      z: Number(state.bleBufferIMU_Z[i] || 0),
    });
    if (tS !== lastTempSecond) {
      tempSamples.push({
        t_s: tS,
        value: Number(state.bleBufferTemperature[i] || 0),
      });
      lastTempSecond = tS;
    }
  }

  const vitalsSamples = [];
  if (Number.isFinite(state.currentHeartRate) || Number.isFinite(state.currentRespiratoryRate)) {
    vitalsSamples.push({
      t_s: 0,
      hr: Number.isFinite(state.currentHeartRate) ? Number(state.currentHeartRate) : null,
      rr: Number.isFinite(state.currentRespiratoryRate) ? Number(state.currentRespiratoryRate) : null,
    });
  }

  return {
    event_id: `ble_${Date.now()}`,
    ts: new Date(endTsMs).toISOString(),
    animal: {
      animal_id: cfg.animalId, species: 'other',
      name: 'unknown', breed: 'unknown', sex: 'unknown',
      age_months: 0, weight_kg: 0,
    },
    device: {
      device_id: cfg.deviceId, firmware: 'unknown',
      sampling_hz: { accel: fs, temperature: fs, temp: fs, vitals: 1 },
    },
    window: {
      start_ts: new Date(startTsMs).toISOString(),
      end_ts: new Date(endTsMs).toISOString(),
      timezone,
    },
    context: {
      notes: 'web ble upload', tags: ['web', 'ble'],
      location: { lat: 0, lng: 0, accuracy_m: 0 },
    },
    signals: {
      accel: { samples: accelSamples },
      temperature: { samples: tempSamples },
      vitals: { samples: vitalsSamples },
    },
  };
}

async function sendOnce() {
  if (!state.bleUploadEnabled) return;
  const cfg = getConfig();
  if (!cfg.url) return;
  const payload = buildPayload();
  if (!payload) return;

  try {
    const resp = await fetch(cfg.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      setStatus(`失败 (${resp.status})`);
      log.error(`上报失败: HTTP ${resp.status}`);
      return;
    }
    state.bleLastUploadTs = Date.now();
    setStatus(`上传中 · 最近 ${new Date(state.bleLastUploadTs).toLocaleTimeString()}`);
  } catch (e) {
    setStatus('异常');
    log.error('上报异常:', e);
  }
}

export function startBleUpload() {
  if (!state.bleConnected) {
    showToast('请先连接蓝牙设备', 'warn');
    return;
  }
  const cfg = getConfig();
  if (!cfg.url || !cfg.animalId || !cfg.deviceId) {
    showToast('请填写上报接口、animal_id 与 device_id', 'warn');
    return;
  }

  setItem(STORAGE_KEYS.BLE_UPLOAD_URL, cfg.url);
  setItem(STORAGE_KEYS.BLE_ANIMAL_ID, cfg.animalId);
  setItem(STORAGE_KEYS.BLE_DEVICE_ID, cfg.deviceId);
  setItem(STORAGE_KEYS.BLE_UPLOAD_INTERVAL, String(cfg.intervalSec));

  state.bleUploadEnabled = true;
  state.bleUploadIntervalSec = cfg.intervalSec;
  setStatus('上传中…');

  if (timer) clearInterval(timer);
  sendOnce();
  timer = setInterval(sendOnce, cfg.intervalSec * 1000);

  showButtonsForUploading(true);
}

export function stopBleUpload() {
  state.bleUploadEnabled = false;
  if (timer) { clearInterval(timer); timer = null; }
  setStatus('未上传');
  showButtonsForUploading(false);
}

function showButtonsForUploading(uploading) {
  const start = byId('bleStartUploadBtn');
  const stop = byId('bleStopUploadBtn');
  if (start) start.style.display = uploading ? 'none' : 'inline-flex';
  if (stop)  stop.style.display  = uploading ? 'inline-flex' : 'none';
}

/** 初始化上报相关 UI（恢复 localStorage） */
export function initBleUploadConfig() {
  const url = (typeof localStorage !== 'undefined' && localStorage.getItem(STORAGE_KEYS.BLE_UPLOAD_URL)) || URLS.INTEGRATION_DEFAULT;
  const animal = (typeof localStorage !== 'undefined' && localStorage.getItem(STORAGE_KEYS.BLE_ANIMAL_ID)) || 'dog_001';
  const device = (typeof localStorage !== 'undefined' && localStorage.getItem(STORAGE_KEYS.BLE_DEVICE_ID)) || 'ble_device_01';
  const interval = parseInt((typeof localStorage !== 'undefined' && localStorage.getItem(STORAGE_KEYS.BLE_UPLOAD_INTERVAL)) || '10', 10);

  const urlEl = byId('bleUploadUrl');
  if (urlEl && !urlEl.value) urlEl.value = url;
  const animalEl = byId('bleAnimalId');
  if (animalEl && !animalEl.value) animalEl.value = animal;
  const deviceEl = byId('bleDeviceId');
  if (deviceEl && !deviceEl.value) deviceEl.value = device;
  const intervalEl = byId('bleUploadInterval');
  if (intervalEl && !intervalEl.value) intervalEl.value = String(interval > 0 ? interval : 10);

  state.bleUploadIntervalSec = interval > 0 ? interval : 10;

  setStatus('未上传');
  showButtonsForUploading(false);
}

export { sendOnce as sendBleUploadOnce, buildPayload as buildBleEventPayload };
