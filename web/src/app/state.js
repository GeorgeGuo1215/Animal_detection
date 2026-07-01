/**
 * 共享状态：BLE 数据缓冲、当前生理参数、录制标志、活动/睡眠/姿态实例。
 *
 * 通过单例 export，避免分散在 RadarWebApp 实例上的耦合。
 */

import { BLE_BUFFER, VITAL } from './config.js';

const state = {
  // 连接
  bleConnected: false,
  bleConnectStartTime: null,
  lastBleRxTs: 0,
  pendingFloat: null,

  // I/Q 缓冲
  bleBufferI: [],
  bleBufferQ: [],
  bleBufferTimestamps: [],
  bleMaxBuffer: BLE_BUFFER.MAX,
  bleMaxBufferHard: BLE_BUFFER.HARD_MAX,

  // IMU 缓冲
  bleBufferIMU_X: [],
  bleBufferIMU_Y: [],
  bleBufferIMU_Z: [],
  bleBufferACC_X: [],
  bleBufferACC_Y: [],
  bleBufferACC_Z: [],
  bleBufferTemperature: [],

  // 计数与统计
  bleDataCount: 0,
  bleStats: {
    startRxTs: 0,
    lastRxTs: 0,
    received: 0,
    expected: 0,
    missed: 0,
    lastGapMs: 0,
    gapEmaMs: 0,
    gapJitterEmaMs: 0,
    lastSeq: null,
    seqBased: false,
  },

  // 心率/呼吸（参考 main.py 的稳定算法）
  heartRateHistory: new Array(VITAL.HISTORY_LEN).fill(VITAL.HEART_INIT),
  respiratoryHistory: new Array(VITAL.HISTORY_LEN).fill(VITAL.RESP_INIT),
  historyIndex: 0,
  historyMaxLength: VITAL.HISTORY_LEN,
  heartRateDelta: VITAL.HEART_DELTA_BPM,
  lastStableHeartRate: VITAL.HEART_INIT,
  lastStableRespRate: VITAL.RESP_INIT,
  currentHeartRate: null,
  currentRespiratoryRate: null,

  // 录制
  bleRecordingFlag: 0,
  bleRecordingData: [],
  bleRecordingRawData: [],
  bleRecordingStartTime: null,

  // 上报
  bleUploadEnabled: false,
  bleUploadIntervalSec: 10,
  bleUploadWindowSec: 10,
  bleLastUploadTs: 0,

  // 模块实例（懒挂载）
  activityMonitor: null,
  activityMonitorEnabled: false,
  sleepMonitor: null,
  sleepMonitorEnabled: false,
  attitudeSolver: null,
  attitudeVisualizer: null,
  attitudeEnabled: false,
  voiceRecognition: null,
  voiceRecognitionActive: false,

  // 处理器与图表
  processor: null,
  charts: {},
  bleCharts: {},
  fileECG: null,
  bleECG: null,

  // 自适应 Y 轴
  adaptiveYAxisEnabled: true,
  adaptiveSampleCount: 0,
  adaptiveStabilizeThreshold: 30,
  adaptiveStabilizeWindow: 50,
  adaptiveLastMinI: Infinity,
  adaptiveLastMaxI: -Infinity,
  adaptiveLastMinQ: Infinity,
  adaptiveLastMaxQ: -Infinity,
  adaptiveStabilized: false,

  // 文件分析
  selectedFiles: [],
  processedResults: [],

  // 姿态记录
  manualPostureLabel: '',
  postureRecordingActive: false,
  postureRecordedData: [],

  // 图表节流
  chartLastUpdateTs: 0,
  chartMinIntervalMs: 100,
  vitalLogLastTs: 0,
  lastVitalUpdateTime: 0,
};

export default state;
export { state };

/** 完全清空 BLE 缓冲（连接 / 重连 / 重置时调用） */
export function resetBleBuffers() {
  state.bleBufferI.length = 0;
  state.bleBufferQ.length = 0;
  state.bleBufferTimestamps.length = 0;
  state.bleBufferIMU_X.length = 0;
  state.bleBufferIMU_Y.length = 0;
  state.bleBufferIMU_Z.length = 0;
  state.bleBufferACC_X.length = 0;
  state.bleBufferACC_Y.length = 0;
  state.bleBufferACC_Z.length = 0;
  state.bleBufferTemperature.length = 0;
  state.bleDataCount = 0;
  state.pendingFloat = null;
  state.bleStats = {
    startRxTs: 0, lastRxTs: 0, received: 0, expected: 0, missed: 0,
    lastGapMs: 0, gapEmaMs: 0, gapJitterEmaMs: 0,
    lastSeq: null, seqBased: false,
  };
}

/** 重置自适应 Y 轴标志位 */
export function resetAdaptiveYAxis() {
  state.adaptiveSampleCount = 0;
  state.adaptiveLastMinI = Infinity;
  state.adaptiveLastMaxI = -Infinity;
  state.adaptiveLastMinQ = Infinity;
  state.adaptiveLastMaxQ = -Infinity;
  state.adaptiveStabilized = false;
}
