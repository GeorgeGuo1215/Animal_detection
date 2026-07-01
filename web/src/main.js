/**
 * 入口文件 / 启动器
 *
 * 在 DOMContentLoaded 之后：
 *   1. 装配 RadarDataProcessor / AzureGPTAnalyzer / 监测器实例并挂到 state
 *   2. 安装 legacy-bridge，把所有 onclick 函数挂到 window
 *   3. 初始化 BLE 事件、上传配置、对话面板、姿态标注、文件拖拽
 *   4. 启动 workspace 路由（与 hash 同步）
 *   5. 订阅事件总线：ble:tick → 实时图表 / 心率呼吸 / 监测；ble:estimate-vitals → 解算
 */

import state from './app/state.js';
import { bus } from './app/event-bus.js';
import { logger, child } from './utils/logger.js';

import RadarDataProcessor from './processors/radar-processor.js';
import AzureGPTAnalyzer from './health/azure-gpt.js';
import ActivityMonitor from './monitors/activity-monitor.js';
import SleepMonitor from './monitors/sleep-monitor.js';
import restingMonitor from './monitors/resting-monitor.js';

import { initializeBleEvents } from './ble/ble-controller.js';
import { initBleUploadConfig } from './ble/ble-uploader.js';
import { initBluetoothCharts } from './charts/bluetooth-charts.js';
import { updateBluetoothVitalSigns } from './processors/vital-signs.js';

import { initWorkspaceRouter } from './ui/workspace-router.js';
import { initManualPostureLabeling } from './ui/posture-labeling.js';
import './ui/attitude-controls.js';

import { bindFileHandlers } from './files/file-handler.js';
import { initializeHealthChat } from './health/health-chat.js';
import { integrationMockInit } from './integration/n8n-mock.js';
import { showToast } from './ui/toast.js';
import { loadPersistedPrompts, loadPersistedRAG } from './ui/modals.js';

import { installLegacyBridge } from './bootstrap/legacy-bridge.js';
import { applyChartTheme } from './charts/chart-theme.js';

const log = child('main');

function bootstrap() {
  log.info('PetMind Web 启动中…');

  // 1) 全局实例
  if (typeof window.Chart !== 'undefined') {
    try { applyChartTheme(); } catch (e) { log.warn('Chart 主题应用失败：', e); }
  }
  state.processor = new RadarDataProcessor();
  state.azureGPT = new AzureGPTAnalyzer();
  state.activityMonitor = new ActivityMonitor(50, 10);
  state.activityMonitorEnabled = true;
  state.sleepMonitor = new SleepMonitor(50, 10);
  // 静息监测需要绑定 app（即 state）以读取 IMU / 心率
  if (typeof restingMonitor.bindToApp === 'function') {
    restingMonitor.bindToApp(state);
  }

  // 2) 注入 legacy-bridge（HTML onclick → ESM）
  installLegacyBridge();

  // 3) 各子系统初始化
  initializeBleEvents();
  initBleUploadConfig();
  initManualPostureLabeling();
  bindFileHandlers();
  initWorkspaceRouter();
  integrationMockInit();
  loadPersistedPrompts();
  loadPersistedRAG();

  // 健康对话：探活 + 历史（成功连接后再 loadChatHistory，避免先把欢迎语清空）
  try { initializeHealthChat(); } catch (e) { log.warn('Agent 探活失败：', e); }

  // 4) 事件总线：实时图表 + 生理参数 + 监测分发
  bus.on('ble:tick', () => {
    // 由 bluetooth-charts.js 内部已经订阅 tick 做实时图表更新
  });

  bus.on('ble:estimate-vitals', () => {
    try { updateBluetoothVitalSigns(); } catch (e) { log.warn('vitals 计算失败：', e); }
  });

  // BLE 单点：转发给监测器
  bus.on('ble:sample', (s) => {
    const ts = s.ts ?? Date.now();
    if (state.activityMonitor) {
      try {
        state.activityMonitor.addAccelerometerData?.(s.accX, s.accY, s.accZ, ts);
        state.activityMonitor.calculateActivityMetrics?.();
      } catch (_) { /* ignore */ }
    }
    if (state.sleepActive && state.sleepMonitor) {
      try {
        state.sleepMonitor.addAccelerometerData?.(s.accX, s.accY, s.accZ, ts);
        state.sleepMonitor.processAccelerometerData?.();
      } catch (_) { /* ignore */ }
    }
    if (restingMonitor?.enabled) {
      try { restingMonitor.update?.(); } catch (_) { /* ignore */ }
    }
  });

  // 5) 状态变化：连接成功后初始化蓝牙图表
  bus.on('ble:state-change', (e) => {
    if (e?.connected) {
      try { initBluetoothCharts(); } catch (err) { log.warn('图表初始化失败：', err); }
    }
  });

  log.info('PetMind Web 启动完成 ✓');
  showToast('PetMind 已就绪', 'success', { duration: 2200 });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}

// 导出（仅供 devtools）
window.PetMind = { state, bus, logger };
