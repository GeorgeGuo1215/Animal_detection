/**
 * Legacy onclick → ESM 桥接
 *
 * 保持 HTML 中现有 onclick="xxx()" 写法不变，逐项映射到新的 ES Modules：
 * 任何 HTML 内联事件处理都会通过 window.* 路径找到对应的 ESM 实现。
 */

import { bus } from '../app/event-bus.js';
import state from '../app/state.js';

// BLE
import {
  bleConnect, bleDisconnect, bleReconnect, bleClearSaved,
  bleStartRecording, bleStopRecording, clearBluetoothData, saveBluetoothData,
} from '../ble/ble-controller.js';
import { startSimulationTest, stopSimulation } from '../ble/ble-simulator.js';
import { bleQuickDiagnose, bleAzureDiagnose } from '../ble/ble-diagnostics.js';
import { startBleUpload, stopBleUpload } from '../ble/ble-uploader.js';

// Charts / 显示控制
import {
  resetAdaptiveYAxis, forceDetailMode, showBluetoothCharts, hideBluetoothCharts,
  forceReinitializeCharts, toggleECGPlayback, resetECG, testECG,
  toggleBLEECGPlayback, resetBLEECG,
} from '../ui/adaptive-charts-actions.js';

// 设置 / 文件 / 导出
import { applySettings, clearFiles, exportResults, exportCharts, toggleSettings } from '../ui/settings-actions.js';
import { processFiles } from '../files/file-handler.js';

// 健康分析 + 报告
import { performHealthAnalysis } from '../health/health-analysis.js';
import {
  showAIConfig, saveAIConfig, testAIConnection,
  showPromptEditor, loadPromptTemplate, createNewPrompt, savePrompt, deletePrompt, previewPrompt,
  showRAGEditor, addRAGEntry, saveRAGEntry, cancelRAGEdit, importRAGData, exportRAGData,
  generateDiagnosticReport, exportReport, exportHealthReport, closeModal, openModal,
} from '../ui/modals.js';

// 健康对话
import {
  initializeHealthChat, sendChatMessage, clearChat,
} from '../health/health-chat.js';

// 监测器
import restingMonitor from '../monitors/resting-monitor.js';
import { showToast } from '../ui/toast.js';

// IMU 姿态 + 标注
import { toggleAttitude, changeAttitudeAlgorithm } from '../ui/attitude-controls.js';
import {
  togglePostureRecording, exportCurrentPostureData, diagnosePostureRecording,
} from '../ui/posture-labeling.js';

// 语音
import { toggleVoiceRecognition } from '../ui/voice-input.js';

// n8n
import { integrationMockLoadFixture, integrationMockSend } from '../integration/n8n-mock.js';

/**
 * 把所有兼容函数挂到 window 上，方便 HTML onclick 直接调用。
 */
export function installLegacyBridge() {
  // BLE 基础
  Object.assign(window, {
    bleConnect, bleDisconnect, bleReconnect, bleClearSaved,
    bleStartRecording, bleStopRecording,
    clearBluetoothData, saveBluetoothData,
    startSimulationTest, stopSimulation,
    bleQuickDiagnose, bleAzureDiagnose,
    bleStartUpload: startBleUpload,
    bleStopUpload: stopBleUpload,
  });

  // 图表 / ECG
  Object.assign(window, {
    resetAdaptiveYAxis, forceDetailMode,
    showBluetoothCharts, hideBluetoothCharts,
    forceReinitializeCharts,
    toggleECGPlayback, resetECG, testECG,
    toggleBLEECGPlayback, resetBLEECG,
  });

  // 设置 / 文件
  Object.assign(window, {
    applySettings, clearFiles, processFiles,
    exportResults, exportCharts, toggleSettings,
  });

  // 健康分析 + 报告
  Object.assign(window, {
    performHealthAnalysis,
    showAIConfig, saveAIConfig, testAIConnection,
    showPromptEditor, loadPromptTemplate, loadCustomPrompt: loadPromptTemplate,
    createNewPrompt, savePrompt, deletePrompt, previewPrompt,
    showRAGEditor, addRAGEntry, saveRAGEntry, cancelRAGEdit,
    importRAGData, exportRAGData,
    generateDiagnosticReport, exportReport, exportHealthReport,
    closeModal, openModal,
  });

  // 健康对话
  Object.assign(window, {
    initializeHealthChat, sendChatMessage,
    clearChatHistory: clearChat,
  });

  // 监测器（直接调用实例方法）
  Object.assign(window, {
    startRestingMonitor: () => restingMonitor.start(),
    stopRestingMonitor: () => restingMonitor.stop(),
    saveRestingData: () => restingMonitor.save(),
    clearRestingData: () => restingMonitor.clear(),
    configRestingMonitor: () => restingMonitor.config?.(),
    startActivityMonitor: () => {
      if (state.activityMonitor) {
        state.activityMonitor.initializeCharts?.();
        showToast('活动量监测已启动', 'success');
      }
    },
    stopActivityMonitor: () => showToast('已停止活动量监测', 'info'),
    resetActivityData: () => state.activityMonitor?.resetDailyData?.(),
    updateActivityGoals: () => {
      const m = state.activityMonitor;
      if (!m) return;
      const g = (id) => parseInt(document.getElementById(id)?.value, 10);
      const sg = g('stepGoalInput'); if (Number.isFinite(sg)) m.dailyGoal = sg;
      const ag = parseFloat(document.getElementById('activityGoalInput')?.value); if (Number.isFinite(ag)) m.activityGoal = ag;
      const cg = g('calorieGoalInput'); if (Number.isFinite(cg)) m.calorieGoal = cg;
      const wg = parseFloat(document.getElementById('petWeightInput')?.value);
      if (Number.isFinite(wg)) {
        m.petWeight = wg;
        m.rerDaily = 70 * Math.pow(wg, 0.75);
        m.bmrPerSec = m.rerDaily / 86400.0;
      }
      m.updateStatistics?.();
      showToast('已更新活动目标', 'success');
    },
    startSleepMonitor: () => {
      state.sleepActive = true;
      state.sleepMonitor?.initializeCharts?.();
      showToast('睡眠监测已开始', 'success');
    },
    stopSleepMonitor: () => {
      state.sleepActive = false;
      showToast('睡眠监测已停止', 'info');
    },
    resetSleepData: () => state.sleepMonitor?.reset?.(),
    generateSleepReport: () => state.sleepMonitor?.generateSleepReport?.(),
  });

  // IMU + 姿态标注
  Object.assign(window, {
    toggleAttitude, changeAttitudeAlgorithm,
    togglePostureRecording, exportCurrentPostureData, diagnosePostureRecording,
  });

  // 语音
  Object.assign(window, { toggleVoiceRecognition });

  // n8n
  Object.assign(window, { integrationMockLoadFixture, integrationMockSend });

  // 暴露 state（便于 onclick 内 `if(app)` 风格调用）
  // 同时把图表自适应方法挂到 state 上以兼容旧 HTML：
  //   onclick="if(app) app.forceReinitializeCharts();"
  state.forceReinitializeCharts = forceReinitializeCharts;
  state.resetAdaptiveYAxis = resetAdaptiveYAxis;
  state.forceDetailMode = forceDetailMode;
  window.app = state;

  // 通用：bus 暴露便于调试
  window.__petmindBus = bus;
}
