/**
 * 蓝牙连接诊断 + AI 诊断（依赖 Azure 配置）。
 */

import state from '../app/state.js';
import { byId } from '../utils/dom.js';
import { showToast } from '../ui/toast.js';
import { child } from '../utils/logger.js';
import { getActualFsHz, getLossRate } from './ble-stats.js';
import { logChat } from './ble-controller.js';

const log = child('diag');

export function buildBleDiagnostics() {
  const stats = state.bleStats || {};
  const lastGX = state.bleBufferIMU_X.at?.(-1) ?? null;
  const lastGY = state.bleBufferIMU_Y.at?.(-1) ?? null;
  const lastGZ = state.bleBufferIMU_Z.at?.(-1) ?? null;

  return {
    ts: new Date().toISOString(),
    bleConnected: !!state.bleConnected,
    samplingRateConfigHz: state.processor?.fs ?? null,
    receivedSamples: stats.received || 0,
    expectedSamples: stats.expected || 0,
    missedSamples: stats.missed || 0,
    lossRateEstimated: getLossRate(),
    actualReceiveRateHz: getActualFsHz(),
    jitterEmaMs: stats.gapJitterEmaMs || null,
    seqBased: !!stats.seqBased,
    buffers: {
      lenI: state.bleBufferI.length,
      lenQ: state.bleBufferQ.length,
      lenGX: state.bleBufferIMU_X.length,
    },
    imuLast: { gx: lastGX, gy: lastGY, gz: lastGZ },
    ui: {
      hasBleIMUChartCanvas: !!byId('bleIMUChart'),
      bluetoothChartsSectionDisplay: byId('bluetoothChartsSection')
        ? getComputedStyle(byId('bluetoothChartsSection')).display : null,
    },
  };
}

export async function bleQuickDiagnose() {
  const diag = buildBleDiagnostics();
  const json = JSON.stringify(diag, null, 2);
  log.info('诊断:', diag);
  try {
    await navigator.clipboard.writeText(json);
    showToast('连接诊断已复制到剪贴板', 'success');
  } catch {
    showToast('已生成诊断（复制失败，请手动从控制台获取）', 'warn');
  }
  logChat(`📊 连接诊断已生成`);
}

export async function bleAzureDiagnose() {
  if (!window.AzureGPTAnalyzer) {
    showToast('Azure 分析器未加载', 'error');
    return;
  }
  const az = new window.AzureGPTAnalyzer();
  const ok = az.loadConfig();
  if (!ok) {
    showToast('请先在 “AI 配置” 中填写 Azure 端点与密钥', 'warn');
    return;
  }

  const session = state._lastSessionStats || {};
  const summary = `本次 BLE 录制摘要：HR=${session.avgHR ?? '--'} bpm, RR=${session.avgRR ?? '--'} bpm, 持续=${session.duration ?? '--'}s, 数据点=${session.points ?? '--'}`;
  try {
    showToast('AI 诊断生成中…', 'info');
    const report = await az.analyzeText(summary);
    logChat(`🩺 AI 诊断已生成`);
    log.info(report);
    showToast('AI 诊断已生成（详见控制台/对话区）', 'success');
  } catch (e) {
    log.error(e);
    showToast(`AI 诊断失败: ${e.message}`, 'error');
  }
}
