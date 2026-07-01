/**
 * 手动姿态标注 + 数据导出（IMU 训练数据采集）
 *
 * 提供：
 * - initManualPostureLabeling()   监听下拉菜单变化、初始化 state
 * - togglePostureRecording()      开始/停止记录并自动导出 JSON
 * - exportCurrentPostureData()    手动导出
 * - diagnosePostureRecording()    诊断当前记录链路状态
 */

import state from '../app/state.js';
import { byId } from '../utils/dom.js';
import { showToast } from './toast.js';
import { child } from '../utils/logger.js';

const log = child('posture');

export function initManualPostureLabeling() {
  const sel = byId('manualPostureInput');
  const disp = byId('manualPostureDisplay');
  if (!sel || !disp) {
    log.warn('manualPostureInput / manualPostureDisplay not found');
    return;
  }

  if (state.manualPosture === undefined) state.manualPosture = '';
  if (!state.postureDataLog) state.postureDataLog = [];
  if (state.postureRecordingFlag === undefined) state.postureRecordingFlag = 0;
  if (!state.postureRecordingStartTime) state.postureRecordingStartTime = null;

  sel.addEventListener('change', () => {
    const selected = sel.value;
    state.manualPosture = selected;

    if (selected === '初始佩戴静止') {
      if (state.attitudeSolver) {
        state.attitudeSolver.reset();
        disp.textContent = '⚙️ 校准中…';
        setTimeout(() => {
          disp.textContent = '✅ 校准完成';
          setTimeout(() => {
            sel.value = '';
            state.manualPosture = '';
            disp.textContent = '未标注';
          }, 1000);
        }, 3000);
      } else {
        showToast('请先启用 IMU 姿态解算', 'warn');
      }
      return;
    }

    disp.textContent = selected || '未标注';
  });

  const c = byId('postureRecordCount');
  if (c) c.textContent = '0';
}

export function togglePostureRecording() {
  if (!state.attitudeEnabled || !state.attitudeSolver) {
    showToast('请先启用 IMU 姿态解算', 'warn');
    return;
  }
  if (!state.manualPosture) {
    showToast('请先在下拉菜单中选择当前真实姿态', 'warn');
    return;
  }
  state.postureRecordingFlag = (1 + (state.postureRecordingFlag || 0)) % 2;
  const btn = byId('postureRecordStartBtn');

  if (state.postureRecordingFlag === 1) {
    state.postureDataLog = [];
    state.postureRecordingStartTime = new Date();
    const c = byId('postureRecordCount');
    const p = byId('postureDataPreview');
    if (c) c.textContent = '0';
    if (p) p.innerHTML = '<div class="text-faint">正在记录数据…</div>';
    if (btn) {
      btn.textContent = '⏹️ 停止记录';
      btn.classList.add('is-recording');
    }
    showToast(`开始记录姿态：${state.manualPosture}`, 'success');
  } else {
    const start = state.postureRecordingStartTime || new Date();
    const duration = ((Date.now() - start.getTime()) / 1000).toFixed(1);

    if (state.postureDataLog && state.postureDataLog.length > 0) {
      const ts = start.toISOString().slice(0, 16).replace('T', '-').replace(/:/g, '-');
      const filename = `posture_data_${ts}.json`;
      const exportData = {
        recordingStartTime: start.toISOString(),
        recordingEndTime: new Date().toISOString(),
        durationSeconds: parseFloat(duration),
        totalRecords: state.postureDataLog.length,
        note: '用于姿态判断算法矫正的训练数据',
        data: state.postureDataLog,
      };
      downloadJson(JSON.stringify(exportData, null, 2), filename);
      showToast(`已导出 ${state.postureDataLog.length} 条姿态数据`, 'success');
    } else {
      showToast('没有记录到任何姿态数据', 'warn');
    }

    if (btn) {
      btn.textContent = '🔴 开始记录姿态';
      btn.classList.remove('is-recording');
    }
    state.postureRecordingStartTime = null;
  }
}

export function exportCurrentPostureData() {
  if (!state.postureDataLog || state.postureDataLog.length === 0) {
    showToast('没有可导出的姿态数据', 'warn');
    return;
  }
  const ts = new Date().toISOString().slice(0, 16).replace('T', '-').replace(/:/g, '-');
  const filename = `posture_data_${ts}.json`;
  const payload = {
    exportTime: new Date().toISOString(),
    totalRecords: state.postureDataLog.length,
    note: '用于姿态判断算法矫正的训练数据',
    data: state.postureDataLog,
  };
  downloadJson(JSON.stringify(payload, null, 2), filename);
  showToast(`已导出 ${state.postureDataLog.length} 条姿态数据`, 'success');
}

export function diagnosePostureRecording() {
  const lines = [];
  lines.push('—— 姿态记录诊断 ——');
  lines.push(`姿态解算: ${state.attitudeEnabled ? '已启用' : '未启用'}`);
  lines.push(`手动标注: ${state.manualPosture || '未选择'}`);
  lines.push(`记录状态: ${state.postureRecordingFlag === 1 ? '记录中' : '未开始'}`);
  lines.push(`数据条数: ${state.postureDataLog?.length || 0}`);
  if (state.bleBufferACC_X?.length) {
    const x = state.bleBufferACC_X.at(-1)?.toFixed(3);
    const y = state.bleBufferACC_Y.at(-1)?.toFixed(3);
    const z = state.bleBufferACC_Z.at(-1)?.toFixed(3);
    lines.push(`加速度计: X=${x} Y=${y} Z=${z}`);
  }
  alert(lines.join('\n'));
}

function downloadJson(content, filename) {
  const blob = new Blob([content], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
