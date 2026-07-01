/**
 * IMU 姿态解算开关 + 算法切换 + 显示更新
 *
 * 把 app.js 里 toggleAttitude / changeAttitudeAlgorithm / updateAttitudeDisplay
 * 移到 ESM；与 posture-labeling.js 共享 state。
 */

import state from '../app/state.js';
import { byId, setText } from '../utils/dom.js';
import { AttitudeSolver, AttitudeVisualizer } from '../monitors/attitude-solver.js';
import { bus } from '../app/event-bus.js';
import { child } from '../utils/logger.js';

const log = child('attitude');

export function toggleAttitude() {
  if (!state.attitudeEnabled) {
    state.attitudeEnabled = true;
    if (!state.attitudeSolver) {
      state.attitudeSolver = new AttitudeSolver();
      state.attitudeSolver.setSampleRate(100);
    }
    const display = byId('attitudeDisplay');
    if (display) display.style.display = 'block';

    setTimeout(() => {
      if (!state.attitudeVisualizer) {
        state.attitudeVisualizer = new AttitudeVisualizer('attitude3DContainer');
      } else {
        state.attitudeVisualizer.onWindowResize?.();
        state.attitudeVisualizer.start?.();
      }
    }, 100);

    const btn = byId('attitudeEnableBtn');
    if (btn) {
      btn.textContent = '停止姿态解算';
      btn.classList.remove('btn-primary');
      btn.classList.add('btn-danger');
    }
    log.info('姿态解算已启用');
  } else {
    state.attitudeEnabled = false;
    state.attitudeVisualizer?.stop?.();
    const display = byId('attitudeDisplay');
    if (display) display.style.display = 'none';
    const btn = byId('attitudeEnableBtn');
    if (btn) {
      btn.textContent = '启用姿态解算';
      btn.classList.remove('btn-danger');
      btn.classList.add('btn-primary');
    }
    log.info('姿态解算已停止');
  }
}

export function changeAttitudeAlgorithm() {
  if (!state.attitudeSolver) return;
  const sel = byId('attitudeAlgorithm');
  const algo = sel?.value;
  if (!algo) return;
  state.attitudeSolver.setAlgorithm(algo);
  state.attitudeSolver.reset();
}

export function updateAttitudeDisplay() {
  if (!state.attitudeEnabled || !state.attitudeSolver) return;
  const a = state.attitudeSolver;

  const e = a.getEulerAngles?.();
  if (e) {
    setText('attitudePitch', e.pitch.toFixed(1) + '°');
    setText('attitudeRoll', e.roll.toFixed(1) + '°');
    setText('attitudeYaw', e.yaw.toFixed(1) + '°');
  }
  const q = a.getQuaternion?.();
  if (q) {
    setText('attitudeQW', q.w.toFixed(3));
    setText('attitudeQX', q.x.toFixed(3));
    setText('attitudeQY', q.y.toFixed(3));
    setText('attitudeQZ', q.z.toFixed(3));
  }
  const v = a.getAngularVelocity?.();
  if (v) {
    setText('attitudeGx', v.gx.toFixed(1));
    setText('attitudeGy', v.gy.toFixed(1));
    setText('attitudeGz', v.gz.toFixed(1));
  }

  const ax = state.bleBufferACC_X?.at(-1) ?? 0;
  const ay = state.bleBufferACC_Y?.at(-1) ?? 0;
  const az = state.bleBufferACC_Z?.at(-1) ?? 1;

  if (ax === 0 && ay === 0 && az === 1) return;

  const posture = a.classifyPosture?.(ax, ay, az);
  if (posture) {
    setText('animalPostureLabel', posture.label);
    setText('animalPostureConf', `${(posture.confidence * 100).toFixed(0)}%`);
  }

  if (state.postureRecordingFlag === 1 && state.manualPosture) {
    state.postureDataLog.push({
      t: Date.now(),
      manualLabel: state.manualPosture,
      ax, ay, az,
      pitch: e?.pitch, roll: e?.roll, yaw: e?.yaw,
      qw: q?.w, qx: q?.x, qy: q?.y, qz: q?.z,
    });
    setText('postureRecordCount', String(state.postureDataLog.length));
  }
}

bus.on('ble:tick', updateAttitudeDisplay);
