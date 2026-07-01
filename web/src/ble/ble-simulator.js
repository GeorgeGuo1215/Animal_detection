/**
 * BLE 模拟数据：用于无设备情况下演示数据流。
 *
 * 周期性向 onLine 注入合成的 ADC + Acc + Gyr + 温度 行。
 */

import { handleBleLine } from './ble-controller.js';

let interval = null;
let t = 0;

export function startSimulationTest() {
  if (interval) return;
  t = 0;
  interval = setInterval(() => {
    const i = Math.round(((Math.sin(t * 0.05) * 0.6 + 1.5) / 3.3 * 2 - 1) * 32767);
    const q = Math.round(((Math.cos(t * 0.05) * 0.6 + 1.5) / 3.3 * 2 - 1) * 32767);
    const ax = (Math.sin(t * 0.02) * 0.4).toFixed(3);
    const ay = (Math.cos(t * 0.02) * 0.4).toFixed(3);
    const az = (1 + Math.sin(t * 0.01) * 0.05).toFixed(3);
    const gx = (Math.sin(t * 0.04) * 5).toFixed(2);
    const gy = (Math.cos(t * 0.04) * 5).toFixed(2);
    const gz = (Math.sin(t * 0.03) * 3).toFixed(2);
    const temp = (28 + Math.sin(t * 0.005) * 1.5).toFixed(1);
    const line = `ADC:${i} ${q}|Acc:${ax} ${ay} ${az}|Gyr:${gx} ${gy} ${gz}|T:${temp}`;
    handleBleLine(line);
    t++;
  }, 20); // 50Hz
}

export function stopSimulation() {
  if (interval) { clearInterval(interval); interval = null; }
}
