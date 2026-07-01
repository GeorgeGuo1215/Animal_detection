/**
 * BLE 单行协议解析器
 *
 * 协议示例：
 *   ADC:-3455 1176|Acc:0.012 0.987 0.034|Gyr:1.2 0.5 -0.3|T:23.4
 *   也兼容 JSON 格式 {ts, i, q, seq?}
 *   也兼容空格分隔 "<ts> <i> <q>" 与无标签纯浮点
 *
 * 输出统一结构（null 表示无效行）：
 *   {
 *     ts: number,       // epoch ms
 *     iVal: number,     // I 通道电压 (V)
 *     qVal: number,     // Q 通道电压 (V)
 *     adcI, adcQ,       // 原始 ADC 值
 *     accX, accY, accZ,
 *     gyrX, gyrY, gyrZ,
 *     temperature, seq
 *   }
 */

const FLOAT_RE = /[+-]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?/g;

function parsePairAfterLabel(text, label) {
  const idx = text.indexOf(label);
  if (idx < 0) return null;
  const seg = text.slice(idx + label.length);
  const firstField = seg.split('|')[0] || '';
  const nums = firstField.match(FLOAT_RE)?.map(parseFloat) || [];
  return nums.length >= 2 ? [nums[0], nums[1]] : null;
}

function parseTripletAfterLabel(text, label) {
  const idx = text.indexOf(label);
  if (idx < 0) return null;
  const seg = text.slice(idx + label.length);
  const firstField = seg.split('|')[0] || '';
  const nums = firstField.match(FLOAT_RE)?.map(parseFloat) || [];
  return nums.length >= 3 ? [nums[0], nums[1], nums[2]] : null;
}

/**
 * 解析单行 BLE 数据。第二参数是用于跨行配对单浮点的可变状态。
 *
 * @param {string} line
 * @param {{pendingFloat: number|null}} carry
 * @returns {object|null}
 */
export function parseBleLine(line, carry = { pendingFloat: null }) {
  const trimmed = String(line || '').trim();
  if (!trimmed) return null;

  let ts;
  let iVal, qVal;
  let adcI = 0, adcQ = 0;
  let accX = 0, accY = 0, accZ = 0;
  let gyrX = 0, gyrY = 0, gyrZ = 0;
  let temperature = null;
  let seq = null;

  try {
    // ADC: 主信号源（I/Q）
    const adc = parsePairAfterLabel(trimmed, 'ADC:') || parsePairAfterLabel(trimmed, 'adc:');
    if (adc) {
      adcI = adc[0];
      adcQ = adc[1];
      // 与 main.py:413 一致的电压换算
      iVal = ((adc[0] / 32767) + 1) * 3.3 / 2;
      qVal = ((adc[1] / 32767) + 1) * 3.3 / 2;
      ts = Date.now();
    }

    // IMU
    const gyr = parseTripletAfterLabel(trimmed, 'Gyr:')
             || parseTripletAfterLabel(trimmed, 'GYR:')
             || parseTripletAfterLabel(trimmed, 'GYR_');
    const acc = parseTripletAfterLabel(trimmed, 'Acc:')
             || parseTripletAfterLabel(trimmed, 'ACC:');
    if (acc) [accX, accY, accZ] = acc;
    if (gyr) [gyrX, gyrY, gyrZ] = gyr;
    else if (acc) [gyrX, gyrY, gyrZ] = acc;

    // 温度
    const tempIdx = trimmed.indexOf('T:');
    if (tempIdx >= 0) {
      const tempSeg = trimmed.slice(tempIdx + 2);
      const m = tempSeg.match(FLOAT_RE);
      if (m && m.length > 0) temperature = parseFloat(m[0]);
    }

    // 序号
    const seqMatch = trimmed.match(/(?:\bSEQ\b|\bseq\b|\bidx\b|\bindex\b)\s*[:=]\s*(\d+)/);
    if (seqMatch) seq = parseInt(seqMatch[1], 10);

    // 兜底解析（无 ADC: 标签时）
    if (!Number.isFinite(iVal) || !Number.isFinite(qVal)) {
      if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
        const obj = JSON.parse(trimmed);
        ts = obj.ts ?? Date.now();
        iVal = parseFloat(obj.i);
        qVal = parseFloat(obj.q);
        if (seq === null && obj.seq !== undefined) seq = parseInt(obj.seq, 10);
      } else {
        const parts = trimmed.split(/\s+/);
        if (parts.length >= 3) {
          ts = parts[0];
          iVal = parseFloat(parts[1]);
          qVal = parseFloat(parts[2]);
        } else {
          const matches = [...trimmed.matchAll(FLOAT_RE)];
          if (matches.length >= 2) {
            let firstStr = matches[0][0];
            let secondStr = matches[1][0];
            const secondIdx = matches[1].index;
            if (secondStr.startsWith('.') && secondIdx > 0) {
              const prevChar = trimmed[secondIdx - 1];
              if (prevChar >= '0' && prevChar <= '9' && /\d$/.test(firstStr)) {
                secondStr = prevChar + secondStr;
                firstStr = firstStr.slice(0, -1);
              }
            }
            ts = Date.now();
            iVal = parseFloat(firstStr);
            qVal = parseFloat(secondStr);
          } else if (matches.length === 1) {
            const v = parseFloat(matches[0][0]);
            if (!Number.isFinite(v)) return null;
            if (carry.pendingFloat === null) {
              carry.pendingFloat = v;
              return null;
            }
            ts = Date.now();
            iVal = carry.pendingFloat;
            qVal = v;
            carry.pendingFloat = null;
          } else {
            return null;
          }
        }
      }
    }
  } catch (_) {
    return null;
  }

  if (!Number.isFinite(iVal) || !Number.isFinite(qVal)) return null;

  if (typeof ts === 'string') {
    const t = Date.parse(ts);
    ts = Number.isFinite(t) ? t : Date.now();
  }
  if (!Number.isFinite(ts)) ts = Date.now();

  return {
    ts, iVal, qVal,
    adcI, adcQ,
    accX, accY, accZ,
    gyrX, gyrY, gyrZ,
    temperature, seq,
  };
}
