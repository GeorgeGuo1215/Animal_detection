/**
 * BLE 接收统计：到达间隔、抖动 EMA、丢包率（基于可选 seq 字段）
 */

import state from '../app/state.js';

const ALPHA = 0.2;

export function updateLossStats(seq = null) {
  const stats = state.bleStats;
  const now = Date.now();
  if (!stats.startRxTs) stats.startRxTs = now;

  if (stats.lastRxTs > 0) {
    const gap = now - stats.lastRxTs;
    stats.lastGapMs = gap;
    stats.gapEmaMs = stats.gapEmaMs ? stats.gapEmaMs * (1 - ALPHA) + gap * ALPHA : gap;
    const dev = Math.abs(gap - stats.gapEmaMs);
    stats.gapJitterEmaMs = stats.gapJitterEmaMs
      ? stats.gapJitterEmaMs * (1 - ALPHA) + dev * ALPHA : dev;
  }
  stats.lastRxTs = now;
  stats.received += 1;

  if (Number.isFinite(seq)) {
    if (stats.lastSeq !== null && Number.isFinite(stats.lastSeq)) {
      const expected = seq - stats.lastSeq;
      if (expected > 0) {
        stats.expected += expected;
        if (expected > 1) stats.missed += (expected - 1);
      }
    }
    stats.lastSeq = seq;
    stats.seqBased = true;
  } else {
    stats.expected = stats.received;
  }
}

/** 估算实际接收速率 (Hz) */
export function getActualFsHz() {
  const s = state.bleStats;
  if (!s.startRxTs || !s.received) return null;
  const elapsed = Math.max(0.001, (Date.now() - s.startRxTs) / 1000);
  return s.received / elapsed;
}

export function getLossRate() {
  const s = state.bleStats;
  if (!s.expected) return null;
  return s.missed / Math.max(1, s.expected);
}
