/**
 * 时区与时间戳格式化
 */

/** 把任意时间戳形式标准化成 epoch ms */
export function toEpochMs(ts) {
  if (ts == null) return Date.now();
  if (typeof ts === 'number') {
    return ts < 1e12 ? Math.round(ts * 1000) : Math.round(ts);
  }
  if (typeof ts === 'string') {
    const t = Date.parse(ts);
    return Number.isFinite(t) ? t : Date.now();
  }
  if (ts instanceof Date) return ts.getTime();
  return Date.now();
}

/** 当前时区偏移，例如 +08:00 / -05:30 */
export function formatTimezoneOffset(date = new Date()) {
  const offsetMin = -date.getTimezoneOffset();
  const sign = offsetMin >= 0 ? '+' : '-';
  const abs = Math.abs(offsetMin);
  const hh = String(Math.floor(abs / 60)).padStart(2, '0');
  const mm = String(abs % 60).padStart(2, '0');
  return `${sign}${hh}:${mm}`;
}

/** ISO 字符串（带本地时区偏移） */
export function toLocalISOString(date = new Date()) {
  const pad = (n) => String(n).padStart(2, '0');
  const yyyy = date.getFullYear();
  const MM = pad(date.getMonth() + 1);
  const dd = pad(date.getDate());
  const hh = pad(date.getHours());
  const mm = pad(date.getMinutes());
  const ss = pad(date.getSeconds());
  return `${yyyy}-${MM}-${dd}T${hh}:${mm}:${ss}${formatTimezoneOffset(date)}`;
}

export function fmtTime(ts = Date.now()) {
  const d = new Date(ts);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
