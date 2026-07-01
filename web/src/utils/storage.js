/**
 * localStorage 安全包装（防 quota 超限、JSON 序列化）
 */

export function getItem(key, fallback = null) {
  try {
    const v = localStorage.getItem(key);
    return v == null ? fallback : v;
  } catch { return fallback; }
}

export function setItem(key, value) {
  try {
    localStorage.setItem(key, value == null ? '' : String(value));
    return true;
  } catch { return false; }
}

export function removeItem(key) {
  try { localStorage.removeItem(key); } catch {}
}

export function getJSON(key, fallback = null) {
  try {
    const v = localStorage.getItem(key);
    if (v == null) return fallback;
    return JSON.parse(v);
  } catch { return fallback; }
}

export function setJSON(key, obj) {
  try {
    localStorage.setItem(key, JSON.stringify(obj));
    return true;
  } catch { return false; }
}
