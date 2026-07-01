/**
 * 健康对话历史持久化（localStorage）
 */

import { STORAGE_KEYS } from '../app/config.js';
import { getJSON, setJSON } from '../utils/storage.js';

const KEY = STORAGE_KEYS.CHAT_HISTORY;

export function getChatHistory() {
  const list = getJSON(KEY, []);
  return Array.isArray(list) ? list : [];
}

export function saveChatMessage(role, content) {
  const list = getChatHistory();
  list.push({ role, content, timestamp: new Date().toISOString() });
  if (list.length > 50) list.splice(0, list.length - 50);
  setJSON(KEY, list);
}

export function clearChatHistory() {
  setJSON(KEY, []);
}

export function recentMessages(maxCount = 12) {
  const list = getChatHistory();
  return list.slice(Math.max(0, list.length - maxCount));
}
