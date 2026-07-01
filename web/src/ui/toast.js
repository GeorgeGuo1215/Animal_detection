/**
 * Toast 通知（替代原 showMessage）
 */

import { h } from '../utils/dom.js';

let stack = null;

function ensureStack() {
  if (stack && document.body.contains(stack)) return stack;
  stack = h('div', { class: 'toast-stack', role: 'status', 'aria-live': 'polite' });
  document.body.appendChild(stack);
  return stack;
}

const TYPE_CLASS = {
  info: 'toast-info',
  success: 'toast-success',
  warn: 'toast-warn',
  warning: 'toast-warn',
  error: 'toast-error',
};

/**
 * 显示一个 toast。
 * @param {string} message
 * @param {'info'|'success'|'warn'|'error'} type
 * @param {{duration?: number}} opts
 */
export function showToast(message, type = 'info', opts = {}) {
  const root = ensureStack();
  const cls = TYPE_CLASS[type] || TYPE_CLASS.info;
  const el = h('div', { class: `toast ${cls}` }, [message]);
  root.appendChild(el);

  const duration = opts.duration ?? (type === 'error' ? 4500 : 2400);
  setTimeout(() => {
    el.classList.add('is-leaving');
    setTimeout(() => el.remove(), 200);
  }, duration);
}

/** 兼容旧 showMessage(msg, type) */
export function showMessage(msg, type = 'info') {
  showToast(msg, type);
}
