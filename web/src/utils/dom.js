/**
 * 轻量 DOM 工具：选择器、文本/数值读写、批量绑定。
 */

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export const byId = (id) => document.getElementById(id);

export function setText(idOrEl, value) {
  const el = typeof idOrEl === 'string' ? byId(idOrEl) : idOrEl;
  if (el) el.textContent = value == null ? '' : String(value);
}

export function setHtml(idOrEl, html) {
  const el = typeof idOrEl === 'string' ? byId(idOrEl) : idOrEl;
  if (el) el.innerHTML = html;
}

export function getNum(idOrEl, fallback = 0) {
  const el = typeof idOrEl === 'string' ? byId(idOrEl) : idOrEl;
  if (!el) return fallback;
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : fallback;
}

export function getStr(idOrEl, fallback = '') {
  const el = typeof idOrEl === 'string' ? byId(idOrEl) : idOrEl;
  if (!el) return fallback;
  return (el.value ?? '').toString();
}

export function show(el) { if (el) el.style.display = ''; }
export function hide(el) { if (el) el.style.display = 'none'; }

export function on(el, event, handler, opts) {
  if (!el) return () => {};
  el.addEventListener(event, handler, opts);
  return () => el.removeEventListener(event, handler, opts);
}

export function delegate(root, selector, event, handler) {
  if (!root) return () => {};
  const wrapped = (e) => {
    const target = e.target.closest(selector);
    if (target && root.contains(target)) handler(e, target);
  };
  root.addEventListener(event, wrapped);
  return () => root.removeEventListener(event, wrapped);
}

/** 等待 DOM ready */
export function ready(fn) {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fn, { once: true });
  } else {
    fn();
  }
}

/** 创建 DOM 元素的简写工厂 */
export function h(tag, attrs = {}, children = []) {
  const el = document.createElement(tag);
  const a = attrs == null ? {} : attrs;
  const kids = children == null ? [] : children;
  for (const [k, v] of Object.entries(a)) {
    if (v == null) continue;
    if (k === 'class' || k === 'className') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'dataset' && typeof v === 'object') Object.assign(el.dataset, v);
    else el.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) {
    if (c == null) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
