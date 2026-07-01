/**
 * 浏览器语音识别（中文）
 *
 * 使用 webkitSpeechRecognition：
 * - 持续识别中将转写写入 chatInput
 * - 与原 app.toggleVoiceRecognition 行为一致
 */

import { byId } from '../utils/dom.js';
import { showToast } from './toast.js';
import state from '../app/state.js';

let recognition = null;
let active = false;

function init() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    showToast('当前浏览器不支持语音识别，请使用 Chrome / Edge / Safari', 'error');
    return null;
  }
  const r = new SR();
  r.lang = 'zh-CN';
  r.continuous = false;
  r.interimResults = true;

  r.onstart = () => updateBtn(true);
  r.onend = () => { active = false; updateBtn(false); };
  r.onerror = (ev) => {
    showToast(`语音识别错误：${ev.error || ''}`, 'error');
    active = false;
    updateBtn(false);
  };

  r.onresult = (ev) => {
    const input = byId('chatInput');
    if (!input) return;
    let finalTx = '';
    let interimTx = '';
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const t = ev.results[i][0]?.transcript || '';
      if (ev.results[i].isFinal) finalTx += t;
      else interimTx += t;
    }
    if (finalTx) {
      const old = state.voicePrefix || '';
      input.value = (old + finalTx).slice(0, 4000);
      state.voicePrefix = input.value;
    } else if (interimTx) {
      const old = state.voicePrefix || '';
      input.value = (old + interimTx).slice(0, 4000);
    }
  };

  return r;
}

function updateBtn(isOn) {
  const btn = byId('voiceBtn');
  if (!btn) return;
  if (isOn) {
    btn.classList.add('is-recording');
    btn.title = '停止录音';
  } else {
    btn.classList.remove('is-recording');
    btn.title = '语音输入';
  }
}

export function toggleVoiceRecognition() {
  if (!recognition) recognition = init();
  if (!recognition) return;

  if (active) {
    try { recognition.stop(); } catch (_) { /* ignore */ }
    active = false;
  } else {
    state.voicePrefix = byId('chatInput')?.value || '';
    try {
      recognition.start();
      active = true;
    } catch (e) {
      showToast(`无法启动语音识别：${e.message}`, 'error');
    }
  }
}
