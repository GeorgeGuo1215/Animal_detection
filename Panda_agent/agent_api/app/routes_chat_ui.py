"""
Panda-themed chat UI served at /chat — same port as the Agent API.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from shared.chat_feedback_widget import FEEDBACK_WIDGET_JS, render_feedback_widget_css

router = APIRouter()

_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Panda Mind — 大熊猫百科</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#f0f7f0;--card:#fff;--border:#c8e6c9;--accent:#2e7d32;--accent-light:#e8f5e9;
       --text:#1a1a1a;--text2:#6b7280;--think-bg:#fff8e1;--think-border:#ffb300;--radius:14px;
       --shadow-sm:0 1px 3px rgba(0,0,0,.06);--shadow-md:0 4px 16px rgba(0,0,0,.08);--shadow-lg:0 8px 32px rgba(0,0,0,.1)}
body{font-family:Inter,"Noto Sans SC",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column;
     background-image:radial-gradient(circle,rgba(46,125,50,.03) 1px,transparent 1px);background-size:22px 22px}
header{background:linear-gradient(135deg,#2e7d32 0%,#43a047 50%,#66bb6a 100%);border-bottom:none;
       padding:14px 24px;display:flex;align-items:center;gap:12px;flex-shrink:0;color:#fff;
       box-shadow:0 2px 24px rgba(46,125,50,.15);position:relative;z-index:10;
       backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px)}
header h1{font-size:19px;font-weight:700;letter-spacing:-.3px}
header .subtitle{font-size:12px;opacity:.8}
.settings{margin-left:auto;display:flex;gap:10px;align-items:center;font-size:13px}
.role-hint{font-size:12px;opacity:.75;white-space:nowrap}
.role-toggle{display:flex;border-radius:8px;overflow:hidden;border:1px solid rgba(255,255,255,.35)}
.role-toggle button{padding:5px 14px;border:none;background:rgba(255,255,255,.1);color:rgba(255,255,255,.75);
       font-size:13px;cursor:pointer;transition:all .2s}
.role-toggle button.active{background:rgba(255,255,255,.3);color:#fff;font-weight:600}
.role-toggle button:hover:not(.active){background:rgba(255,255,255,.18)}
#chat{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:800px;width:100%;margin:0 auto;display:flex;gap:12px}
.msg.user{flex-direction:row-reverse}
.msg .bubble{padding:12px 16px;border-radius:var(--radius);line-height:1.75;font-size:14px;word-break:break-word;
             transition:box-shadow .2s}
.msg.user .bubble{background:linear-gradient(135deg,#2e7d32,#43a047);color:#fff;border-bottom-right-radius:4px;
                  white-space:pre-wrap;box-shadow:0 2px 12px rgba(46,125,50,.2)}
.msg.assistant .bubble{background:linear-gradient(150deg,#fff 60%,#f6fbf6);border:1px solid var(--border);
                       border-bottom-left-radius:4px;min-width:60px;box-shadow:var(--shadow-sm)}
.msg.assistant .bubble.raw-text{white-space:pre-wrap}
.msg .avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;
             justify-content:center;font-size:17px;flex-shrink:0;box-shadow:0 2px 8px rgba(0,0,0,.1);
             transition:transform .2s}
.msg .avatar:hover{transform:scale(1.1)}
.msg.user .avatar{background:linear-gradient(135deg,#e8f5e9,#a5d6a7);color:var(--accent)}
.msg.assistant .avatar{background:linear-gradient(135deg,#e8f5e9,#81c784)}
/* Markdown rendered content styles */
.msg.assistant .bubble h1,.msg.assistant .bubble h2,.msg.assistant .bubble h3{margin:14px 0 6px;font-weight:600;color:var(--accent)}
.msg.assistant .bubble h1{font-size:18px}.msg.assistant .bubble h2{font-size:16px}.msg.assistant .bubble h3{font-size:15px}
.msg.assistant .bubble p{margin:6px 0}
.msg.assistant .bubble ul,.msg.assistant .bubble ol{margin:6px 0;padding-left:20px}
.msg.assistant .bubble li{margin:3px 0}
.msg.assistant .bubble strong{color:#1b5e20}
.msg.assistant .bubble code{background:#e8f5e9;padding:1px 5px;border-radius:4px;font-size:13px}
.msg.assistant .bubble pre{background:#263238;color:#e0e0e0;padding:12px;border-radius:8px;
       overflow-x:auto;margin:8px 0;font-size:13px}
.msg.assistant .bubble pre code{background:none;color:inherit;padding:0}
.msg.assistant .bubble blockquote{border-left:3px solid var(--accent);padding:8px 14px;margin:10px 0;
       background:var(--accent-light);border-radius:0 8px 8px 0;font-size:13px;color:#2e7d32}
.msg.assistant .bubble table{border-collapse:collapse;margin:8px 0;width:100%}
.msg.assistant .bubble th,.msg.assistant .bubble td{border:1px solid var(--border);padding:6px 10px;font-size:13px}
.msg.assistant .bubble th{background:var(--accent-light);font-weight:600}
.msg.assistant .bubble a{color:var(--accent);text-decoration:underline;transition:opacity .2s}
.msg.assistant .bubble a:hover{opacity:.7}
.msg.assistant .bubble a::after{content:" ↗";font-size:11px;opacity:.6}
.think-box{max-width:800px;width:100%;margin:8px auto;background:linear-gradient(135deg,#fff8e1,#fffde7);
           border:1px solid var(--think-border);border-radius:10px;overflow:hidden;
           transition:box-shadow .3s}
.think-box:hover{box-shadow:0 2px 12px rgba(255,179,0,.12)}
.think-header{padding:9px 14px;cursor:pointer;font-size:13px;color:#e65100;
              display:flex;align-items:center;gap:6px;user-select:none;transition:background .2s}
.think-header:hover{background:rgba(255,179,0,.08)}
.think-header::before{content:"";display:inline-block;width:0;height:0;
       border-left:5px solid #e65100;border-top:4px solid transparent;border-bottom:4px solid transparent;
       transition:transform .25s cubic-bezier(.4,0,.2,1)}
.think-box.open .think-header::before{transform:rotate(90deg)}
.think-body{display:none;padding:6px 14px 10px;font-size:12px;color:#78716c;
            font-family:"SF Mono",Consolas,monospace;line-height:1.5;max-height:300px;overflow-y:auto}
.think-box.open .think-body{display:block}
.think-line{padding:2px 0;animation:fadeSlideIn .25s ease both}
.think-line.tool{color:#e65100}
.think-line.status{color:#1565c0}
.think-line.hits{color:#2e7d32}
#input-area{flex-shrink:0;background:var(--card);border-top:none;padding:14px 20px;
            box-shadow:0 -4px 24px rgba(0,0,0,.05);position:relative;z-index:5}
#input-wrap{max-width:800px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
#input-wrap textarea{flex:1;resize:none;border:1.5px solid var(--border);border-radius:var(--radius);
       padding:10px 16px;font-size:14px;font-family:inherit;line-height:1.5;min-height:46px;max-height:120px;
       outline:none;transition:border-color .2s,box-shadow .2s;background:#fafff9}
#input-wrap textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(46,125,50,.1)}
#send-btn{padding:0 22px;height:46px;background:linear-gradient(135deg,#2e7d32,#43a047);color:#fff;border:none;
          border-radius:var(--radius);font-size:14px;font-weight:600;cursor:pointer;
          transition:all .2s;white-space:nowrap;display:inline-flex;align-items:center;gap:6px;
          box-shadow:0 2px 8px rgba(46,125,50,.2)}
#send-btn:disabled{opacity:.4;cursor:not-allowed;box-shadow:none}
#send-btn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 4px 16px rgba(46,125,50,.3)}
#send-btn:active:not(:disabled){transform:translateY(0)}
.timing{font-size:11px;color:var(--text2);text-align:center;padding:2px 0}
.welcome{max-width:800px;margin:32px auto;text-align:center;color:var(--text2);animation:fadeSlideIn .5s ease both}
.welcome .panda-icon{font-size:64px;margin-bottom:12px;animation:floatBounce 3s ease-in-out infinite;display:inline-block}
.welcome h2{color:var(--accent);margin-bottom:8px;font-size:22px;letter-spacing:-.3px}
.welcome p{font-size:14px;line-height:1.8}
/* Role picker cards */
.role-picker{display:flex;gap:16px;justify-content:center;margin:20px 0 4px;flex-wrap:wrap}
.role-card{flex:1;min-width:180px;max-width:240px;background:#fff;border:2.5px solid var(--border);
           border-radius:18px;padding:22px 18px 20px;cursor:pointer;transition:all .25s cubic-bezier(.4,0,.2,1);
           text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.05)}
.role-card:hover{border-color:var(--accent);box-shadow:0 6px 24px rgba(46,125,50,.15);transform:translateY(-4px)}
.role-card.selected{border-color:var(--accent);background:var(--accent-light);
                    box-shadow:0 4px 20px rgba(46,125,50,.2)}
.role-card .rc-icon{font-size:38px;margin-bottom:8px;display:inline-block;transition:transform .3s}
.role-card:hover .rc-icon{transform:scale(1.15) rotate(-5deg)}
.role-card .rc-name{font-size:17px;font-weight:700;color:var(--accent);margin-bottom:4px}
.role-card .rc-desc{font-size:12px;color:var(--text2);line-height:1.5}
.role-card.selected .rc-desc{color:#388e3c}
.welcome .examples{margin-top:18px;display:flex;flex-wrap:wrap;gap:10px;justify-content:center}
.welcome .examples button{background:var(--accent-light);border:1px solid var(--border);border-radius:22px;
       padding:8px 18px;font-size:13px;color:var(--accent);cursor:pointer;transition:all .25s cubic-bezier(.4,0,.2,1);
       box-shadow:0 1px 4px rgba(0,0,0,.04)}
.welcome .examples button:hover{background:var(--accent);color:#fff;transform:translateY(-2px);
       box-shadow:0 4px 16px rgba(46,125,50,.2)}
__FEEDBACK_WIDGET_CSS__
/* Keyframe animations */
@keyframes fadeSlideIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes floatBounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes pulse3{0%,80%,100%{opacity:.3}40%{opacity:1}}
@keyframes spin{to{transform:rotate(360deg)}}
/* Scrollbar */
html{scroll-behavior:smooth}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(46,125,50,.2);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(46,125,50,.35)}
/* Message entrance */
.msg{animation:fadeSlideIn .35s ease both}
.timing{animation:fadeSlideIn .3s ease both}
@media(max-width:600px){
header{padding:10px 14px;gap:8px}
header h1{font-size:16px}
.role-hint{display:none}
.role-toggle button{padding:4px 10px;font-size:12px}
.role-picker{gap:10px}
.role-card{min-width:140px;padding:16px 12px 14px}
.role-card .rc-icon{font-size:30px}
.role-card .rc-name{font-size:15px}
#input-wrap textarea{font-size:16px}
}
</style>
</head>
<body>

<header>
  <h1>Panda Mind</h1>
  <span class="subtitle">大熊猫百科知识问答</span>
  <div class="settings">
    <span class="role-hint" id="header-role-label">身份：🐼 爱好者</span>
    <div class="role-toggle">
      <button id="role-enthusiast" class="active" onclick="setRole('enthusiast')">🐼 爱好者</button>
      <button id="role-researcher" onclick="setRole('researcher')">🔬 科研人员</button>
    </div>
  </div>
</header>

<div id="chat">
  <div class="welcome" id="welcome-panel">
    <div class="panda-icon">🐼</div>
    <h2>欢迎使用 Panda Mind</h2>
    <p>我是大熊猫百科 AI 助手，知识来自专业书籍和学术论文。<br>请先选择你的身份，获取最适合你的回答风格 👇</p>
    <div class="role-picker">
      <div class="role-card selected" id="card-enthusiast" onclick="setRole('enthusiast')">
        <div class="rc-icon">🐼</div>
        <div class="rc-name">爱好者</div>
        <div class="rc-desc">通俗易懂、生动有趣<br>适合科普爱好者和普通大众</div>
      </div>
      <div class="role-card" id="card-researcher" onclick="setRole('researcher')">
        <div class="rc-icon">🔬</div>
        <div class="rc-name">科研人员</div>
        <div class="rc-desc">专业详尽、引用文献<br>适合学者、兽医和保护工作者</div>
      </div>
    </div>
    <div class="examples" id="example-questions"></div>
  </div>
</div>

<div id="input-area">
  <div id="input-wrap">
    <textarea id="user-input" rows="1" placeholder="输入你的问题..." autofocus></textarea>
    <button id="send-btn">发送</button>
  </div>
</div>

<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

const EXAMPLES = {
  enthusiast: [
    '大熊猫一天能吃多少竹子？',
    '大熊猫为什么是黑白色的？',
    '大熊猫的寿命有多长？',
    '野生大熊猫还剩多少只？',
  ],
  researcher: [
    '大熊猫 MHC 基因多态性研究进展',
    '圈养大熊猫繁殖中的人工授精技术',
    '大熊猫犬瘟热疫苗免疫方案',
    '秦岭与四川大熊猫种群遗传分化',
  ],
};

let currentRole = 'enthusiast';
const messages = [];
let streaming = false;
let welcomeShown = true;

function setRole(role) {
  currentRole = role;
  // Header toggle buttons
  document.getElementById('role-enthusiast').classList.toggle('active', role === 'enthusiast');
  document.getElementById('role-researcher').classList.toggle('active', role === 'researcher');
  // Welcome page cards
  const ce = document.getElementById('card-enthusiast');
  const cr = document.getElementById('card-researcher');
  if (ce) ce.classList.toggle('selected', role === 'enthusiast');
  if (cr) cr.classList.toggle('selected', role === 'researcher');
  // Header label
  const lbl = document.getElementById('header-role-label');
  if (lbl) lbl.textContent = role === 'researcher' ? '身份：🔬 科研人员' : '身份：🐼 爱好者';
  renderExamples();
  inputEl.placeholder = role === 'researcher'
    ? '输入科研问题，如：大熊猫种群遗传多样性分析...'
    : '输入你的问题，如：大熊猫为什么爱吃竹子？';
}

function renderExamples() {
  const container = document.getElementById('example-questions');
  if (!container) return;
  container.innerHTML = '';
  (EXAMPLES[currentRole] || []).forEach(q => {
    const btn = document.createElement('button');
    btn.textContent = q;
    btn.onclick = () => { inputEl.value = q; send(); };
    container.appendChild(btn);
  });
}
renderExamples();

const renderer = new marked.Renderer();
const origLinkRenderer = renderer.link.bind(renderer);
renderer.link = function(href, title, text) {
  const html = origLinkRenderer(href, title, text);
  return html.replace('<a ', '<a target="_blank" rel="noopener noreferrer" ');
};
marked.setOptions({ breaks: true, gfm: true, renderer });

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
sendBtn.addEventListener('click', send);

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function addUserMsg(text) {
  if (welcomeShown) {
    const w = document.getElementById('welcome-panel');
    if (w) w.remove();
    welcomeShown = false;
  }
  chatEl.insertAdjacentHTML('beforeend',
    `<div class="msg user"><div class="avatar">👤</div><div class="bubble">${esc(text)}</div></div>`);
  scroll();
}

function createAssistantMsg() {
  const id = 'msg-' + Date.now();
  chatEl.insertAdjacentHTML('beforeend',
    `<div class="think-box open" id="${id}-think">
       <div class="think-header" onclick="this.parentElement.classList.toggle('open')">思考过程</div>
       <div class="think-body"></div>
     </div>
     <div class="msg assistant" id="${id}">
       <div class="avatar">🐼</div>
       <div class="bubble raw-text"></div>
     </div>`);
  scroll();
  return id;
}

function appendThink(id, text, cls = '') {
  const body = document.querySelector(`#${id}-think .think-body`);
  if (!body) return;
  body.insertAdjacentHTML('beforeend', `<div class="think-line ${cls}">${esc(text)}</div>`);
  body.scrollTop = body.scrollHeight;
}

let _mdTimer = null;
let _mdPending = null;

function _flushMd() {
  _mdTimer = null;
  if (!_mdPending) return;
  const { id, text } = _mdPending;
  _mdPending = null;
  const bubble = document.querySelector(`#${id} .bubble`);
  if (!bubble) return;
  bubble.classList.remove('raw-text');
  try { bubble.innerHTML = marked.parse(text); } catch(e) { bubble.textContent = text; }
  scroll();
}

function streamRender(id, fullText) {
  _mdPending = { id, text: fullText };
  if (!_mdTimer) _mdTimer = setTimeout(_flushMd, 80);
}

function renderMarkdown(id, fullText) {
  clearTimeout(_mdTimer); _mdTimer = null; _mdPending = null;
  const bubble = document.querySelector(`#${id} .bubble`);
  if (!bubble) return;
  bubble.classList.remove('raw-text');
  try { bubble.innerHTML = marked.parse(fullText); } catch(e) { bubble.textContent = fullText; }
  scroll();
}

function addTiming(text) {
  chatEl.insertAdjacentHTML('beforeend', `<div class="timing">${esc(text)}</div>`);
}

function scroll() { chatEl.scrollTop = chatEl.scrollHeight; }

async function send() {
  const text = inputEl.value.trim();
  if (!text || streaming) return;
  streaming = true;
  sendBtn.disabled = true;
  inputEl.value = '';
  inputEl.style.height = 'auto';

  addUserMsg(text);
  messages.push({ role: 'user', content: text });
  const msgId = createAssistantMsg();
  let fullContent = '';
  let capturedRequestId = '';

  const headers = { 'Content-Type': 'application/json' };

  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST', headers,
      body: JSON.stringify({
        model: 'panda-plan-solve',
        messages: messages.slice(-12),
        stream: true, temperature: 0.7, max_tokens: 2048,
        debug_timing: true,
        user_role: currentRole
      })
    });

    if (!resp.ok) {
      const err = await resp.text();
      const eb = document.querySelector(`#${msgId} .bubble`); if (eb) { eb.classList.remove('raw-text'); eb.textContent = `Error ${resp.status}: ${err}`; }
      streaming = false; sendBtn.disabled = false; return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;
        const payload = trimmed.slice(6);
        if (payload === '[DONE]') continue;
        try {
          const chunk = JSON.parse(payload);
          if (chunk.id) capturedRequestId = chunk.id;
          const delta = chunk.choices?.[0]?.delta;
          const status = chunk.agent_status;
          const detail = chunk.agent_detail || {};
          const finish = chunk.choices?.[0]?.finish_reason;

          if (status === 'planning') appendThink(msgId, '📋 ' + (detail.message || '制定计划中...'), 'status');
          else if (status === 'thinking') appendThink(msgId, '🤔 ' + (detail.message || '思考中...'), 'status');
          else if (status === 'plan_complete') {
            const steps = (detail.plan || []).filter(s => s.type === 'tool');
            appendThink(msgId, '✅ 计划完成: ' + steps.map(s => s.tool_name).join(' → '), 'status');
          }
          else if (status === 'tool_calling') {
            appendThink(msgId, '🔧 调用工具: ' + (detail.tool_name || ''), 'tool');
            if (detail.reason) appendThink(msgId, '   原因: ' + detail.reason, '');
          }
          else if (status === 'tool_complete') appendThink(msgId, '✅ 返回 ' + (detail.hits_count ?? 0) + ' 条结果', 'hits');
          else if (status === 'decided_final') appendThink(msgId, '💡 ' + (detail.reason || '准备生成回答'), 'status');
          else if (status === 'generating') {
            appendThink(msgId, '📝 生成回答...', 'status');
            const thinkBox = document.getElementById(msgId + '-think');
            if (thinkBox) thinkBox.classList.remove('open');
          }
          else if (status === 'timing_summary') {
            const t = detail.timing || [];
            const total = t.find(x => x.step === 'total');
            if (total) addTiming(`耗时 ${(total.ms / 1000).toFixed(1)}s`);
          }

          if (status === 'streaming' && delta?.content) {
            fullContent += delta.content;
            streamRender(msgId, fullContent);
          }
          if (finish === 'stop') {
            if (fullContent) renderMarkdown(msgId, fullContent);
            break;
          }
        } catch {}
      }
    }
  } catch (e) {
    const nb = document.querySelector(`#${msgId} .bubble`); if (nb) { nb.classList.remove('raw-text'); nb.textContent = 'Network error: ' + e.message; }
  }

  if (fullContent) {
    renderMarkdown(msgId, fullContent);
    messages.push({ role: 'assistant', content: fullContent });
    if (capturedRequestId) showFeedback(msgId, capturedRequestId);
  }
  streaming = false;
  sendBtn.disabled = false;
  inputEl.focus();
}

__FEEDBACK_WIDGET_JS__
</script>
</body>
</html>"""

_CHAT_HTML = _CHAT_HTML.replace("__FEEDBACK_WIDGET_CSS__", render_feedback_widget_css("panda"))
_CHAT_HTML = _CHAT_HTML.replace("__FEEDBACK_WIDGET_JS__", FEEDBACK_WIDGET_JS)


@router.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return _CHAT_HTML


_ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Panda Mind — 管理后台</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#f0f7f0;--card:#fff;--border:#c8e6c9;--accent:#2e7d32;--accent-light:#e8f5e9;
      --text:#1a1a1a;--text2:#6b7280;--radius:10px;--danger:#c62828;
      --shadow-sm:0 1px 3px rgba(0,0,0,.06);--shadow-md:0 4px 16px rgba(0,0,0,.08)}
body{font-family:Inter,"Noto Sans SC",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:var(--bg);color:var(--text);min-height:100vh;
     background-image:radial-gradient(circle,rgba(46,125,50,.025) 1px,transparent 1px);background-size:20px 20px}
header{background:linear-gradient(135deg,#1b5e20 0%,#2e7d32 50%,#43a047 100%);padding:16px 24px;
       display:flex;align-items:center;gap:12px;color:#fff;
       box-shadow:0 2px 24px rgba(27,94,32,.15);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
       position:sticky;top:0;z-index:50}
header h1{font-size:19px;font-weight:700;letter-spacing:-.3px}
header .sub{font-size:12px;opacity:.75;margin-left:4px}
header .logout{margin-left:auto;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.25);
               color:#fff;padding:6px 16px;border-radius:8px;cursor:pointer;font-size:13px;
               transition:all .25s;font-weight:500}
header .logout:hover{background:rgba(255,255,255,.25);transform:translateY(-1px);
                     box-shadow:0 2px 8px rgba(0,0,0,.15)}

/* Login overlay */
#login-overlay{position:fixed;inset:0;background:rgba(0,0,0,.35);display:flex;
               align-items:center;justify-content:center;z-index:999;
               backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.login-box{background:#fff;border-radius:16px;padding:40px 34px;width:360px;
           box-shadow:0 16px 48px rgba(0,0,0,.16);animation:scaleIn .35s ease both}
.login-box h2{font-size:22px;color:var(--accent);margin-bottom:6px;letter-spacing:-.3px}
.login-box p{font-size:13px;color:var(--text2);margin-bottom:22px;line-height:1.5}
.login-box input{width:100%;padding:11px 16px;border:1.5px solid var(--border);
                 border-radius:10px;font-size:14px;outline:none;margin-bottom:14px;
                 transition:border-color .2s,box-shadow .2s}
.login-box input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(46,125,50,.1)}
.login-box button{width:100%;padding:11px;background:linear-gradient(135deg,#2e7d32,#43a047);color:#fff;
                  border:none;border-radius:10px;font-size:15px;cursor:pointer;font-weight:600;
                  transition:all .2s;box-shadow:0 2px 8px rgba(46,125,50,.2)}
.login-box button:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(46,125,50,.3)}
.login-box .err{color:var(--danger);font-size:13px;margin-top:8px;min-height:18px}

/* Tabs */
.tabs{display:flex;gap:0;border-bottom:2px solid var(--border);background:#fff;
      padding:0 24px;box-shadow:0 1px 4px rgba(0,0,0,.03)}
.tab-btn{padding:13px 24px;border:none;background:none;font-size:14px;
         color:var(--text2);cursor:pointer;border-bottom:3px solid transparent;
         margin-bottom:-2px;font-weight:500;transition:all .25s;border-radius:8px 8px 0 0;
         position:relative}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent);font-weight:700}
.tab-btn:hover:not(.active){color:var(--text);background:var(--accent-light);transform:translateY(-1px)}
.tab-content{display:none;padding:24px}
.tab-content.active{display:block;animation:fadeSlideIn .3s ease both}

/* Cards */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px}
.stat-card{background:#fff;border:1px solid var(--border);border-radius:var(--radius);
           padding:22px 20px;text-align:center;transition:all .25s cubic-bezier(.4,0,.2,1);
           border-left:3px solid transparent;position:relative;overflow:hidden}
.stat-card:hover{transform:translateY(-4px);box-shadow:0 8px 28px rgba(0,0,0,.08);border-left-color:var(--accent)}
.stat-card .val{font-size:32px;font-weight:700;color:var(--accent);font-variant-numeric:tabular-nums;
                transition:color .2s}
.stat-card:hover .val{color:#1b5e20}
.stat-card .lbl{font-size:12px;color:var(--text2);margin-top:6px;font-weight:500}

/* Tables */
.section{background:#fff;border:1px solid var(--border);border-radius:var(--radius);
         margin-bottom:20px;overflow:hidden;transition:box-shadow .3s}
.section:hover{box-shadow:0 4px 20px rgba(0,0,0,.05)}
.section-title{padding:14px 18px;font-size:14px;font-weight:700;color:var(--accent);
               border-bottom:1px solid var(--border);background:linear-gradient(135deg,var(--accent-light),#f1f8e9)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{padding:11px 14px;text-align:left;font-weight:600;color:var(--text2);
   border-bottom:1px solid var(--border);background:linear-gradient(180deg,#fafafa,#f5f5f5);font-size:12px;
   text-transform:uppercase;letter-spacing:.3px}
td{padding:11px 14px;border-bottom:1px solid #f0f0f0;vertical-align:top;transition:all .15s}
tr:last-child td{border-bottom:none}
tr:nth-child(even):not(.detail-row) td{background:rgba(232,245,233,.3)}
tr.expandable{cursor:pointer}
tr.expandable:hover td{background:var(--accent-light);border-left:3px solid var(--accent);padding-left:11px}
tr.detail-row td{padding:0}
.detail-inner{display:none;padding:14px 18px;background:linear-gradient(135deg,#f9fbe7,#fffde7);font-size:13px;
              line-height:1.7;white-space:pre-wrap;border-top:1px solid var(--border)}
.detail-inner.open{display:block}

/* Filters */
.filters{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px;align-items:center}
.filters input,.filters select{padding:8px 14px;border:1.5px solid var(--border);
                               border-radius:8px;font-size:13px;outline:none;
                               transition:border-color .2s,box-shadow .2s}
.filters input:focus,.filters select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(46,125,50,.08)}
.btn{padding:8px 20px;background:linear-gradient(135deg,#2e7d32,#43a047);color:#fff;border:none;border-radius:8px;
     font-size:13px;cursor:pointer;font-weight:600;transition:all .2s;box-shadow:0 2px 6px rgba(46,125,50,.15)}
.btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(46,125,50,.25)}
.btn:active{transform:translateY(0)}
.btn-sm{padding:4px 12px;font-size:12px;border-radius:5px}

/* Pagination */
.pagination{display:flex;gap:8px;justify-content:center;margin-top:18px}
.page-btn{padding:6px 16px;border:1.5px solid var(--border);background:#fff;
          border-radius:8px;cursor:pointer;font-size:13px;transition:all .2s;font-weight:500}
.page-btn.active{background:linear-gradient(135deg,#2e7d32,#43a047);color:#fff;border-color:var(--accent);
                 box-shadow:0 2px 8px rgba(46,125,50,.2)}
.page-btn:hover:not(.active){border-color:var(--accent);color:var(--accent);transform:translateY(-1px);
                              box-shadow:0 2px 8px rgba(0,0,0,.06)}

/* Gap list */
.gap-item{padding:14px 18px;border-bottom:1px solid var(--border);display:flex;
          align-items:flex-start;gap:14px;transition:all .2s}
.gap-item:hover{background:var(--accent-light);padding-left:22px}
.gap-item:last-child{border-bottom:none}
.gap-q{flex:1;font-size:14px;line-height:1.5}
.gap-meta{font-size:12px;color:var(--text2);white-space:nowrap;text-align:right}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;font-size:11px;font-weight:600;
       transition:transform .2s}
.badge:hover{transform:scale(1.05)}
.badge-web{background:#e3f2fd;color:#1565c0}
.badge-norag{background:#fce4ec;color:#c62828}
.count-badge{background:var(--accent-light);color:var(--accent);padding:2px 10px;
             border-radius:10px;font-size:12px;font-weight:700;transition:transform .2s}
.count-badge:hover{transform:scale(1.05)}

.loading{color:var(--text2);font-size:14px;padding:24px;text-align:center}
.loading::before{content:'';display:inline-block;width:16px;height:16px;border:2px solid var(--border);
                 border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;
                 margin-right:8px;vertical-align:middle}
.empty{color:var(--text2);font-size:14px;padding:40px;text-align:center;animation:fadeSlideIn .4s ease both}
.empty::before{content:'📭';display:block;font-size:36px;margin-bottom:10px}

/* Keyframe animations */
@keyframes fadeSlideIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes scaleIn{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
@keyframes spin{to{transform:rotate(360deg)}}
/* Scrollbar */
html{scroll-behavior:smooth}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(46,125,50,.2);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(46,125,50,.35)}
</style>
</head>
<body>

<!-- Login overlay -->
<div id="login-overlay">
  <div class="login-box">
    <h2>🐼 管理员登录</h2>
    <p>请输入 Admin Token 以访问管理后台</p>
    <input type="password" id="token-input" placeholder="X-Admin-Token" autocomplete="off">
    <button onclick="doLogin()">登录</button>
    <div class="err" id="login-err"></div>
  </div>
</div>

<header>
  <span style="font-size:24px">🐼</span>
  <div>
    <h1>Panda Mind 管理后台</h1>
    <span class="sub">问答记录 · 访问统计 · 知识盲区</span>
  </div>
  <button class="logout" onclick="doLogout()">退出登录</button>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('stats',this)">访问统计</button>
  <button class="tab-btn" onclick="switchTab('history',this)">问答历史</button>
  <button class="tab-btn" onclick="switchTab('gaps',this)">知识盲区</button>
  <button class="tab-btn" onclick="switchTab('feedback',this)">用户反馈</button>
</div>

<!-- Tab 1: Stats -->
<div id="tab-stats" class="tab-content active">
  <div class="stat-grid" id="stat-cards">
    <div class="loading">加载中...</div>
  </div>
  <div class="section">
    <div class="section-title">每日访问量（近 30 天）</div>
    <table>
      <thead><tr><th>日期</th><th>问答次数</th></tr></thead>
      <tbody id="daily-tbody"><tr><td colspan="2" class="loading">加载中...</td></tr></tbody>
    </table>
  </div>
  <div class="section">
    <div class="section-title">模型使用分布</div>
    <table>
      <thead><tr><th>模型</th><th>使用次数</th></tr></thead>
      <tbody id="model-tbody"><tr><td colspan="2" class="loading">加载中...</td></tr></tbody>
    </table>
  </div>
</div>

<!-- Tab 2: History -->
<div id="tab-history" class="tab-content">
  <div class="filters">
    <input type="text" id="kw-input" placeholder="关键词搜索" style="width:200px">
    <input type="date" id="date-from" title="起始日期">
    <span style="color:var(--text2);font-size:13px">至</span>
    <input type="date" id="date-to" title="结束日期">
    <button class="btn btn-sm" onclick="loadHistory(1)">搜索</button>
    <button class="btn btn-sm" style="background:#757575" onclick="clearFilters()">清空</button>
    <span id="hist-total" style="font-size:13px;color:var(--text2)"></span>
  </div>
  <div class="section">
    <div class="section-title">问答记录</div>
    <table>
      <thead>
        <tr>
          <th style="width:130px">时间</th>
          <th>问题</th>
          <th style="width:120px">工具</th>
          <th style="width:70px">RAG命中</th>
          <th style="width:60px">评分</th>
          <th style="width:80px">耗时</th>
        </tr>
      </thead>
      <tbody id="hist-tbody"><tr><td colspan="6" class="loading">加载中...</td></tr></tbody>
    </table>
  </div>
  <div class="pagination" id="pagination"></div>
</div>

<!-- Tab 3: Gaps -->
<div id="tab-gaps" class="tab-content">
  <div style="background:#fff3e0;border:1px solid #ffe0b2;border-radius:8px;padding:12px 16px;
              margin-bottom:18px;font-size:13px;color:#e65100">
    以下问题在本地知识库中 RAG 命中为 0，建议将相关知识补充进语料库。
  </div>
  <div class="section" id="gaps-section">
    <div class="section-title">知识盲区问题列表</div>
    <div id="gaps-list"><div class="loading">加载中...</div></div>
  </div>
</div>

<!-- Tab 4: Feedback -->
<div id="tab-feedback" class="tab-content">
  <div class="stat-grid" id="fb-stat-cards"><div class="loading">加载中...</div></div>
  <div class="filters">
    <select id="fb-filter">
      <option value="all">全部已评分</option>
      <option value="low">低分（1-2 星）</option>
      <option value="high">高分（4-5 星）</option>
    </select>
    <button class="btn btn-sm" onclick="loadFeedbackTab()">筛选</button>
  </div>
  <div class="section">
    <div class="section-title">用户反馈列表</div>
    <table>
      <thead><tr><th style="width:60px">评分</th><th>问题摘要</th><th>评论</th><th style="width:140px">时间</th></tr></thead>
      <tbody id="fb-tbody"><tr><td colspan="4" class="loading">加载中...</td></tr></tbody>
    </table>
  </div>
</div>

<script>
const API = '';  // same origin

let _token = sessionStorage.getItem('admin_token') || '';
let _histPage = 1;
const PAGE_SIZE = 20;

function hdr() {
  return { 'X-Admin-Token': _token };
}

async function apiFetch(url) {
  const r = await fetch(API + url, { headers: hdr() });
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

function doLogin() {
  const v = document.getElementById('token-input').value.trim();
  if (!v) return;
  _token = v;
  // Test the token
  fetch(API + '/qa/stats', { headers: { 'X-Admin-Token': v } }).then(r => {
    if (r.ok) {
      sessionStorage.setItem('admin_token', v);
      document.getElementById('login-overlay').style.display = 'none';
      loadAll();
    } else {
      document.getElementById('login-err').textContent = 'Token 无效，请重试';
      _token = '';
    }
  }).catch(() => {
    document.getElementById('login-err').textContent = '网络错误，请检查服务是否运行';
  });
}

document.getElementById('token-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') doLogin();
});

function doLogout() {
  sessionStorage.removeItem('admin_token');
  _token = '';
  document.getElementById('login-overlay').style.display = 'flex';
  document.getElementById('token-input').value = '';
  document.getElementById('login-err').textContent = '';
}

function switchTab(name, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'stats') loadStats();
  if (name === 'history') loadHistory(1);
  if (name === 'gaps') loadGaps();
  if (name === 'feedback') loadFeedbackTab();
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function fmt_ms(ms) {
  if (ms < 1000) return ms + 'ms';
  return (ms/1000).toFixed(1) + 's';
}

function starsHtml(n) {
  let s = '';
  for (let i = 1; i <= 5; i++) s += `<span style="color:${i <= n ? '#fbbf24' : '#d1d5db'};font-size:14px">&#9733;</span>`;
  return s;
}

// ---- Stats ----
async function loadStats() {
  try {
    const d = await apiFetch('/qa/stats');
    let fbData = {};
    try { fbData = await apiFetch('/qa/feedback-stats'); } catch {}
    const today = new Date().toISOString().slice(0,10);
    const todayRow = (d.daily || []).find(r => r.date_key === today);
    const todayCount = todayRow ? todayRow.count : 0;
    const avgR = fbData.avg_rating || 0;
    const ratedPct = d.total > 0 ? Math.round((fbData.rated_count || 0) / d.total * 100) : 0;

    document.getElementById('stat-cards').innerHTML = `
      <div class="stat-card"><div class="val">${d.total}</div><div class="lbl">累计问答总数</div></div>
      <div class="stat-card"><div class="val">${todayCount}</div><div class="lbl">今日问答次数</div></div>
      <div class="stat-card"><div class="val">${fmt_ms(Math.round(d.avg_response_ms))}</div><div class="lbl">平均响应耗时</div></div>
      <div class="stat-card"><div class="val">${d.web_search_count}</div><div class="lbl">网络搜索次数</div></div>
      <div class="stat-card"><div class="val">${avgR.toFixed(1)} ${starsHtml(Math.round(avgR))}</div><div class="lbl">平均用户评分</div></div>
      <div class="stat-card"><div class="val">${ratedPct}%</div><div class="lbl">已评分比例 (${fbData.rated_count||0}/${d.total})</div></div>
    `;

    const daily = d.daily || [];
    document.getElementById('daily-tbody').innerHTML = daily.length
      ? daily.map(r => `<tr><td>${esc(r.date_key)}</td><td><span class="count-badge">${r.count}</span></td></tr>`).join('')
      : '<tr><td colspan="2" class="empty">暂无数据</td></tr>';

    const models = d.model_distribution || [];
    document.getElementById('model-tbody').innerHTML = models.length
      ? models.map(r => `<tr><td>${esc(r.model)}</td><td><span class="count-badge">${r.count}</span></td></tr>`).join('')
      : '<tr><td colspan="2" class="empty">暂无数据</td></tr>';
  } catch(e) {
    document.getElementById('stat-cards').innerHTML = `<div class="empty">加载失败: ${e.message}</div>`;
  }
}

// ---- History ----
async function loadHistory(page) {
  _histPage = page;
  const kw = document.getElementById('kw-input').value.trim();
  const df = document.getElementById('date-from').value;
  const dt = document.getElementById('date-to').value;
  let url = `/qa/history?page=${page}&page_size=${PAGE_SIZE}`;
  if (kw) url += '&keyword=' + encodeURIComponent(kw);
  if (df) url += '&date_from=' + df;
  if (dt) url += '&date_to=' + dt;

  document.getElementById('hist-tbody').innerHTML = '<tr><td colspan="6" class="loading">加载中...</td></tr>';
  try {
    const d = await apiFetch(url);
    document.getElementById('hist-total').textContent = `共 ${d.total} 条`;
    const rows = d.records || [];
    if (!rows.length) {
      document.getElementById('hist-tbody').innerHTML = '<tr><td colspan="6" class="empty">暂无记录</td></tr>';
      document.getElementById('pagination').innerHTML = '';
      return;
    }
    document.getElementById('hist-tbody').innerHTML = rows.map((r,i) => {
      const tools = (r.tools_used || []).join(', ') || '—';
      const shortQ = esc(r.question.length > 60 ? r.question.slice(0,60)+'…' : r.question);
      const tsShort = r.ts.replace('T',' ');
      const ratingHtml = r.feedback_rating > 0 ? starsHtml(r.feedback_rating) : '<span style="color:var(--text2)">—</span>';
      const fbLine = r.feedback_comment ? `\n<strong>用户评论：</strong>${esc(r.feedback_comment)}` : '';
      return `
        <tr class="expandable" onclick="toggleDetail(${i})">
          <td style="font-size:12px;color:var(--text2)">${tsShort}</td>
          <td>${shortQ}</td>
          <td style="font-size:12px">${esc(tools)}</td>
          <td style="text-align:center">${r.rag_hit_count}</td>
          <td style="text-align:center">${ratingHtml}</td>
          <td style="font-size:12px">${fmt_ms(r.response_time_ms)}</td>
        </tr>
        <tr class="detail-row">
          <td colspan="6">
            <div class="detail-inner" id="detail-${i}">
<strong>问题：</strong>${esc(r.question)}

<strong>回答：</strong>
${esc(r.answer)}

<strong>模型：</strong>${esc(r.model)} | <strong>来源IP：</strong>${esc(r.source_ip)} | <strong>角色：</strong>${esc(r.user_role)}${fbLine}</div>
          </td>
        </tr>`;
    }).join('');

    // Pagination
    const totalPages = Math.ceil(d.total / PAGE_SIZE);
    let pages = '';
    for (let p = Math.max(1, page-2); p <= Math.min(totalPages, page+2); p++) {
      pages += `<button class="page-btn${p===page?' active':''}" onclick="loadHistory(${p})">${p}</button>`;
    }
    if (page < totalPages) pages += `<button class="page-btn" onclick="loadHistory(${page+1})">下一页 »</button>`;
    document.getElementById('pagination').innerHTML = pages;
  } catch(e) {
    document.getElementById('hist-tbody').innerHTML = `<tr><td colspan="6" class="empty">加载失败: ${e.message}</td></tr>`;
  }
}

function toggleDetail(i) {
  const el = document.getElementById('detail-' + i);
  el.classList.toggle('open');
}

function clearFilters() {
  document.getElementById('kw-input').value = '';
  document.getElementById('date-from').value = '';
  document.getElementById('date-to').value = '';
  loadHistory(1);
}

// ---- Gaps ----
async function loadGaps() {
  document.getElementById('gaps-list').innerHTML = '<div class="loading">加载中...</div>';
  try {
    const d = await apiFetch('/qa/knowledge-gaps?limit=100');
    const gaps = d.gaps || [];
    if (!gaps.length) {
      document.getElementById('gaps-list').innerHTML = '<div class="empty">暂无知识盲区记录，本地知识库覆盖良好！</div>';
      return;
    }
    document.getElementById('gaps-list').innerHTML = gaps.map(g => `
      <div class="gap-item">
        <div class="gap-q">${esc(g.question)}</div>
        <div class="gap-meta">
          <span class="count-badge">×${g.occurrences}</span><br>
          <span style="font-size:11px;color:var(--text2)">${g.last_asked.replace('T',' ')}</span><br>
          ${g.ever_web_searched ? '<span class="badge badge-web">用过网搜</span>' : '<span class="badge badge-norag">纯盲区</span>'}
        </div>
      </div>`).join('');
  } catch(e) {
    document.getElementById('gaps-list').innerHTML = `<div class="empty">加载失败: ${e.message}</div>`;
  }
}

// ---- Feedback tab ----
async function loadFeedbackTab() {
  try {
    const d = await apiFetch('/qa/feedback-stats');
    const avgR = d.avg_rating || 0;
    const ratedPct = d.total_answers > 0 ? Math.round(d.rated_count / d.total_answers * 100) : 0;
    const dist = d.distribution || {};
    let distHtml = '';
    for (let i = 5; i >= 1; i--) distHtml += `<span style="margin-right:10px">${i}星: <b>${dist[i]||0}</b></span>`;
    document.getElementById('fb-stat-cards').innerHTML = `
      <div class="stat-card"><div class="val">${avgR.toFixed(1)} ${starsHtml(Math.round(avgR))}</div><div class="lbl">平均评分</div></div>
      <div class="stat-card"><div class="val">${d.rated_count}</div><div class="lbl">已评分数</div></div>
      <div class="stat-card"><div class="val">${ratedPct}%</div><div class="lbl">评分率</div></div>
      <div class="stat-card" style="grid-column:span 2"><div style="font-size:14px">${distHtml}</div><div class="lbl" style="margin-top:6px">评分分布</div></div>
    `;
    const filter = document.getElementById('fb-filter').value;
    let items = d.recent_feedback || [];
    if (filter === 'low') items = items.filter(r => r.feedback_rating <= 2);
    else if (filter === 'high') items = items.filter(r => r.feedback_rating >= 4);
    if (!items.length) {
      document.getElementById('fb-tbody').innerHTML = '<tr><td colspan="4" class="empty">暂无反馈记录</td></tr>';
      return;
    }
    document.getElementById('fb-tbody').innerHTML = items.map(r => {
      const shortQ = esc((r.question||'').length > 50 ? r.question.slice(0,50)+'…' : (r.question||''));
      return `<tr>
        <td style="text-align:center">${starsHtml(r.feedback_rating)}</td>
        <td>${shortQ}</td>
        <td style="font-size:12px;color:var(--text2)">${esc(r.feedback_comment || '—')}</td>
        <td style="font-size:12px;color:var(--text2)">${(r.feedback_ts||'').replace('T',' ')}</td>
      </tr>`;
    }).join('');
  } catch(e) {
    document.getElementById('fb-stat-cards').innerHTML = `<div class="empty">加载失败: ${e.message}</div>`;
  }
}

function loadAll() {
  loadStats();
}

// Auto-login if token stored
if (_token) {
  fetch(API + '/qa/stats', { headers: hdr() }).then(r => {
    if (r.ok) {
      document.getElementById('login-overlay').style.display = 'none';
      loadAll();
    } else {
      sessionStorage.removeItem('admin_token');
      _token = '';
    }
  }).catch(() => {});
}
</script>
</body>
</html>"""


@router.get("/admin", response_class=HTMLResponse)
async def admin_ui():
    return _ADMIN_HTML
