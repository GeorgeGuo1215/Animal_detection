"""
LivestockMind chat UI served at /chat — same port as the Agent API.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .qa_store import get_recent_example_questions

_PANDA_Q_PATTERN = re.compile(r"大熊猫|熊猫|panda|圈养.*繁殖|尾腺|内八字|肠梗阻|爱吃竹子", re.I)

DEFAULT_EXAMPLES: dict[str, dict[str, list[str]]] = {
    "farmer": {
        "zh": [
            "奶牛酮病如何早期识别？",
            "猪断奶腹泻怎么处理？",
            "羊三联疫苗什么时候打？",
            "夏季牛热应激怎么预防？",
        ],
        "en": [
            "How to spot ketosis in dairy cows early?",
            "What to do about post-weaning diarrhea in pigs?",
            "When should sheep get the triple vaccine?",
            "How to prevent heat stress in cattle during summer?",
        ],
    },
    "veterinarian": {
        "zh": [
            "奶牛真胃移位的诊断要点",
            "猪蓝耳病与圆环病毒如何鉴别？",
            "羊传染性胸膜肺炎治疗方案",
            "马肠绞痛的紧急处置流程",
        ],
        "en": [
            "Key diagnostic points for left displaced abomasum in dairy cows",
            "How to differentiate PRRS from PCV2 in pigs?",
            "Treatment protocol for contagious pleuropneumonia in sheep",
            "Emergency management steps for equine colic",
        ],
    },
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from shared.chat_feedback_widget import FEEDBACK_WIDGET_JS, render_feedback_widget_css
from shared.powered_by_footer import POWERED_BY_FOOTER_CSS, POWERED_BY_FOOTER_HTML

router = APIRouter()

_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LivestockMind — Large Livestock Veterinary Expert</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#f7f3ea;--card:#fff;--border:#e8ddd0;--accent:#6d4c41;--accent-light:#f7f3ea;
       --text:#1a1a1a;--text2:#6d5c52;--think-bg:#fefce8;--think-border:#fbbf24;--radius:16px;
       --powered-footer-bg:rgba(255,255,255,.72);--powered-footer-border:rgba(109,76,65,.12);
       --powered-logo-bg:#fff;--powered-logo-border:rgba(0,0,0,.08);
       --powered-logo-shadow:none;--powered-lab-color:var(--accent);
       --shadow-sm:0 4px 12px rgba(109,76,65,0.05);
       --shadow-md:0 8px 30px rgba(109,76,65,0.12);
       --shadow-lg:0 12px 40px rgba(109,76,65,0.15)}
body{font-family:Inter,"Noto Sans SC",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column;
     background-image:radial-gradient(circle,rgba(109,76,65,.04) 1px,transparent 1px);background-size:24px 24px}
header{background:rgba(255,255,255,0.75);padding:14px 24px;
       display:flex;align-items:center;gap:12px;flex-shrink:0;color:var(--text);border-bottom:1px solid rgba(109,76,65,0.1);
       box-shadow:0 4px 30px rgba(0,0,0,0.03);position:relative;z-index:10;
       backdrop-filter:blur(20px) saturate(160%);-webkit-backdrop-filter:blur(20px) saturate(160%);}
header h1{font-size:20px;font-weight:700;letter-spacing:-.3px;color:var(--accent)}
header .subtitle{font-size:12px;opacity:.8;color:var(--text2);font-weight:500;}
.settings{margin-left:auto;display:flex;gap:10px;align-items:center;font-size:13px}
.role-hint{font-size:12px;color:var(--text2);white-space:nowrap;font-weight:500;}
.lang-toggle,.role-toggle{display:flex;border-radius:10px;overflow:hidden;border:1px solid rgba(109,76,65,.15);background:rgba(255,255,255,0.5)}
.lang-toggle button,.role-toggle button{padding:6px 16px;border:none;background:transparent;color:var(--text2);font-size:13px;cursor:pointer;transition:all .3s cubic-bezier(0.4,0,0.2,1);font-weight:500}
.lang-toggle button.active,.role-toggle button.active{background:var(--accent-light);color:var(--accent);font-weight:600;box-shadow:0 2px 8px rgba(109,76,65,.1)}
.lang-toggle button:hover:not(.active),.role-toggle button:hover:not(.active){background:rgba(109,76,65,.05);color:var(--accent)}
#chat{flex:1;overflow-y:auto;padding:24px 20px;display:flex;flex-direction:column;gap:20px}
.msg{max-width:820px;width:100%;margin:0 auto;display:flex;gap:14px}
.msg.user{flex-direction:row-reverse}
.msg .bubble{padding:14px 18px;border-radius:var(--radius);line-height:1.75;font-size:15px;word-break:break-word;
             transition:all .3s cubic-bezier(0.4,0,0.2,1)}
.msg.user .bubble{background:linear-gradient(135deg,#8d6e63,#6d4c41);color:#fff;border-bottom-right-radius:4px;
                  white-space:pre-wrap;box-shadow:0 6px 18px rgba(109,76,65,.2)}
.msg.assistant .bubble{background:#fff;border:1px solid var(--border);
                       border-bottom-left-radius:4px;min-width:60px;box-shadow:0 4px 20px rgba(0,0,0,.04)}
.msg.assistant .bubble.raw-text{white-space:pre-wrap}
.msg .avatar{width:40px;height:40px;border-radius:50%;display:flex;align-items:center;
             justify-content:center;font-size:20px;flex-shrink:0;box-shadow:0 4px 12px rgba(0,0,0,.08);
             transition:all .3s cubic-bezier(0.34,1.56,0.64,1);background:#fff;}
.msg .avatar:hover{transform:scale(1.15) rotate(-5deg)}
.msg.user .avatar{background:linear-gradient(135deg,#efebe9,#bcaaa4);color:var(--accent)}
.msg.assistant .avatar{background:linear-gradient(135deg,#f7f3ea,#d7ccc8); border:2px solid #fff}
/* Markdown rendered content styles */
.msg.assistant .bubble h1,.msg.assistant .bubble h2,.msg.assistant .bubble h3{margin:14px 0 6px;font-weight:600;color:var(--accent)}
.msg.assistant .bubble h1{font-size:18px}.msg.assistant .bubble h2{font-size:16px}.msg.assistant .bubble h3{font-size:15px}
.msg.assistant .bubble p{margin:6px 0}
.msg.assistant .bubble ul,.msg.assistant .bubble ol{margin:6px 0;padding-left:20px}
.msg.assistant .bubble li{margin:3px 0}
.msg.assistant .bubble strong{color:#4e342e}
.msg.assistant .bubble code{background:#efebe9;padding:1px 5px;border-radius:4px;font-size:13px;color:#4e342e}
.msg.assistant .bubble pre{background:#3e2723;color:#efebe9;padding:14px;border-radius:10px;
       overflow-x:auto;margin:10px 0;font-size:13px;box-shadow:inset 0 2px 6px rgba(0,0,0,.15)}
.msg.assistant .bubble pre code{background:none;color:inherit;padding:0}
.msg.assistant .bubble blockquote{border-left:3px solid var(--accent);padding:8px 14px;margin:10px 0;
       background:var(--accent-light);border-radius:0 8px 8px 0;font-size:13px;color:#6d4c41}
.msg.assistant .bubble table{border-collapse:collapse;margin:10px 0;width:100%;border-radius:8px;overflow:hidden}
.msg.assistant .bubble th,.msg.assistant .bubble td{border:1px solid var(--border);padding:8px 12px;font-size:13px}
.msg.assistant .bubble th{background:var(--accent-light);font-weight:600}
.msg.assistant .bubble a{color:var(--accent);text-decoration:underline;transition:opacity .2s}
.msg.assistant .bubble a:hover{opacity:.7}
.msg.assistant .bubble a::after{content:" ↗";font-size:11px;opacity:.6}
.think-box{max-width:820px;width:100%;margin:4px auto;background:linear-gradient(135deg,#fefce8,#fffde7);
           border:1px solid var(--border);border-radius:12px;overflow:hidden;
           transition:all .3s cubic-bezier(0.4,0,0.2,1)}
.think-box:hover{box-shadow:0 4px 16px rgba(251,191,36,.1);border-color:var(--think-border)}
.think-header{padding:10px 16px;cursor:pointer;font-size:13px;color:#b45309;font-weight:500;
              display:flex;align-items:center;gap:8px;user-select:none;transition:background .2s}
.think-header:hover{background:rgba(251,191,36,.06)}
.think-header::before{content:"";display:inline-block;width:0;height:0;
       border-left:5px solid #b45309;border-top:4px solid transparent;border-bottom:4px solid transparent;
       transition:transform .3s cubic-bezier(0.34,1.56,0.64,1)}
.think-box.open .think-header::before{transform:rotate(90deg)}
.think-body{display:none;padding:8px 16px 14px;font-size:12px;color:#78716c;
            font-family:"SF Mono",Consolas,monospace;line-height:1.6;max-height:300px;overflow-y:auto}
.think-box.open .think-body{display:block;animation:slideDown .3s cubic-bezier(0.4,0,0.2,1) both}
.think-line{padding:3px 0;animation:fadeSlideIn .3s ease both}
.think-line.tool{color:#d97706;font-weight:500}
.think-line.status{color:#0284c7}
.think-line.hits{color:#15803d;font-weight:500}
#input-area{flex-shrink:0;background:rgba(255,255,255,0.85);border-top:1px solid rgba(109,76,65,0.08);padding:16px 20px 24px;
            box-shadow:0 -8px 30px rgba(0,0,0,.03);position:relative;z-index:5;backdrop-filter:blur(20px)}
#input-wrap{max-width:820px;margin:0 auto;display:flex;gap:12px;align-items:flex-end;
            background:#fff;border-radius:20px;padding:6px;box-shadow:0 4px 20px rgba(0,0,0,.05);border:1px solid var(--border);
            transition:all .3s cubic-bezier(0.4,0,0.2,1)}
#input-wrap:focus-within{border-color:var(--accent);box-shadow:0 8px 32px rgba(109,76,65,.15);transform:translateY(-1px)}
#input-wrap textarea{flex:1;resize:none;border:none;background:transparent;
       padding:10px 14px;font-size:15px;font-family:inherit;line-height:1.5;min-height:24px;max-height:150px;
       outline:none}
#input-wrap textarea::placeholder{color:#a8a29e}
#send-btn{padding:0 24px;height:44px;background:linear-gradient(135deg,#8d6e63,#6d4c41);color:#fff;border:none;
          border-radius:14px;font-size:15px;font-weight:600;cursor:pointer;
          transition:all .3s cubic-bezier(0.34,1.56,0.64,1);white-space:nowrap;display:inline-flex;align-items:center;justify-content:center;
          box-shadow:0 4px 12px rgba(109,76,65,.3)}
#send-btn:disabled{background:#e5e7eb;color:#9ca3af;cursor:not-allowed;box-shadow:none;transform:none}
#send-btn:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 6px 20px rgba(109,76,65,.4)}
#send-btn:active:not(:disabled){transform:translateY(0);box-shadow:0 2px 8px rgba(109,76,65,.4)}
.timing{font-size:11px;color:var(--text2);text-align:center;padding:4px 0;opacity:0.7}
.welcome{max-width:820px;margin:40px auto;text-align:center;color:var(--text2);animation:fadeSlideIn .6s cubic-bezier(0.2,0.8,0.2,1) both;padding:0 20px}
.welcome .livestock-icon{font-size:72px;margin-bottom:16px;animation:floatBounce 4s cubic-bezier(0.45,0,0.55,1) infinite;display:inline-block;filter:drop-shadow(0 10px 20px rgba(109,76,65,.15))}
.welcome h2{color:var(--text);margin-bottom:12px;font-size:28px;letter-spacing:-.5px;font-weight:700}
.welcome p{font-size:15px;line-height:1.8;max-width:540px;margin:0 auto;color:#6b7280}
/* Role picker cards */
.role-picker{display:flex;gap:20px;justify-content:center;margin:32px 0 12px;flex-wrap:wrap}
.role-card{flex:1;min-width:200px;max-width:260px;background:#fff;border:2px solid var(--border);
           border-radius:24px;padding:28px 20px 24px;cursor:pointer;transition:all .4s cubic-bezier(0.34,1.56,0.64,1);
           text-align:center;box-shadow:0 4px 16px rgba(0,0,0,.04);position:relative;overflow:hidden}
.role-card::after{content:'';position:absolute;inset:0;border-radius:22px;box-shadow:inset 0 0 0 2px var(--accent);opacity:0;transition:opacity .4s ease}
.role-card:hover{transform:translateY(-6px);box-shadow:0 16px 40px rgba(109,76,65,.1),0 4px 12px rgba(0,0,0,.04);border-color:transparent}
.role-card:hover::after{opacity:0.3}
.role-card.selected{border-color:transparent;background:linear-gradient(160deg,#fff,#f7f3ea);
                    box-shadow:0 12px 32px rgba(109,76,65,.15),0 2px 6px rgba(109,76,65,.08)}
.role-card.selected::after{opacity:1;box-shadow:inset 0 0 0 2.5px var(--accent)}
.role-card .rc-icon{font-size:42px;margin-bottom:12px;display:inline-block;transition:all .4s cubic-bezier(0.34,1.56,0.64,1)}
.role-card:hover .rc-icon{transform:scale(1.15) translateY(-4px) rotate(8deg);filter:drop-shadow(0 8px 12px rgba(109,76,65,.2))}
.role-card .rc-name{font-size:18px;font-weight:700;color:var(--text);margin-bottom:8px;transition:color .3s}
.role-card.selected .rc-name{color:var(--accent)}
.role-card .rc-desc{font-size:13px;color:var(--text2);line-height:1.6}
.role-card.selected .rc-desc{color:#8d6e63;font-weight:500}
.welcome .examples{margin-top:24px;display:flex;flex-wrap:wrap;gap:12px;justify-content:center}
.welcome .examples button{background:#fff;border:1px solid var(--border);border-radius:100px;
       padding:10px 20px;font-size:14px;color:var(--text);cursor:pointer;transition:all .3s cubic-bezier(0.4,0,0.2,1);
       box-shadow:0 2px 8px rgba(0,0,0,.03);font-weight:500}
.welcome .examples button:hover{background:var(--accent);color:#fff;border-color:var(--accent);transform:translateY(-3px);
       box-shadow:0 8px 20px rgba(109,76,65,.25)}
__FEEDBACK_WIDGET_CSS__
/* Keyframe animations */
@keyframes fadeSlideIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideDown{from{opacity:0;transform:translateY(-10px);max-height:0}to{opacity:1;transform:translateY(0);max-height:300px}}
@keyframes floatBounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-12px)}}
@keyframes pulse3{0%,80%,100%{opacity:.3}40%{opacity:1}}
@keyframes spin{to{transform:rotate(360deg)}}
/* Scrollbar */
html{scroll-behavior:smooth}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(109,76,65,.25);border-radius:10px;border:2px solid var(--bg)}
::-webkit-scrollbar-thumb:hover{background:rgba(109,76,65,.45)}
/* Message entrance */
.msg{animation:fadeSlideIn .4s cubic-bezier(0.2,0.8,0.2,1) both}
.timing{animation:fadeSlideIn .4s ease both}
@media(max-width:600px){
  header{padding:12px 16px;gap:10px}
  header h1{font-size:18px}
  .role-hint{display:none}
  .role-toggle button{padding:6px 12px;font-size:13px}
  .role-picker{gap:12px}
  .role-card{min-width:140px;padding:20px 16px;border-radius:18px}
  .role-card .rc-icon{font-size:36px}
  .role-card .rc-name{font-size:16px}
  #input-wrap{padding:4px 8px}
  #input-wrap textarea{font-size:16px;padding:8px 10px}
  #send-btn{padding:0 18px;height:40px}
}
__POWERED_BY_FOOTER_CSS__
</style>
</head>
<body>

<header>
  <h1>LivestockMind</h1>
  <span class="subtitle" id="header-subtitle">Large Livestock Veterinary Expert</span>
  <div class="settings">
    <div class="lang-toggle">
      <button id="lang-zh" onclick="setLang('zh')">中文</button>
      <button id="lang-en" class="active" onclick="setLang('en')">EN</button>
    </div>
    <span class="role-hint" id="header-role-label">Role: 🐄 Farmer</span>
    <div class="role-toggle">
      <button id="role-farmer" class="active" onclick="setRole('farmer')">🐄 Farmer</button>
      <button id="role-veterinarian" onclick="setRole('veterinarian')">🩺 Veterinarian</button>
    </div>
  </div>
</header>

<div id="chat">
  <div class="welcome" id="welcome-panel">
    <div class="livestock-icon">🐄</div>
    <h2 id="welcome-title">Welcome to LivestockMind</h2>
    <p id="welcome-desc">I am an AI assistant for cattle, pigs, sheep, and horses, powered by veterinary textbooks and literature.<br>Choose your role first for answers tailored to you 👇</p>
    <div class="role-picker">
      <div class="role-card selected" id="card-farmer" onclick="setRole('farmer')">
        <div class="rc-icon">🐄</div>
        <div class="rc-name" id="card-farmer-name">Farmer</div>
        <div class="rc-desc" id="card-farmer-desc">Practical and production-focused<br>For ranchers and frontline workers</div>
      </div>
      <div class="role-card" id="card-veterinarian" onclick="setRole('veterinarian')">
        <div class="rc-icon">🩺</div>
        <div class="rc-name" id="card-vet-name">Veterinarian</div>
        <div class="rc-desc" id="card-vet-desc">Detailed and evidence-based<br>For vets and researchers</div>
      </div>
    </div>
    <div class="examples" id="example-questions"></div>
  </div>
</div>

<div id="input-area">
  <div id="input-wrap">
    <textarea id="user-input" rows="1" placeholder="Ask a question, e.g. How to spot ketosis in dairy cows?" autofocus></textarea>
    <button id="send-btn">Send</button>
  </div>
</div>

__POWERED_BY_FOOTER_HTML__

<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

const I18N = {
  zh: {
    pageTitle: 'LivestockMind — 大型畜牧兽医专家',
    subtitle: '大型畜牧兽医专家',
    roleFarmer: '🐄 牧场主',
    roleVet: '🩺 兽医',
    roleLabelFarmer: '身份：🐄 牧场主',
    roleLabelVet: '身份：🩺 兽医',
    welcomeTitle: '欢迎使用 LivestockMind',
    welcomeDesc: '我是大型畜牧兽医 AI 助手，知识来自专业教材与兽医文献。<br>请先选择你的身份，获取最适合你的回答风格 👇',
    cardFarmerName: '牧场主',
    cardFarmerDesc: '务实易懂、贴近生产<br>适合养殖户与一线从业者',
    cardVetName: '兽医',
    cardVetDesc: '专业详尽、引用文献<br>适合兽医师与科研人员',
    placeholderFarmer: '输入你的问题，如：奶牛酮病如何识别？',
    placeholderVet: '输入专业问题，如：奶牛真胃移位的诊断要点...',
    sendBtn: '发送',
    thinkHeader: '思考过程',
    planning: '制定计划中...',
    thinking: '思考中...',
    planComplete: '计划完成:',
    toolCalling: '调用工具:',
    reason: '原因:',
    toolComplete: '返回',
    results: '条结果',
    decidedFinal: '准备生成回答',
    generating: '生成回答...',
    askUser: '需要确认:',
    timingPrefix: '耗时',
    fbLabel: '回答质量如何？',
    fbPh: '补充评论（可选）',
    fbSubmit: '提交',
    fbSubmitting: '提交中...',
    fbThanks: '感谢你的反馈！',
    fbFail: '提交失败',
    fbNet: '网络错误',
    fbStarTitle: '{n} 星',
    examplesFarmer: [
      '奶牛酮病如何早期识别？',
      '猪断奶腹泻怎么处理？',
      '羊三联疫苗什么时候打？',
      '夏季牛热应激怎么预防？',
    ],
    examplesVet: [
      '奶牛真胃移位的诊断要点',
      '猪蓝耳病与圆环病毒如何鉴别？',
      '羊传染性胸膜肺炎治疗方案',
      '马肠绞痛的紧急处置流程',
    ],
  },
  en: {
    pageTitle: 'LivestockMind — Large Livestock Veterinary Expert',
    subtitle: 'Large Livestock Veterinary Expert',
    roleFarmer: '🐄 Farmer',
    roleVet: '🩺 Veterinarian',
    roleLabelFarmer: 'Role: 🐄 Farmer',
    roleLabelVet: 'Role: 🩺 Veterinarian',
    welcomeTitle: 'Welcome to LivestockMind',
    welcomeDesc: 'I am an AI assistant for cattle, pigs, sheep, and horses, powered by veterinary textbooks and literature.<br>Choose your role first for answers tailored to you 👇',
    cardFarmerName: 'Farmer',
    cardFarmerDesc: 'Practical and production-focused<br>For ranchers and frontline workers',
    cardVetName: 'Veterinarian',
    cardVetDesc: 'Detailed and evidence-based<br>For vets and researchers',
    placeholderFarmer: 'Ask a question, e.g. How to spot ketosis in dairy cows?',
    placeholderVet: 'Ask a clinical question, e.g. diagnostic points for LDA in cattle...',
    sendBtn: 'Send',
    thinkHeader: 'Reasoning',
    planning: 'Planning...',
    thinking: 'Thinking...',
    planComplete: 'Plan complete:',
    toolCalling: 'Calling tool:',
    reason: 'Reason:',
    toolComplete: 'Returned',
    results: 'results',
    decidedFinal: 'Preparing answer',
    generating: 'Generating answer...',
    askUser: 'Need confirmation:',
    timingPrefix: 'Elapsed',
    fbLabel: 'How helpful was this answer?',
    fbPh: 'Optional comment',
    fbSubmit: 'Submit',
    fbSubmitting: 'Submitting...',
    fbThanks: 'Thanks for your feedback!',
    fbFail: 'Submit failed',
    fbNet: 'Network error',
    fbStarTitle: '{n} star(s)',
    examplesFarmer: [
      'How to spot ketosis in dairy cows early?',
      'What to do about post-weaning diarrhea in pigs?',
      'When should sheep get the triple vaccine?',
      'How to prevent heat stress in cattle during summer?',
    ],
    examplesVet: [
      'Key diagnostic points for left displaced abomasum in dairy cows',
      'How to differentiate PRRS from PCV2 in pigs?',
      'Treatment protocol for contagious pleuropneumonia in sheep',
      'Emergency management steps for equine colic',
    ],
  },
};

let currentLang = localStorage.getItem('livestock_lang') || 'en';
let currentRole = 'farmer';
let dynamicExamplesByRole = {};
const messages = [];
let streaming = false;
let welcomeShown = true;
let sessionId = sessionStorage.getItem('livestock_session_id') || '';

function t(key) {
  return (I18N[currentLang] && I18N[currentLang][key]) || I18N.zh[key] || key;
}

function setLang(lang) {
  if (lang !== 'zh' && lang !== 'en') return;
  currentLang = lang;
  localStorage.setItem('livestock_lang', lang);
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.getElementById('lang-zh').classList.toggle('active', lang === 'zh');
  document.getElementById('lang-en').classList.toggle('active', lang === 'en');
  dynamicExamplesByRole = {};
  applyLanguage();
  loadExamples(currentRole);
}

function applyLanguage() {
  document.title = t('pageTitle');
  const sub = document.getElementById('header-subtitle');
  if (sub) sub.textContent = t('subtitle');
  const wt = document.getElementById('welcome-title');
  if (wt) wt.textContent = t('welcomeTitle');
  const wd = document.getElementById('welcome-desc');
  if (wd) wd.innerHTML = t('welcomeDesc');
  const cfn = document.getElementById('card-farmer-name');
  if (cfn) cfn.textContent = t('cardFarmerName');
  const cfd = document.getElementById('card-farmer-desc');
  if (cfd) cfd.innerHTML = t('cardFarmerDesc');
  const cvn = document.getElementById('card-vet-name');
  if (cvn) cvn.textContent = t('cardVetName');
  const cvd = document.getElementById('card-vet-desc');
  if (cvd) cvd.innerHTML = t('cardVetDesc');
  document.getElementById('role-farmer').textContent = t('roleFarmer');
  document.getElementById('role-veterinarian').textContent = t('roleVet');
  sendBtn.textContent = t('sendBtn');
  document.documentElement.lang = currentLang === 'zh' ? 'zh-CN' : 'en';
  document.getElementById('lang-zh').classList.toggle('active', currentLang === 'zh');
  document.getElementById('lang-en').classList.toggle('active', currentLang === 'en');
  setRole(currentRole);
}

async function ensureSessionId() {
  if (sessionId) return sessionId;
  try {
    const resp = await fetch('/sessions', { method: 'POST' });
    const data = await resp.json();
    if (data && data.ok && data.session_id) {
      sessionId = data.session_id;
      sessionStorage.setItem('livestock_session_id', sessionId);
    }
  } catch {}
  return sessionId;
}

function setRole(role) {
  currentRole = role;
  // Header toggle buttons
  document.getElementById('role-farmer').classList.toggle('active', role === 'farmer');
  document.getElementById('role-veterinarian').classList.toggle('active', role === 'veterinarian');
  // Welcome page cards
  const ce = document.getElementById('card-farmer');
  const cr = document.getElementById('card-veterinarian');
  if (ce) ce.classList.toggle('selected', role === 'farmer');
  if (cr) cr.classList.toggle('selected', role === 'veterinarian');
  // Header label
  const lbl = document.getElementById('header-role-label');
  if (lbl) lbl.textContent = role === 'veterinarian' ? t('roleLabelVet') : t('roleLabelFarmer');
  renderExamples();
  loadExamples(role);
  inputEl.placeholder = role === 'veterinarian' ? t('placeholderVet') : t('placeholderFarmer');
}

function staticExamples(role) {
  return role === 'veterinarian' ? t('examplesVet') : t('examplesFarmer');
}

function renderExamples() {
  const container = document.getElementById('example-questions');
  if (!container) return;
  container.innerHTML = '';
  const dynamicExamples = dynamicExamplesByRole[currentRole] || [];
  const examples = dynamicExamples.length ? dynamicExamples : staticExamples(currentRole);
  examples.forEach(q => {
    const btn = document.createElement('button');
    btn.textContent = q;
    btn.onclick = () => { inputEl.value = q; send(); };
    container.appendChild(btn);
  });
}

async function loadExamples(role = currentRole) {
  if (dynamicExamplesByRole[role]?.length) {
    if (role === currentRole) renderExamples();
    return;
  }
  try {
    const resp = await fetch(`/chat/example-questions?user_role=${encodeURIComponent(role)}&lang=${currentLang}`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data || !Array.isArray(data.examples) || !data.examples.length) return;
    dynamicExamplesByRole[role] = data.examples
      .filter(item => typeof item === 'string' && item.trim())
      .slice(0, 6);
    if (role === currentRole) renderExamples();
  } catch {}
}

renderExamples();
loadExamples(currentRole);
applyLanguage();

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
       <div class="think-header" onclick="this.parentElement.classList.toggle('open')">${t('thinkHeader')}</div>
       <div class="think-body"></div>
     </div>
     <div class="msg assistant" id="${id}">
       <div class="avatar">🐄</div>
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
  bubble.classList.add('raw-text');
  bubble.textContent = text;
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
    const effectiveSessionId = await ensureSessionId();
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST', headers,
      body: JSON.stringify({
        model: 'livestock-plan-solve',
        messages: messages.slice(-12),
        session_id: effectiveSessionId || undefined,
        stream: true, temperature: 0.7, max_tokens: 2048,
        debug_timing: true,
        user_role: currentRole,
        response_lang: currentLang
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

          if (status === 'planning') appendThink(msgId, '📋 ' + (detail.message || t('planning')), 'status');
          else if (status === 'thinking') appendThink(msgId, '🤔 ' + (detail.message || t('thinking')), 'status');
          else if (status === 'plan_complete') {
            const steps = (detail.plan || []).filter(s => s.type === 'tool');
            appendThink(msgId, '✅ ' + t('planComplete') + ' ' + steps.map(s => s.tool_name).join(' → '), 'status');
          }
          else if (status === 'tool_calling') {
            appendThink(msgId, '🔧 ' + t('toolCalling') + ' ' + (detail.tool_name || ''), 'tool');
            if (detail.reason) appendThink(msgId, '   ' + t('reason') + ' ' + detail.reason, '');
          }
          else if (status === 'tool_complete') appendThink(msgId, '✅ ' + t('toolComplete') + ' ' + (detail.hits_count ?? 0) + ' ' + t('results'), 'hits');
          else if (status === 'decided_final') appendThink(msgId, '💡 ' + (detail.reason || t('decidedFinal')), 'status');
          else if (status === 'generating') {
            appendThink(msgId, '📝 ' + t('generating'), 'status');
            const thinkBox = document.getElementById(msgId + '-think');
            if (thinkBox) thinkBox.classList.remove('open');
          }
          else if (status === 'timing_summary') {
            const t = detail.timing || [];
            const total = t.find(x => x.step === 'total');
            if (total) addTiming(`${t('timingPrefix')} ${(total.ms / 1000).toFixed(1)}s`);
          }
          else if (status === 'ask_user') {
            const question = detail.question || delta?.content || '';
            if (question) {
              fullContent = question;
              appendThink(msgId, '❓ ' + t('askUser') + ' ' + question, 'status');
              renderMarkdown(msgId, fullContent);
            }
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

_CHAT_HTML = _CHAT_HTML.replace("__FEEDBACK_WIDGET_CSS__", render_feedback_widget_css("livestock"))
_CHAT_HTML = _CHAT_HTML.replace("__POWERED_BY_FOOTER_CSS__", POWERED_BY_FOOTER_CSS)
_CHAT_HTML = _CHAT_HTML.replace("__POWERED_BY_FOOTER_HTML__", POWERED_BY_FOOTER_HTML)
_CHAT_HTML = _CHAT_HTML.replace("__FEEDBACK_WIDGET_JS__", FEEDBACK_WIDGET_JS)


@router.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return _CHAT_HTML


@router.get("/chat/example-questions")
async def chat_example_questions(
    limit: int = 6,
    user_role: str = "farmer",
    lang: str = "en",
):
    role = user_role if user_role in {"farmer", "veterinarian"} else "farmer"
    lang_key = "en" if lang.lower().startswith("en") else "zh"
    examples = await get_recent_example_questions(
        limit=max(1, min(limit, 8)),
        user_role=role,
    )
    filtered = [q for q in examples if q and not _PANDA_Q_PATTERN.search(q)]
    if not filtered:
        filtered = DEFAULT_EXAMPLES.get(role, {}).get(lang_key, [])
    return {"ok": True, "examples": filtered[: max(1, min(limit, 8))]}


_ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LivestockMind — Admin</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#f7f3ea;--card:#fff;--border:#e8ddd0;--accent:#6d4c41;--accent-light:#f7f3ea;
      --text:#1a1a1a;--text2:#6b7280;--radius:10px;--danger:#c62828;
      --shadow-sm:0 1px 3px rgba(0,0,0,.06);--shadow-md:0 4px 16px rgba(0,0,0,.08)}
body{font-family:Inter,"Noto Sans SC",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:var(--bg);color:var(--text);min-height:100vh;
     background-image:radial-gradient(circle,rgba(109,76,65,.025) 1px,transparent 1px);background-size:20px 20px}
header{background:linear-gradient(135deg,#4e342e 0%,#6d4c41 50%,#8d6e63 100%);padding:16px 24px;
       display:flex;align-items:center;gap:12px;color:#fff;
       box-shadow:0 2px 24px rgba(78,52,46,.15);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
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
.login-box input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(109,76,65,.1)}
.login-box button{width:100%;padding:11px;background:linear-gradient(135deg,#6d4c41,#8d6e63);color:#fff;
                  border:none;border-radius:10px;font-size:15px;cursor:pointer;font-weight:600;
                  transition:all .2s;box-shadow:0 2px 8px rgba(109,76,65,.2)}
.login-box button:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(109,76,65,.3)}
.login-box .err{color:var(--danger);font-size:13px;margin-top:8px;min-height:18px}

.lang-toggle{display:flex;border-radius:8px;overflow:hidden;border:1px solid rgba(255,255,255,.25);background:rgba(255,255,255,.12)}
.lang-toggle button{padding:5px 12px;border:none;background:transparent;color:rgba(255,255,255,.85);font-size:12px;cursor:pointer;font-weight:500}
.lang-toggle button.active{background:rgba(255,255,255,.22);color:#fff;font-weight:700}

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
.stat-card:hover .val{color:#4e342e}
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
.filters input:focus,.filters select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(109,76,65,.08)}
.btn{padding:8px 20px;background:linear-gradient(135deg,#6d4c41,#8d6e63);color:#fff;border:none;border-radius:8px;
     font-size:13px;cursor:pointer;font-weight:600;transition:all .2s;box-shadow:0 2px 6px rgba(109,76,65,.15)}
.btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(109,76,65,.25)}
.btn:active{transform:translateY(0)}
.btn-sm{padding:4px 12px;font-size:12px;border-radius:5px}

/* Pagination */
.pagination{display:flex;gap:8px;justify-content:center;margin-top:18px}
.page-btn{padding:6px 16px;border:1.5px solid var(--border);background:#fff;
          border-radius:8px;cursor:pointer;font-size:13px;transition:all .2s;font-weight:500}
.page-btn.active{background:linear-gradient(135deg,#6d4c41,#8d6e63);color:#fff;border-color:var(--accent);
                 box-shadow:0 2px 8px rgba(109,76,65,.2)}
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
::-webkit-scrollbar-thumb{background:rgba(109,76,65,.2);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(109,76,65,.35)}
</style>
</head>
<body>

<!-- Login overlay -->
<div id="login-overlay">
  <div class="login-box">
    <h2>🐄 Admin Login</h2>
    <p>Enter Admin Token to access the dashboard</p>
    <input type="password" id="token-input" placeholder="X-Admin-Token" autocomplete="off">
    <button onclick="doLogin()">Login</button>
    <div class="err" id="login-err"></div>
  </div>
</div>

<header>
  <span style="font-size:24px">🐄</span>
  <div>
    <h1 id="admin-title">LivestockMind Admin</h1>
    <span class="sub" id="admin-sub">Q&A · Analytics · Knowledge Gaps</span>
  </div>
  <div class="lang-toggle" style="margin-left:auto;margin-right:12px">
    <button id="admin-lang-zh" onclick="setAdminLang('zh')">中文</button>
    <button id="admin-lang-en" class="active" onclick="setAdminLang('en')">EN</button>
  </div>
  <button class="logout" onclick="doLogout()">Logout</button>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('stats',this)">Analytics</button>
  <button class="tab-btn" onclick="switchTab('history',this)">History</button>
  <button class="tab-btn" onclick="switchTab('gaps',this)">Knowledge Gaps</button>
  <button class="tab-btn" onclick="switchTab('quality',this)">Quality</button>
  <button class="tab-btn" onclick="switchTab('feedback',this)">Feedback</button>
</div>

<!-- Tab 1: Stats -->
<div id="tab-stats" class="tab-content active">
  <div class="filters">
    <select id="stats-role-filter" title="按角色查看统计">
      <option value="">全部角色（总览）</option>
      <option value="farmer">牧场主</option>
      <option value="veterinarian">兽医</option>
    </select>
    <button class="btn btn-sm" onclick="loadStats()">筛选</button>
  </div>
  <div class="stat-grid" id="stat-cards">
    <div class="loading">加载中...</div>
  </div>
  <div class="section">
    <div class="section-title">角色分布</div>
    <table>
      <thead><tr><th>角色</th><th>问答次数</th><th>占比</th></tr></thead>
      <tbody id="role-tbody"><tr><td colspan="3" class="loading">加载中...</td></tr></tbody>
    </table>
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
    <select id="role-filter" title="按角色筛选">
      <option value="">全部角色</option>
      <option value="farmer">牧场主</option>
      <option value="veterinarian">兽医</option>
    </select>
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
          <th style="width:92px">角色</th>
          <th>问题</th>
          <th style="width:120px">工具</th>
          <th style="width:70px">RAG命中</th>
          <th style="width:60px">评分</th>
          <th style="width:80px">耗时</th>
        </tr>
      </thead>
      <tbody id="hist-tbody"><tr><td colspan="7" class="loading">加载中...</td></tr></tbody>
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
<div id="tab-quality" class="tab-content">
  <div class="stat-grid" id="quality-stat-cards"><div class="loading">加载中...</div></div>
  <div class="filters">
    <select id="quality-role-filter" title="按角色筛选质量评估">
      <option value="">全部角色（总览）</option>
      <option value="farmer">牧场主</option>
      <option value="veterinarian">兽医</option>
    </select>
    <input type="date" id="quality-date-from" title="质量评估起始日期">
    <span style="color:var(--text2);font-size:13px">至</span>
    <input type="date" id="quality-date-to" title="质量评估结束日期">
    <button class="btn btn-sm" onclick="loadQualityTab()">筛选</button>
    <button class="btn btn-sm" style="background:#757575" onclick="clearQualityFilters()">清空</button>
    <span id="quality-total" style="font-size:13px;color:var(--text2)"></span>
  </div>
  <div class="section">
    <div class="section-title">质量趋势（近 30 天）</div>
    <table>
      <thead><tr><th>日期</th><th>评估数</th><th>平均质量分</th><th>问题率</th><th>缺少网搜交叉验证</th></tr></thead>
      <tbody id="quality-trend-tbody"><tr><td colspan="5" class="loading">加载中...</td></tr></tbody>
    </table>
  </div>
  <div class="section">
    <div class="section-title">问题样本</div>
    <table>
      <thead><tr><th style="width:130px">时间</th><th>问题摘要</th><th style="width:100px">质量分</th><th style="width:220px">问题标记</th></tr></thead>
      <tbody id="quality-samples-tbody"><tr><td colspan="4" class="loading">加载中...</td></tr></tbody>
    </table>
  </div>
</div>

<!-- Tab 5: Feedback -->
<div id="tab-feedback" class="tab-content">
  <div class="stat-grid" id="fb-stat-cards"><div class="loading">加载中...</div></div>
  <div class="filters">
    <select id="fb-role-filter" title="按角色筛选反馈">
      <option value="">全部角色（总览）</option>
      <option value="farmer">牧场主</option>
      <option value="veterinarian">兽医</option>
    </select>
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
let adminLang = localStorage.getItem('livestock_lang') || 'en';

const ADMIN_I18N = {
  zh: {
    title: 'LivestockMind 管理后台',
    sub: '问答记录 · 访问统计 · 知识盲区',
    logout: '退出登录',
    loginTitle: '🐄 管理员登录',
    loginDesc: '请输入 Admin Token 以访问管理后台',
    loginBtn: '登录',
    tabStats: '访问统计',
    tabHistory: '问答历史',
    tabGaps: '知识盲区',
    tabQuality: '质量评估',
    tabFeedback: '用户反馈',
    invalidToken: 'Token 无效，请重试',
    networkError: '网络错误，请检查服务是否运行',
  },
  en: {
    title: 'LivestockMind Admin',
    sub: 'Q&A · Analytics · Knowledge Gaps',
    logout: 'Logout',
    loginTitle: '🐄 Admin Login',
    loginDesc: 'Enter Admin Token to access the dashboard',
    loginBtn: 'Login',
    tabStats: 'Analytics',
    tabHistory: 'History',
    tabGaps: 'Knowledge Gaps',
    tabQuality: 'Quality',
    tabFeedback: 'Feedback',
    invalidToken: 'Invalid token, please retry',
    networkError: 'Network error — check if the service is running',
  },
};

function at(key) {
  return (ADMIN_I18N[adminLang] && ADMIN_I18N[adminLang][key]) || ADMIN_I18N.zh[key] || key;
}

function setAdminLang(lang) {
  if (lang !== 'zh' && lang !== 'en') return;
  adminLang = lang;
  localStorage.setItem('livestock_lang', lang);
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.getElementById('admin-lang-zh').classList.toggle('active', lang === 'zh');
  document.getElementById('admin-lang-en').classList.toggle('active', lang === 'en');
  applyAdminLang();
}

function applyAdminLang() {
  document.title = at('title');
  const lt = document.querySelector('.login-box h2');
  if (lt) lt.textContent = at('loginTitle');
  const lp = document.querySelector('.login-box p');
  if (lp) lp.textContent = at('loginDesc');
  const lb = document.querySelector('.login-box button');
  if (lb) lb.textContent = at('loginBtn');
  document.getElementById('admin-title').textContent = at('title');
  document.getElementById('admin-sub').textContent = at('sub');
  document.querySelector('header .logout').textContent = at('logout');
  const tabs = document.querySelectorAll('.tab-btn');
  const tabKeys = ['tabStats', 'tabHistory', 'tabGaps', 'tabQuality', 'tabFeedback'];
  tabs.forEach((btn, i) => { if (tabKeys[i]) btn.textContent = at(tabKeys[i]); });
}
applyAdminLang();

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
      document.getElementById('login-err').textContent = at('invalidToken');
      _token = '';
    }
  }).catch(() => {
    document.getElementById('login-err').textContent = at('networkError');
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
  if (name === 'quality') loadQualityTab();
  if (name === 'feedback') loadFeedbackTab();
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function roleLabel(role) {
  const zh = { farmer: '牧场主', veterinarian: '兽医', unlabeled: '未标注' };
  const en = { farmer: 'Farmer', veterinarian: 'Veterinarian', unlabeled: 'Unlabeled' };
  const map = adminLang === 'en' ? en : zh;
  if (role === 'veterinarian') return map.veterinarian;
  if (role === 'farmer') return map.farmer;
  return map.unlabeled;
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
    const statsRole = document.getElementById('stats-role-filter').value;
    let statsUrl = '/qa/stats';
    if (statsRole) statsUrl += `?user_role=${encodeURIComponent(statsRole)}`;
    const d = await apiFetch(statsUrl);
    let fbData = {};
    try {
      let fbUrl = '/qa/feedback-stats';
      if (statsRole) fbUrl += `?user_role=${encodeURIComponent(statsRole)}`;
      fbData = await apiFetch(fbUrl);
    } catch {}
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

    const roles = d.role_distribution || [];
    document.getElementById('role-tbody').innerHTML = roles.length
      ? roles.map(r => {
          const count = Number(r.count || 0);
          const pct = d.total > 0 ? ((count / d.total) * 100).toFixed(1) : '0.0';
          return `<tr><td>${esc(roleLabel(r.user_role || ''))}</td><td><span class="count-badge">${count}</span></td><td>${pct}%</td></tr>`;
        }).join('')
      : '<tr><td colspan="3" class="empty">暂无数据</td></tr>';

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
  const role = document.getElementById('role-filter').value;
  const df = document.getElementById('date-from').value;
  const dt = document.getElementById('date-to').value;
  let url = `/qa/history?page=${page}&page_size=${PAGE_SIZE}`;
  if (kw) url += '&keyword=' + encodeURIComponent(kw);
  if (role) url += '&user_role=' + encodeURIComponent(role);
  if (df) url += '&date_from=' + df;
  if (dt) url += '&date_to=' + dt;

  document.getElementById('hist-tbody').innerHTML = '<tr><td colspan="7" class="loading">加载中...</td></tr>';
  try {
    const d = await apiFetch(url);
    document.getElementById('hist-total').textContent = `共 ${d.total} 条`;
    const rows = d.records || [];
    if (!rows.length) {
      document.getElementById('hist-tbody').innerHTML = '<tr><td colspan="7" class="empty">暂无记录</td></tr>';
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
          <td style="font-size:12px">${esc(roleLabel(r.user_role || ''))}</td>
          <td>${shortQ}</td>
          <td style="font-size:12px">${esc(tools)}</td>
          <td style="text-align:center">${r.rag_hit_count}</td>
          <td style="text-align:center">${ratingHtml}</td>
          <td style="font-size:12px">${fmt_ms(r.response_time_ms)}</td>
        </tr>
        <tr class="detail-row">
          <td colspan="7">
            <div class="detail-inner" id="detail-${i}">
<strong>问题：</strong>${esc(r.question)}

<strong>回答：</strong>
${esc(r.answer)}

<strong>模型：</strong>${esc(r.model)} | <strong>来源IP：</strong>${esc(r.source_ip)} | <strong>角色：</strong>${esc(roleLabel(r.user_role || ''))}${fbLine}</div>
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
    document.getElementById('hist-tbody').innerHTML = `<tr><td colspan="7" class="empty">加载失败: ${e.message}</td></tr>`;
  }
}

function toggleDetail(i) {
  const el = document.getElementById('detail-' + i);
  el.classList.toggle('open');
}

function clearFilters() {
  document.getElementById('kw-input').value = '';
  document.getElementById('role-filter').value = '';
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
    const role = document.getElementById('fb-role-filter').value;
    let url = '/qa/feedback-stats';
    if (role) url += `?user_role=${encodeURIComponent(role)}`;
    const d = await apiFetch(url);
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
        <td>${shortQ}<div style="font-size:11px;color:var(--text2);margin-top:4px">${esc(roleLabel(r.user_role || ''))}</div></td>
        <td style="font-size:12px;color:var(--text2)">${esc(r.feedback_comment || '—')}</td>
        <td style="font-size:12px;color:var(--text2)">${(r.feedback_ts||'').replace('T',' ')}</td>
      </tr>`;
    }).join('');
  } catch(e) {
    document.getElementById('fb-stat-cards').innerHTML = `<div class="empty">加载失败: ${e.message}</div>`;
  }
}

// ---- Quality tab ----
async function loadQualityTab() {
  try {
    const role = document.getElementById('quality-role-filter').value;
    const df = document.getElementById('quality-date-from').value;
    const dt = document.getElementById('quality-date-to').value;
    let url = '/qa/quality-report?limit=300&sample_size=20';
    if (role) url += '&user_role=' + encodeURIComponent(role);
    if (df) url += '&date_from=' + df;
    if (dt) url += '&date_to=' + dt;
    const d = await apiFetch(url);
    const summary = d.trend_summary || {};
    const recent = summary.recent_7d || {};
    const qDelta = Number(summary.quality_score_delta || 0);
    const pDelta = Number(summary.problem_rate_delta || 0);
    const qDeltaStr = `${qDelta >= 0 ? '+' : ''}${qDelta.toFixed(1)}`;
    const pDeltaStr = `${pDelta >= 0 ? '+' : ''}${pDelta.toFixed(1)}%`;
    document.getElementById('quality-total').textContent = `共评估 ${d.evaluated_count || 0} 条`;

    document.getElementById('quality-stat-cards').innerHTML = `
      <div class="stat-card"><div class="val">${Number(d.avg_quality_score || 0).toFixed(1)}</div><div class="lbl">整体平均质量分</div></div>
      <div class="stat-card"><div class="val">${d.problem_rate || 0}%</div><div class="lbl">整体问题率</div></div>
      <div class="stat-card"><div class="val">${Number(recent.avg_quality_score || 0).toFixed(1)} <span style="font-size:14px;color:${qDelta >= 0 ? '#2e7d32' : '#c62828'}">${qDeltaStr}</span></div><div class="lbl">近 7 天质量分（较前 7 天）</div></div>
      <div class="stat-card"><div class="val">${recent.problem_rate || 0}% <span style="font-size:14px;color:${pDelta <= 0 ? '#2e7d32' : '#c62828'}">${pDeltaStr}</span></div><div class="lbl">近 7 天问题率（较前 7 天）</div></div>
      <div class="stat-card"><div class="val">${d.evaluated_count}</div><div class="lbl">本次评估样本数</div></div>
      <div class="stat-card"><div class="val">${(d.flag_counts && d.flag_counts.missing_web_cross_check) || 0}</div><div class="lbl">缺少网搜交叉验证</div></div>
    `;

    const trend = d.trend || [];
    document.getElementById('quality-trend-tbody').innerHTML = trend.length
      ? trend.map(r => `<tr>
          <td>${esc(r.date_key)}</td>
          <td><span class="count-badge">${r.evaluated_count}</span></td>
          <td>${Number(r.avg_quality_score || 0).toFixed(1)}</td>
          <td>${r.problem_rate || 0}%</td>
          <td>${r.missing_web_cross_check || 0}</td>
        </tr>`).join('')
      : '<tr><td colspan="5" class="empty">暂无趋势数据</td></tr>';

    const samples = d.samples || [];
    document.getElementById('quality-samples-tbody').innerHTML = samples.length
      ? samples.map((r, i) => `<tr class="expandable" onclick="toggleQualityDetail(${i})">
          <td style="font-size:12px;color:var(--text2)">${(r.ts || '').replace('T',' ')}</td>
          <td>${esc((r.question || '').length > 70 ? r.question.slice(0,70) + '…' : (r.question || ''))}</td>
          <td>${Number(r.quality_score || 0).toFixed(0)}</td>
          <td style="font-size:12px;color:var(--text2)">${esc((r.flags || []).join(', ') || '—')}</td>
        </tr>
        <tr class="detail-row">
          <td colspan="4">
            <div class="detail-inner" id="quality-detail-${i}">
<strong>问题：</strong>${esc(r.question || '')}

<strong>质量分：</strong>${Number(r.quality_score || 0).toFixed(0)} | <strong>模型：</strong>${esc(r.model || '—')} | <strong>请求 ID：</strong>${esc(r.request_id || '—')}

<strong>工具：</strong>${esc((r.tools_used || []).join(', ') || '—')} | <strong>RAG 命中：</strong>${r.rag_hit_count ?? 0} | <strong>RAG 最高分：</strong>${r.rag_best_score ?? 0}

<strong>标记：</strong>${esc((r.flags || []).join(', ') || '—')}</div>
          </td>
        </tr>`).join('')
      : '<tr><td colspan="4" class="empty">暂无问题样本</td></tr>';
  } catch(e) {
    document.getElementById('quality-stat-cards').innerHTML = `<div class="empty">加载失败: ${e.message}</div>`;
    document.getElementById('quality-trend-tbody').innerHTML = `<tr><td colspan="5" class="empty">加载失败: ${e.message}</td></tr>`;
    document.getElementById('quality-samples-tbody').innerHTML = `<tr><td colspan="4" class="empty">加载失败: ${e.message}</td></tr>`;
    document.getElementById('quality-total').textContent = '';
  }
}

function toggleQualityDetail(i) {
  const el = document.getElementById('quality-detail-' + i);
  if (el) el.classList.toggle('open');
}

function clearQualityFilters() {
  document.getElementById('quality-role-filter').value = '';
  document.getElementById('quality-date-from').value = '';
  document.getElementById('quality-date-to').value = '';
  loadQualityTab();
}

function loadAll() {
  loadStats();
  loadQualityTab();
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
