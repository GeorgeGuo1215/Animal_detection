/**
 * Markdown 安全渲染 + Agent 状态格式化。
 */

const STATUS_ICON = {
  thinking: '🤔', planning: '📋', plan_complete: '✅',
  tool_calling: '🔍', tool_complete: '✅',
  decided_final: '💡', generating: '💭',
  user_action_needed: '⚠️',
  routing: '🧭', expert_calling: '👨‍⚕️', expert_complete: '✅', reviewing: '🛡️',
};

export function formatChatMessage(text) {
  if (!text) return '';
  try {
    const hasMarked = typeof window.marked !== 'undefined' && typeof window.marked.parse === 'function';
    const hasPurify = typeof window.DOMPurify !== 'undefined' && typeof window.DOMPurify.sanitize === 'function';
    if (hasMarked) {
      const raw = window.marked.parse(String(text), { gfm: true, breaks: true });
      if (hasPurify) {
        return window.DOMPurify.sanitize(raw, {
          ALLOWED_TAGS: ['p','br','strong','em','del','ul','ol','li','blockquote','pre','code','hr','a','h1','h2','h3','h4','h5','h6','table','thead','tbody','tr','th','td'],
          ALLOWED_ATTR: ['href','title','target','rel'],
        });
      }
    }
  } catch {}
  return escapeHtml(text).replace(/\n/g, '<br/>');
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

export function formatAgentStatus(status, detail = {}) {
  const icon = STATUS_ICON[status] || '⚙️';
  let text = '';
  switch (status) {
    case 'thinking':       text = `思考中…（第 ${detail?.round || 1} 轮）`; break;
    case 'planning':       text = '正在制定计划…'; break;
    case 'plan_complete':  text = '计划制定完成'; break;
    case 'tool_calling':   text = `第 ${detail?.round || 1} 轮工具调用：${detail?.tool_name || 'rag.search'}`; break;
    case 'tool_complete':  text = formatToolComplete(detail); break;
    case 'decided_final':
      text = `决定生成回答${detail?.reason ? `（${String(detail.reason).slice(0, 50)}…）` : ''}`;
      break;
    case 'generating':     text = '正在生成回答…'; break;
    case 'user_action_needed':
      text = (detail && (detail.message || detail.hint)) || '需要补充信息';
      break;
    case 'routing':        text = formatRouting(detail); break;
    case 'expert_calling':
      text = `${detail?.name_zh || '专家'} 会诊中${typeof detail?.weight === 'number' ? `（权重 ${detail.weight.toFixed(2)}）` : ''}`;
      break;
    case 'expert_complete':
      text = `${detail?.name_zh || '专家'} 意见已出${typeof detail?.confidence === 'number' ? `（置信度 ${detail.confidence.toFixed(2)}）` : ''}`;
      break;
    case 'reviewing':
      text = detail?.verdict ? `边界审核完成：${formatVerdict(detail.verdict)}` : '边界审核中…';
      break;
    default:
      text = (detail && typeof detail.message === 'string' && detail.message) ? detail.message : String(status);
  }
  return { icon, text, status };
}

function formatRouting(detail) {
  if (!detail || typeof detail !== 'object') return '正在分诊与路由…';
  if (detail.out_of_scope) return '判定与宠物健康无关，已拒答';
  const experts = Array.isArray(detail.selected_experts) ? detail.selected_experts : null;
  if (experts && experts.length) {
    const tag = detail.emergency ? '⚠️急症 · ' : '';
    return `${tag}路由完成，激活 ${experts.length} 位专家`;
  }
  return '正在分诊与路由…';
}

function formatVerdict(verdict) {
  switch (String(verdict)) {
    case 'pass':   return '通过';
    case 'revise': return '需修订（已加约束）';
    case 'block':  return '拦截（安全兜底）';
    default:       return String(verdict);
  }
}

function formatToolComplete(detail) {
  if (!detail || typeof detail !== 'object') return '工具执行完成';
  const tn = String(detail.tool_name || '');
  const hits = typeof detail.hits_count === 'number' ? detail.hits_count : null;
  const rows = typeof detail.row_count === 'number' ? detail.row_count : null;
  if (tn === 'sql.search' || rows != null) {
    return rows === 0 ? '数据库查询完成（0 条记录）' : `数据库查询完成（${rows} 条记录）`;
  }
  if (tn.startsWith('mcp.web_search') || tn.includes('web_search')) {
    return hits === 0 ? '联网检索完成（0 条结果）' : `联网检索完成（${hits ?? 0} 条结果）`;
  }
  if (tn === 'rag.search' || tn.includes('rag')) {
    return hits === 0 ? '知识库检索完成（0 条匹配）' : `知识库找到 ${hits ?? 0} 条相关信息`;
  }
  if (hits != null) return hits === 0 ? `${tn || '工具'} 完成（0 条结果）` : `${tn || '工具'} 完成（${hits} 条）`;
  if (rows != null) return `${tn || '工具'} 完成（${rows} 行）`;
  return `${tn || '工具'} 已执行`;
}

export function stripAnswerPlaceholders(raw) {
  if (!raw) return '';
  return String(raw)
    .replace(/\*\*Generating response\.\.\.\*\*/gi, '')
    .replace(/\*\*Generating final answer\*\*[^\n]*/gi, '')
    .trim();
}
