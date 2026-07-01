/**
 * 分级日志：默认仅 warn / error 输出。?debug=1 或 localStorage.PETMIND_DEBUG=1 打开 debug。
 */

const LEVELS = { silent: 0, error: 1, warn: 2, info: 3, debug: 4 };

function detectLevel() {
  try {
    const params = new URLSearchParams(location.search);
    if (params.has('debug')) return 'debug';
    const stored = localStorage.getItem('PETMIND_DEBUG');
    if (stored === '1' || stored === 'true') return 'debug';
  } catch {}
  return 'warn';
}

const current = detectLevel();
const cur = LEVELS[current] ?? LEVELS.warn;

function log(level, ...args) {
  if (LEVELS[level] > cur) return;
  const fn = console[level === 'debug' ? 'log' : level] || console.log;
  fn.call(console, ...args);
}

export const logger = {
  debug: (...a) => log('debug', '[D]', ...a),
  info:  (...a) => log('info',  '[I]', ...a),
  warn:  (...a) => log('warn',  '[W]', ...a),
  error: (...a) => log('error', '[E]', ...a),
  level: current,
};

/** 简单创建带前缀的子 logger */
export function child(prefix) {
  return {
    debug: (...a) => logger.debug(`[${prefix}]`, ...a),
    info:  (...a) => logger.info(`[${prefix}]`, ...a),
    warn:  (...a) => logger.warn(`[${prefix}]`, ...a),
    error: (...a) => logger.error(`[${prefix}]`, ...a),
  };
}
