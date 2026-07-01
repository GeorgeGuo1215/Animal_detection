/**
 * 黑白主题：覆盖 Chart.js 默认配色 / 字体 / 网格。
 *
 * 调用 applyChartTheme() 一次即可（依赖 window.Chart）。
 */

const PALETTE = {
  primary: '#1D1D1F',
  primaryAlt: '#2C2C2E',
  muted: '#8E8E93',
  faint: '#E5E5EA',
  grid: 'rgba(0,0,0,0.06)',
  text: '#1D1D1F',
  textMuted: '#6E6E73',
  bg: '#FFFFFF',
};

export const SERIES_COLORS = [
  '#1D1D1F', '#6E6E73', '#A1A1A6',
  '#3A3A3C', '#86868B', '#48484A',
];

export function applyChartTheme() {
  if (typeof window.Chart === 'undefined') return;
  const Chart = window.Chart;

  Chart.defaults.color = PALETTE.text;
  Chart.defaults.font.family =
    '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "PingFang SC", system-ui, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.borderColor = PALETTE.faint;

  if (Chart.defaults.scale) {
    Chart.defaults.scale.grid.color = PALETTE.grid;
    Chart.defaults.scale.ticks.color = PALETTE.textMuted;
  }
  if (Chart.defaults.scales) {
    for (const k of Object.keys(Chart.defaults.scales)) {
      const s = Chart.defaults.scales[k];
      if (s.grid)   s.grid.color = PALETTE.grid;
      if (s.ticks)  s.ticks.color = PALETTE.textMuted;
      if (s.title)  s.title.color = PALETTE.textMuted;
    }
  }
  Chart.defaults.plugins = Chart.defaults.plugins || {};
  Chart.defaults.plugins.legend = Chart.defaults.plugins.legend || {};
  Chart.defaults.plugins.legend.labels = {
    ...Chart.defaults.plugins.legend.labels,
    color: PALETTE.text,
    boxWidth: 12,
    boxHeight: 4,
  };
  Chart.defaults.plugins.tooltip = {
    ...Chart.defaults.plugins.tooltip,
    backgroundColor: 'rgba(29,29,31,0.92)',
    borderColor: 'rgba(255,255,255,0.06)',
    borderWidth: 1,
    titleColor: '#FFFFFF',
    bodyColor: '#F5F5F7',
    cornerRadius: 8,
    padding: 10,
    titleFont: { weight: '600' },
  };
  // 关闭整体动画以提升实时性
  Chart.defaults.animation = false;
  Chart.defaults.responsive = true;
  Chart.defaults.maintainAspectRatio = false;
}

export const COLORS = PALETTE;
