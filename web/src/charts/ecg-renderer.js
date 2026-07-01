/**
 * ECG canvas 播放器：呼吸 / 心跳两路实时画布。
 *
 * 用于 BLE 与 文件分析两种数据源。
 */

import { COLORS } from './chart-theme.js';

function makeTrack(canvas) {
  return { canvas, ctx: canvas.getContext('2d'), data: [], playing: false, cursor: 0 };
}

export function createECGPlayer(resCanvas, hbCanvas, opts = {}) {
  if (!resCanvas || !hbCanvas) return null;
  const player = {
    res: makeTrack(resCanvas),
    hb:  makeTrack(hbCanvas),
    raf: null,
    showGrid: opts.showGrid ?? true,
    drawCount: 0,
  };

  function drawTrack(track, color) {
    const { canvas, ctx, data, cursor } = track;
    const w = canvas.width = canvas.clientWidth || 600;
    const h = canvas.height = canvas.clientHeight || 160;
    ctx.clearRect(0, 0, w, h);

    if (player.showGrid) {
      ctx.strokeStyle = 'rgba(0,0,0,0.06)';
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 24) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
      for (let y = 0; y < h; y += 24) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    }

    if (data.length === 0) return;
    ctx.strokeStyle = color || COLORS.text;
    ctx.lineWidth = 1.6;
    ctx.beginPath();

    if (opts.mode === 'cursor') {
      const display = Math.min(200, data.length);
      const startIdx = Math.max(0, cursor - display);
      for (let i = 0; i < display && startIdx + i < data.length; i++) {
        const x = (i / display) * w;
        const y = h / 2 - (data[startIdx + i] || 0) * h * 0.25;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
    } else {
      const view = 1000;
      const start = Math.max(0, data.length - view);
      for (let i = start; i < data.length; i++) {
        const x = (i - start) / view * w;
        const y = h / 2 - (data[i] || 0) * h * 0.4;
        if (i === start) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }

  player.draw = () => {
    drawTrack(player.res, COLORS.text);
    drawTrack(player.hb,  COLORS.text);
    if (opts.mode === 'cursor') {
      if (player.res.playing || player.hb.playing) {
        if (player.res.data.length) player.res.cursor = (player.res.cursor + 1) % player.res.data.length;
        if (player.hb.data.length)  player.hb.cursor  = (player.hb.cursor  + 1) % player.hb.data.length;
      }
    }
    if (player.res.playing || player.hb.playing) {
      player.raf = requestAnimationFrame(player.draw);
    } else if (player.raf) {
      cancelAnimationFrame(player.raf);
      player.raf = null;
    }
  };

  player.play = () => {
    player.res.playing = true;
    player.hb.playing = true;
    if (!player.raf) player.draw();
  };
  player.pause = () => {
    player.res.playing = false;
    player.hb.playing = false;
  };
  player.reset = () => {
    player.res.data = [];
    player.hb.data = [];
    player.res.cursor = 0;
    player.hb.cursor = 0;
    player.draw();
  };
  player.draw();
  return player;
}
