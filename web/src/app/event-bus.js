/**
 * 极简事件总线：模块间解耦通信
 *
 * 关键事件：
 *   ble:line              单行 BLE 数据到达
 *   ble:vital-updated     心率/呼吸更新
 *   ble:state-change      连接/断开
 *   recording:start|stop  录制开始/结束
 *   workspace:change      切换 workspace
 *   chart:reset           重置图表
 *   ui:toast              触发 toast
 */

class EventBus {
  constructor() { this._handlers = new Map(); }

  on(event, handler) {
    if (!this._handlers.has(event)) this._handlers.set(event, new Set());
    this._handlers.get(event).add(handler);
    return () => this.off(event, handler);
  }

  once(event, handler) {
    const off = this.on(event, (...args) => { off(); handler(...args); });
    return off;
  }

  off(event, handler) {
    const set = this._handlers.get(event);
    if (set) set.delete(handler);
  }

  emit(event, ...args) {
    const set = this._handlers.get(event);
    if (!set) return;
    for (const h of set) {
      try { h(...args); } catch (e) { console.error(`[bus] handler for "${event}" threw:`, e); }
    }
  }
}

export const bus = new EventBus();
