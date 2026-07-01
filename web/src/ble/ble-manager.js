/**
 * Web Bluetooth 管理器（ESM 版本）
 *
 * 功能：
 *  - 请求设备并连接
 *  - 优先尝试 Nordic UART (NUS)；失败则发现并订阅所有可通知特征
 *  - 把字节按 UTF-8 累积成行，逐行回调 onLine
 *  - 提供 send / disconnect
 */

import { child } from '../utils/logger.js';

const log = child('BLE');

class BluetoothManager {
  constructor() {
    this.device = null;
    this.server = null;
    this.notifyCharacteristic = null;
    this.writeCharacteristic = null;
    this.decoder = new TextDecoder('utf-8');
    this._rxBuffer = '';
    this.onConnect = null;
    this.onDisconnect = null;
    this.onLine = null;
    this.onError = null;
    this.onServiceDiscovered = null;
  }

  async connect() {
    if (!navigator.bluetooth) {
      this._emitError(new Error('当前浏览器不支持 Web Bluetooth'));
      return;
    }
    try {
      this.device = await navigator.bluetooth.requestDevice({
        acceptAllDevices: true,
        optionalServices: [
          '6e400001-b5a3-f393-e0a9-e50e24dcca9e',
          0x180D, 0x180F, 0x1800, 0x1801, 0x180A,
          0xFFE0, 0xFFF0,
          '0000ffe0-0000-1000-8000-00805f9b34fb',
          '0000fff0-0000-1000-8000-00805f9b34fb',
        ],
      });

      this.device.addEventListener('gattserverdisconnected', this._handleDisconnect.bind(this));
      this.server = await this.device.gatt.connect();

      try {
        await this._setupNUS('6e400001-b5a3-f393-e0a9-e50e24dcca9e');
        this._emitServiceDiscovered('使用 Nordic UART Service (NUS) 协议');
      } catch (_) {
        await this._setupAllNotifiableCharacteristics();
      }

      this._emitConnect();
    } catch (err) {
      this._emitError(err);
      await this.disconnect();
    }
  }

  async _setupNUS(serviceUuid) {
    const service = await this.server.getPrimaryService(serviceUuid);
    const txUuid = '6e400003-b5a3-f393-e0a9-e50e24dcca9e';
    const rxUuid = '6e400002-b5a3-f393-e0a9-e50e24dcca9e';
    this.notifyCharacteristic = await service.getCharacteristic(txUuid);
    this.writeCharacteristic = await service.getCharacteristic(rxUuid);
    await this._startNotifications(this.notifyCharacteristic);
  }

  async _setupAllNotifiableCharacteristics() {
    const services = await this.server.getPrimaryServices();
    let notifiableCount = 0, writableCount = 0;
    this._emitServiceDiscovered(`发现 ${services.length} 个服务，开始扫描特征…`);

    for (const service of services) {
      try {
        const characteristics = await service.getCharacteristics();
        const sUuid = this._formatUuid(service.uuid);
        this._emitServiceDiscovered(`服务 ${sUuid}: ${characteristics.length} 个特征`);
        for (const ch of characteristics) {
          const props = ch.properties;
          if (props.notify || props.indicate) {
            if (!this.notifyCharacteristic) this.notifyCharacteristic = ch;
            await this._startNotifications(ch);
            notifiableCount++;
          }
          if (props.write || props.writeWithoutResponse) {
            if (!this.writeCharacteristic) this.writeCharacteristic = ch;
            writableCount++;
          }
        }
      } catch (e) {
        this._emitServiceDiscovered(`服务 ${this._formatUuid(service.uuid)} 访问失败: ${e.message}`);
      }
    }
    if (notifiableCount === 0) throw new Error('未找到可通知 (Notify/Indicate) 的特征');
    this._emitServiceDiscovered(`完成扫描：${notifiableCount} 个可通知特征，${writableCount} 个可写特征`);
  }

  async _startNotifications(characteristic = null) {
    const ch = characteristic || this.notifyCharacteristic;
    if (!ch) throw new Error('缺少通知特征');
    await ch.startNotifications();
    ch.addEventListener('characteristicvaluechanged', (event) => {
      const value = event.target.value;
      const chunk = this.decoder.decode(value);
      this._handleIncomingText(chunk);
    });
    log.debug(`已启动特征 ${ch.uuid} 的通知`);
  }

  _handleIncomingText(text) {
    this._rxBuffer += text;
    let idx;
    while ((idx = this._rxBuffer.indexOf('\n')) >= 0) {
      let line = this._rxBuffer.slice(0, idx);
      this._rxBuffer = this._rxBuffer.slice(idx + 1);
      line = line.replace(/[\r\n]+/g, '').trim();
      if (line && typeof this.onLine === 'function') {
        try { this.onLine(line); } catch (e) { log.error('onLine callback error:', e); }
      }
    }
  }

  async send(text) {
    if (!this.writeCharacteristic) throw new Error('该设备不支持写入或未发现写入特征');
    await this.writeCharacteristic.writeValue(new TextEncoder().encode(text));
  }

  async disconnect() {
    try {
      if (this.notifyCharacteristic) {
        try { await this.notifyCharacteristic.stopNotifications(); } catch (_) {}
        this.notifyCharacteristic = null;
      }
      if (this.device && this.device.gatt && this.device.gatt.connected) {
        await this.device.gatt.disconnect();
      }
    } finally {
      this.server = null;
      this.device = null;
      this._emitDisconnect();
    }
  }

  _handleDisconnect() {
    this.server = null;
    this.notifyCharacteristic = null;
    this.writeCharacteristic = null;
    this._emitDisconnect();
  }

  _emitConnect()    { try { this.onConnect && this.onConnect(this.device); } catch (e) { log.error('onConnect:', e); } }
  _emitDisconnect() { try { this.onDisconnect && this.onDisconnect(); } catch (e) { log.error('onDisconnect:', e); } }
  _emitError(err)   { try { this.onError && this.onError(err); } catch (_) {} log.error(err); }
  _emitServiceDiscovered(info) {
    try { this.onServiceDiscovered && this.onServiceDiscovered(info); } catch (_) {}
    log.debug(info);
  }

  _formatUuid(uuid) {
    if (typeof uuid === 'number') return `0x${uuid.toString(16).toUpperCase().padStart(4, '0')}`;
    if (typeof uuid === 'string' && uuid.length === 36) {
      return uuid.startsWith('0000') && uuid.endsWith('-0000-1000-8000-00805f9b34fb')
        ? `0x${uuid.substring(4, 8).toUpperCase()}`
        : `${uuid.substring(0, 8)}…`;
    }
    return uuid;
  }
}

const ble = new BluetoothManager();

export default ble;
export { BluetoothManager, ble };

if (typeof window !== 'undefined') {
  window.BluetoothManager = BluetoothManager;
  window.BLE = ble;
}
