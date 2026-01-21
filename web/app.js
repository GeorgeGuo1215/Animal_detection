/**
 * 毫米波雷达数据处理Web应用主控制器
 */

class RadarWebApp {
    constructor() {
        // 采样率固定为50Hz，与串口接收频率一致
        const samplingRate = 50;
        this.processor = new RadarDataProcessor(samplingRate);
        this.selectedFiles = [];
        this.processedResults = [];
        this.charts = {}; // 文件数据图表
        
        // 蓝牙数据相关
        this.bleConnected = false;
        this.bleCharts = {}; // 蓝牙数据图表
        this.bleBufferI = [];
        this.bleBufferQ = [];

        // 自适应Y轴相关属性
        this.adaptiveYAxisEnabled = true; // 启用自适应Y轴以放大显示微小变化
        this.adaptiveSampleCount = 0; // 已收集的样本数量
        this.adaptiveStabilizeThreshold = 30; // 稳定前需要的样本数（降低阈值以更快响应）
        this.adaptiveStabilizeWindow = 50; // 检测稳定的窗口大小
        this.adaptiveLastMinI = Infinity;
        this.adaptiveLastMaxI = -Infinity;
        this.adaptiveLastMinQ = Infinity;
        this.adaptiveLastMaxQ = -Infinity;
        this.adaptiveStabilized = false; // 是否已稳定
        // IMU(默认存陀螺仪)三轴缓存：gx/gy/gz
        this.bleBufferIMU_X = [];
        this.bleBufferIMU_Y = [];
        this.bleBufferIMU_Z = [];
        this.bleBufferTemperature = []; // 温度数据缓存
        this.bleBufferTimestamps = [];
        this.bleMaxBuffer = 5000; // 逻辑窗口长度
        // 避免每条数据都 splice(0,1) 造成 O(n) 内存搬移：允许轻微超出，超出后一次性裁剪
        this.bleMaxBufferHard = this.bleMaxBuffer + 200;
        this.blePendingFloat = null; // 仅有单个浮点时用于配对
        this.bleDataCount = 0;
        this.bleConnectStartTime = null;
        this.bleConnectTimer = null;
        this.lastBleRxTs = 0;
        this.rxWatchdogTimer = null;
        this._simInterval = null;

        // ===== 心率稳定机制（参考main.py第48-51行）=====
        this.heartRateHistory = new Array(200).fill(70);  // 固定200个心率历史记录，与Python端一致
        this.respiratoryHistory = new Array(200).fill(18); // 固定200个呼吸频率历史记录
        this.historyIndex = 0;  // 循环数组索引
        this.historyMaxLength = 200;  // 固定200个历史值
        this.heartRateDelta = 5;    // 心率最大变化幅度（bpm）参考main.py第51行
        this.lastStableHeartRate = 70; // 上次稳定的心率
        this.lastStableRespRate = 18;  // 上次稳定的呼吸频率

        // ===== 丢包/采样率统计（估算）=====
        // 说明：若设备每条数据=1个采样点，则可根据到达间隔估算丢包；
        // 若未来协议携带 seq，则可切换为 seq 更精准统计。
        this.bleStats = {
            startRxTs: 0,
            lastRxTs: 0,
            received: 0,
            expected: 0,
            missed: 0,
            // 抖动统计（到达间隔）
            lastGapMs: 0,
            gapEmaMs: 0,
            gapJitterEmaMs: 0,
            // seq（可选）
            lastSeq: null,
            seqBased: false
        };

        // ===== 性能优化：日志/图表节流 =====
        this._bleLogLines = [];
        this._bleLogRenderTimer = null;
        this._bleRawLines = [];
        this._bleRawRenderTimer = null;

        this._bleChartRaf = null;
        this._bleChartLastUpdateTs = 0;
        this._bleChartMinIntervalMs = 100; // 10Hz 刷新图表足够流畅

        this._bleVitalLogLastTs = 0; // 限制生理参数日志刷屏
        
        // 实时保存相关 (参考main.py)
        this.bleRecordingFlag = 0;  // 0: 不记录, 1: 记录中
        this.bleRecordingData = []; // 记录的处理后数据缓存
        this.bleRecordingRawData = []; // 记录的原始蓝牙数据缓存
        this.bleRecordingStartTime = null;

        // ===== BLE 上报到 Integration =====
        this.bleUploadEnabled = false;
        this.bleUploadIntervalSec = 10;
        this.bleUploadWindowSec = 10;
        this.bleUploadTimer = null;
        this.bleLastUploadTs = 0;
        
        // 当前心率和呼吸率（供静息监测模块使用）
        this.currentHeartRate = null;
        this.currentRespiratoryRate = null;
        
        this.initializeEventListeners();
        this.initBleUploadConfig();
        this.initializeCharts();
        this.initializeBluetoothCharts();
        this.initializeBLEECG();
        this.initializeFileECG();
        this.initializeHealthChat();

        // 初始化BLE事件
        this.initializeBLE();
        
        // 测试FFT是否正常工作
        this.testFFT();

        // 启动接收看门狗：若长时间无数据则判定断连
        this.startRxWatchdog();
    }

    /**
     * 初始化健康对话设置
     */
    initializeHealthChat() {
        const chatAgentEndpointEl = document.getElementById('chatAgentEndpoint');
        if (chatAgentEndpointEl) {
            chatAgentEndpointEl.value = localStorage.getItem('chatAgentEndpoint') || 'http://localhost:9001';
        }

        // 添加回车发送消息功能
        const chatInputEl = document.getElementById('chatInput');
        if (chatInputEl) {
            chatInputEl.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (!document.getElementById('sendChatBtn').disabled) {
                        this.sendChatMessage();
                    }
                }
            });
        }
    }

    /**
     * 测试FFT功能
     */
    testFFT() {
        try {
            console.log('🔍 测试FFT功能...');
            
            if (typeof FFT === 'undefined') {
                console.error('❌ FFT对象未定义！');
                return;
            }
            
            // 创建测试信号: 10Hz + 30Hz 正弦波
            const testData = [];
            const fs = (this.processor && Number.isFinite(this.processor.fs)) ? this.processor.fs : 100;
            for (let i = 0; i < 256; i++) {
                const t = i / fs;
                const signal = Math.sin(2 * Math.PI * 10 * t) + 0.5 * Math.sin(2 * Math.PI * 30 * t);
                testData.push([signal, 0]); // 复数格式
            }
            
            const fftResult = FFT.fft(testData);
            const magnitude = fftResult.map(([real, imag]) => Math.sqrt(real * real + imag * imag));
            
            // 找到峰值
            const peakIdx1 = magnitude.slice(0, 128).indexOf(Math.max(...magnitude.slice(0, 128)));
            const peakIdx2 = magnitude.slice(peakIdx1 + 5, 128).indexOf(Math.max(...magnitude.slice(peakIdx1 + 5, 128))) + peakIdx1 + 5;
            
            const freq1 = peakIdx1 * fs / 256;
            const freq2 = peakIdx2 * fs / 256;
            
            console.log(`✅ FFT测试成功！检测到峰值频率: ${freq1.toFixed(1)}Hz, ${freq2.toFixed(1)}Hz (期望: 10Hz, 30Hz)`);
            
        } catch (error) {
            console.error('❌ FFT测试失败:', error);
        }
    }

    /**
     * 初始化事件监听器
     */
    initializeEventListeners() {
        // 文件上传相关
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');

        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileSelect({ target: { files: e.dataTransfer.files } });
        });

        // 设置面板
        const settingsToggle = document.querySelector('.settings-toggle');
        settingsToggle.addEventListener('click', () => this.toggleSettings());
    }

    initBleUploadConfig() {
        const urlEl = document.getElementById('bleUploadUrl');
        const animalEl = document.getElementById('bleAnimalId');
        const deviceEl = document.getElementById('bleDeviceId');
        const intervalEl = document.getElementById('bleUploadInterval');

        if (urlEl) {
            urlEl.value = localStorage.getItem('bleUploadUrl') || 'http://127.0.0.1:9001/ingest';
        }
        if (animalEl) {
            animalEl.value = localStorage.getItem('bleAnimalId') || '';
        }
        if (deviceEl) {
            deviceEl.value = localStorage.getItem('bleDeviceId') || '';
        }
        if (intervalEl) {
            intervalEl.value = localStorage.getItem('bleUploadInterval') || String(this.bleUploadIntervalSec);
        }
    }

    /**
     * 初始化 BLE 事件
     */
    initializeBLE() {
        if (!window.BLE) return;
        BLE.onConnect = (device) => {
            this.bleConnected = true;
            this.addBLELog(`✓ 已连接: ${device.name || '未知设备'} (${device.id})`);
            this.addBLELog(`📡 正在扫描可用服务和特征...`);
            
            // 显示实时数据区域
            document.getElementById('bleRealTimeData').style.display = 'block';
            
            // 开始计时
            this.bleConnectStartTime = Date.now();
            this.startBluetoothTimer();
            
            // 重置数据
            this.resetBluetoothData();
            
            this.updateBLEButtons();

            // 连接后自动展开蓝牙图表（避免用户觉得“没有gx/gy/gz可视图”）
            const chartsSection = document.getElementById('bluetoothChartsSection');
            if (chartsSection) {
                chartsSection.style.display = 'block';
                console.log('✅ 蓝牙图表区域已展开');
            }

            // 确保图表已初始化
            if (!this.bleCharts.iSignal || !this.bleCharts.qSignal) {
                console.log('🔄 重新初始化蓝牙图表...');
                this.initializeBluetoothCharts();
            }

            // 触发一次 resize/update，解决 display:none 时 Chart.js 尺寸为0的问题
            setTimeout(() => {
                try {
                    console.log('📊 刷新所有蓝牙图表...');
                    Object.values(this.bleCharts || {}).forEach(ch => {
                        if (ch && typeof ch.resize === 'function') ch.resize();
                        if (ch && typeof ch.update === 'function') ch.update('none');
                    });
                } catch (error) {
                    console.error('❌ 图表刷新失败:', error);
                }
            }, 100);

        // 自动初始化并启动蓝牙ECG播放
        this.initializeBLEECG();

        // 调试：强制检查并重新初始化图表（如果需要）
        setTimeout(() => {
            this.forceReinitializeCharts();
        }, 200);
            const playBtn = document.getElementById('blePlayBtn');
            const pauseBtn = document.getElementById('blePauseBtn');
            if (this._bleECG) {
                this._bleECG.res.playing = true;
                this._bleECG.hb.playing = true;
                if (playBtn && pauseBtn) { playBtn.style.display = 'none'; pauseBtn.style.display = 'inline-block'; }
                if (!this._bleECG.raf) this._bleECG.draw();
            }
        };
        BLE.onDisconnect = () => {
            this.bleConnected = false;
            this.addBLELog('⚠️ 已断开连接');
            
            // 隐藏实时数据区域
            document.getElementById('bleRealTimeData').style.display = 'none';
            
            // 停止计时
            this.stopBluetoothTimer();
            // 停止任何模拟数据
            this.stopSimulation();

            // 断开后停止上报
            this.stopBleUpload();
            
            this.updateBLEButtons();
        };
        BLE.onError = (err) => {
            this.addBLELog(`❌ 错误: ${err.message}`);
        };
        BLE.onServiceDiscovered = (info) => {
            this.addBLELog(info);
        };
        BLE.onLine = (line) => this.handleBLELine(line);
        this.updateBLEButtons();
    }

    updateBLEButtons() {
        const c = document.getElementById('bleConnectBtn');
        const d = document.getElementById('bleDisconnectBtn');
        const s = document.getElementById('bleShowChartsBtn');
        const diagBtn = document.getElementById('bleDiagBtn');
        const startBtn = document.getElementById('bleStartRecordBtn');
        const stopBtn = document.getElementById('bleStopRecordBtn');
        const azureBtn = document.getElementById('bleAzureBtn');
        const uploadStartBtn = document.getElementById('bleStartUploadBtn');
        const uploadStopBtn = document.getElementById('bleStopUploadBtn');
        if (!c || !d || !s || !startBtn || !stopBtn || !azureBtn) return;
        
        c.style.display = this.bleConnected ? 'none' : 'inline-block';
        d.style.display = this.bleConnected ? 'inline-block' : 'none';
        s.style.display = this.bleConnected ? 'inline-block' : 'none';
        if (diagBtn) diagBtn.style.display = this.bleConnected ? 'inline-block' : 'none';
        
        // 录制按钮分离（开始/结束）
        if (this.bleConnected) {
            startBtn.style.display = this.bleRecordingFlag === 1 ? 'none' : 'inline-block';
            stopBtn.style.display = this.bleRecordingFlag === 1 ? 'inline-block' : 'none';
            azureBtn.style.display = 'inline-block';
            if (uploadStartBtn && uploadStopBtn) {
                uploadStartBtn.style.display = this.bleUploadEnabled ? 'none' : 'inline-block';
                uploadStopBtn.style.display = this.bleUploadEnabled ? 'inline-block' : 'none';
            }
        } else {
            startBtn.style.display = 'none';
            stopBtn.style.display = 'none';
            azureBtn.style.display = 'none';
            if (uploadStartBtn && uploadStopBtn) {
                uploadStartBtn.style.display = 'none';
                uploadStopBtn.style.display = 'none';
            }
        }
        
        // 静息监测按钮（独立模块）
        const restingStartBtn = document.getElementById('restingStartBtn');
        const restingStopBtn = document.getElementById('restingStopBtn');
        const restingSaveBtn = document.getElementById('restingSaveBtn');
        const restingConfigBtn = document.getElementById('restingConfigBtn');
        const restingClearBtn = document.getElementById('restingClearBtn');
        
        if (restingStartBtn) {
            restingStartBtn.style.display = this.bleConnected ? 'inline-block' : 'none';
        }
        if (restingStopBtn) {
            // 由静息监测模块自己控制显示
        }
        if (restingSaveBtn) {
            restingSaveBtn.style.display = this.bleConnected ? 'inline-block' : 'none';
        }
        if (restingConfigBtn) {
            restingConfigBtn.style.display = this.bleConnected ? 'inline-block' : 'none';
        }
        if (restingClearBtn) {
            restingClearBtn.style.display = this.bleConnected ? 'inline-block' : 'none';
        }
    }

    _setBleUploadStatus(text) {
        const statusEl = document.getElementById('bleUploadStatus');
        if (statusEl) statusEl.textContent = text;
    }

    _getBleUploadConfig() {
        const urlEl = document.getElementById('bleUploadUrl');
        const animalEl = document.getElementById('bleAnimalId');
        const deviceEl = document.getElementById('bleDeviceId');
        const intervalEl = document.getElementById('bleUploadInterval');

        const url = urlEl ? urlEl.value.trim() : '';
        const animalId = animalEl ? animalEl.value.trim() : '';
        const deviceId = deviceEl ? deviceEl.value.trim() : '';
        const intervalSec = intervalEl ? parseInt(intervalEl.value, 10) : this.bleUploadIntervalSec;

        return {
            url,
            animalId,
            deviceId,
            intervalSec: Number.isFinite(intervalSec) && intervalSec > 0 ? intervalSec : this.bleUploadIntervalSec
        };
    }

    startBleUpload() {
        if (!this.bleConnected) {
            alert('请先连接蓝牙设备');
            return;
        }
        const cfg = this._getBleUploadConfig();
        if (!cfg.url) {
            alert('请填写上报接口地址');
            return;
        }
        if (!cfg.animalId) {
            alert('请填写 animal_id');
            return;
        }
        if (!cfg.deviceId) {
            alert('请填写 device_id');
            return;
        }

        localStorage.setItem('bleUploadUrl', cfg.url);
        localStorage.setItem('bleAnimalId', cfg.animalId);
        localStorage.setItem('bleDeviceId', cfg.deviceId);
        localStorage.setItem('bleUploadInterval', String(cfg.intervalSec));

        this.bleUploadEnabled = true;
        this.bleUploadIntervalSec = cfg.intervalSec;
        this._setBleUploadStatus('上传中');
        this.updateBLEButtons();

        if (this.bleUploadTimer) clearInterval(this.bleUploadTimer);
        this._sendBleUploadOnce();
        this.bleUploadTimer = setInterval(() => this._sendBleUploadOnce(), this.bleUploadIntervalSec * 1000);
    }

    stopBleUpload() {
        this.bleUploadEnabled = false;
        if (this.bleUploadTimer) {
            clearInterval(this.bleUploadTimer);
            this.bleUploadTimer = null;
        }
        this._setBleUploadStatus('未上传');
        this.updateBLEButtons();
    }

    _toEpochMs(ts) {
        if (Number.isFinite(ts)) return Number(ts);
        if (typeof ts === 'string') {
            const parsed = Date.parse(ts);
            if (!Number.isNaN(parsed)) return parsed;
        }
        return Date.now();
    }

    _formatTimezoneOffset() {
        const offsetMin = -new Date().getTimezoneOffset();
        const sign = offsetMin >= 0 ? '+' : '-';
        const abs = Math.abs(offsetMin);
        const hh = String(Math.floor(abs / 60)).padStart(2, '0');
        const mm = String(abs % 60).padStart(2, '0');
        return `${sign}${hh}:${mm}`;
    }

    _buildBleEventPayload() {
        const cfg = this._getBleUploadConfig();
        const fs = (this.processor && Number.isFinite(this.processor.fs)) ? this.processor.fs : 50;
        const len = this.bleBufferI.length;
        if (len < Math.max(10, fs * 2)) {
            this.addBLELog('⚠️ 上报跳过：数据点不足');
            return null;
        }

        const windowSize = Math.min(len, Math.max(10, fs * this.bleUploadWindowSec));
        const startIndex = len - windowSize;
        const endIndex = len - 1;

        const startTsMs = this._toEpochMs(this.bleBufferTimestamps[startIndex]);
        const endTsMs = this._toEpochMs(this.bleBufferTimestamps[endIndex]);
        const timezone = this._formatTimezoneOffset();

        const accelSamples = [];
        const tempSamples = [];
        let lastTempSecond = -1;
        for (let i = startIndex; i <= endIndex; i++) {
            const tMs = Math.round(((i - startIndex) / fs) * 1000);
            const tS = Math.floor((i - startIndex) / fs);
            accelSamples.push({
                t_ms: tMs,
                x: Number(this.bleBufferIMU_X[i] || 0),
                y: Number(this.bleBufferIMU_Y[i] || 0),
                z: Number(this.bleBufferIMU_Z[i] || 0)
            });
            if (tS !== lastTempSecond) {
                tempSamples.push({
                    t_s: tS,
                    value: Number(this.bleBufferTemperature[i] || 0)
                });
                lastTempSecond = tS;
            }
        }

        const vitalsSamples = [];
        if (Number.isFinite(this.currentHeartRate) || Number.isFinite(this.currentRespiratoryRate)) {
            vitalsSamples.push({
                t_s: 0,
                hr: Number.isFinite(this.currentHeartRate) ? Number(this.currentHeartRate) : null,
                rr: Number.isFinite(this.currentRespiratoryRate) ? Number(this.currentRespiratoryRate) : null
            });
        }

        return {
            event_id: `ble_${Date.now()}`,
            ts: new Date(endTsMs).toISOString(),
            animal: {
                animal_id: cfg.animalId,
                species: 'other',
                name: 'unknown',
                breed: 'unknown',
                sex: 'unknown',
                age_months: 0,
                weight_kg: 0
            },
            device: {
                device_id: cfg.deviceId,
                firmware: 'unknown',
                sampling_hz: { accel: fs, temperature: fs, temp: fs, vitals: 1 }
            },
            window: {
                start_ts: new Date(startTsMs).toISOString(),
                end_ts: new Date(endTsMs).toISOString(),
                timezone
            },
            context: {
                notes: 'web ble upload',
                tags: ['web', 'ble'],
                location: { lat: 0, lng: 0, accuracy_m: 0 }
            },
            signals: {
                accel: { samples: accelSamples },
                temperature: { samples: tempSamples },
                vitals: { samples: vitalsSamples }
            }
        };
    }

    async _sendBleUploadOnce() {
        if (!this.bleUploadEnabled) return;
        const cfg = this._getBleUploadConfig();
        if (!cfg.url) return;
        const payload = this._buildBleEventPayload();
        if (!payload) return;

        try {
            const resp = await fetch(cfg.url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!resp.ok) {
                this.addBLELog(`❌ 上报失败: HTTP ${resp.status}`);
                this._setBleUploadStatus(`失败(${resp.status})`);
                return;
            }
            this.bleLastUploadTs = Date.now();
            this._setBleUploadStatus(`上传中(最近: ${new Date(this.bleLastUploadTs).toLocaleTimeString()})`);
        } catch (e) {
            this.addBLELog(`❌ 上报异常: ${e.message}`);
            this._setBleUploadStatus('异常');
        }
    }

    /**
     * 构建“连接诊断”信息（用于排查：采样率配置、是否真50Hz、是否有IMU、是否丢包/抖动等）
     */
    buildBleDiagnostics() {
        const now = Date.now();
        const fsCfg = (this.processor && Number.isFinite(this.processor.fs)) ? this.processor.fs : null;
        const stats = this.bleStats || {};
        const elapsedSec = stats.startRxTs ? Math.max(0.001, (now - stats.startRxTs) / 1000) : null;
        const actualFs = (elapsedSec && stats.received) ? (stats.received / elapsedSec) : null;
        const lossRate = (stats.expected && stats.missed !== undefined) ? (stats.missed / Math.max(1, stats.expected)) : null;

        const lastGX = this.bleBufferIMU_X?.length ? this.bleBufferIMU_X[this.bleBufferIMU_X.length - 1] : null;
        const lastGY = this.bleBufferIMU_Y?.length ? this.bleBufferIMU_Y[this.bleBufferIMU_Y.length - 1] : null;
        const lastGZ = this.bleBufferIMU_Z?.length ? this.bleBufferIMU_Z[this.bleBufferIMU_Z.length - 1] : null;

        return {
            ts: new Date().toISOString(),
            bleConnected: !!this.bleConnected,
            samplingRateConfigHz: fsCfg,
            receivedSamples: stats.received ?? 0,
            expectedSamples: stats.expected ?? 0,
            missedSamples: stats.missed ?? 0,
            lossRateEstimated: lossRate,
            actualReceiveRateHz: actualFs,
            jitterEmaMs: stats.gapJitterEmaMs ?? null,
            seqBased: !!stats.seqBased,
            buffers: {
                lenI: this.bleBufferI?.length ?? 0,
                lenQ: this.bleBufferQ?.length ?? 0,
                lenGX: this.bleBufferIMU_X?.length ?? 0
            },
            imuLast: { gx: lastGX, gy: lastGY, gz: lastGZ },
            ui: {
                hasBleIMUChartCanvas: !!document.getElementById('bleIMUChart'),
                bluetoothChartsSectionDisplay: document.getElementById('bluetoothChartsSection')
                    ? getComputedStyle(document.getElementById('bluetoothChartsSection')).display
                    : null
            }
        };
    }

    addBLELog(msg) {
        const log = document.getElementById('bleLog');
        if (!log) return;
        const ts = new Date().toLocaleTimeString();
        this._bleLogLines.push(`[${ts}] ${msg}`);
        if (this._bleLogLines.length > 120) this._bleLogLines.splice(0, this._bleLogLines.length - 120);

        // 节流渲染（避免每次都触发 DOM 重排）
        if (this._bleLogRenderTimer) return;
        this._bleLogRenderTimer = setTimeout(() => {
            this._bleLogRenderTimer = null;
            log.style.whiteSpace = 'pre-line';
            log.textContent = this._bleLogLines.join('\n');
            log.scrollTop = log.scrollHeight;
        }, 200); // 5Hz
    }

    /**
     * 处理 BLE 行数据 - 蓝牙实时数据接口
     * 默认逐行格式: ts i q
     */
    handleBLELine(line) {
        // 保存原始蓝牙数据（如果正在录制）
        if (this.bleRecordingFlag === 1) {
            this.bleRecordingRawData.push(line);
        }

        // 打印原始数据
        this.printRawData(line);
        this.lastBleRxTs = Date.now();
        // 允许 JSON 格式 {ts:..., i:..., q:...}；也兼容无空格双小数如 "1.6421.588"
        let ts, iVal, qVal;
        let imuX = 0, imuY = 0, imuZ = 0; // gx/gy/gz（优先取 Gyr:）
        let temperature = null; // 温度数据
        let adcI = 0, adcQ = 0; // ADC原始值
        let accX = 0, accY = 0, accZ = 0; // Acc原始值
        try {
            const trimmed = line.trim();
            const floatRe = /[+-]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?/g;

            // 🔍 调试：打印前10行的完整信息
            if (this.bleDataCount < 10) {
                console.log(`\n========== 数据行 #${this.bleDataCount} ==========`);
                console.log('原始行:', line);
                console.log('Trim后:', trimmed);
            }

            // 兼容管道格式：ADC:...|Acc:...|Gyr:...|T:...
            
            // 提取 ADC 两值（I/Q通道）
            const parsePairAfterLabel = (label) => {
                const idx = trimmed.indexOf(label);
                if (idx < 0) {
                    if (this.bleDataCount < 10) {
                        console.log(`  ❌ 未找到标签 "${label}"`);
                    }
                    return null;
                }
                
                const seg = trimmed.slice(idx + label.length);
                const firstField = seg.split('|')[0] || '';
                
                // 🔍 更强的数字提取：确保能匹配 "-3455 1176" 这样的格式
                const nums = firstField.match(floatRe)?.map(v => parseFloat(v)) || [];
                
                // 🔍 调试：打印解析过程
                if (this.bleDataCount < 10) {
                    console.log(`  解析${label}:`);
                    console.log(`    idx=${idx}`);
                    console.log(`    seg前50字符="${seg.substring(0, 50)}"`);
                    console.log(`    firstField="${firstField}"`);
                    console.log(`    正则匹配结果:`, nums);
                    console.log(`    nums.length=${nums.length}`);
                    if (nums.length >= 2) {
                        console.log(`    ✅ 提取成功: [0]=${nums[0]}, [1]=${nums[1]}`);
                    } else {
                        console.log(`    ❌ 提取失败: 数字不足2个`);
                    }
                }
                
                return nums.length >= 2 ? [nums[0], nums[1]] : null;
            };
            
            // 提取三值（IMU/温度等）
            const parseTripletAfterLabel = (label) => {
                const idx = trimmed.indexOf(label);
                if (idx < 0) return null;
                const seg = trimmed.slice(idx + label.length);
                const firstField = seg.split('|')[0] || '';
                const nums = firstField.match(floatRe)?.map(v => parseFloat(v)) || [];
                return nums.length >= 3 ? [nums[0], nums[1], nums[2]] : null;
            };
            
            // 先尝试解析 ADC（I/Q通道）
            const adc = parsePairAfterLabel('ADC:') || parsePairAfterLabel('adc:');
            if (adc) {
                // 保存原始ADC值
                adcI = adc[0];
                adcQ = adc[1];

                // ADC 转换公式（与 main.py 第413行一致）：
                // voltage = ((adc_value / 32767) + 1) * 3.3 / 2
                // 这将 -32768~32767 的整数转换为 0~3.3V 的电压
                iVal = ((adc[0] / 32767) + 1) * 3.3 / 2;
                qVal = ((adc[1] / 32767) + 1) * 3.3 / 2;
                ts = Date.now();
                // 🔍 调试日志
                if (this.bleDataCount < 10) {
                    console.log(`  ✅ ADC解析成功!`);
                    console.log(`  原始ADC: I=${adc[0]}, Q=${adc[1]}`);
                    console.log(`  转换电压: I=${iVal.toFixed(4)}V, Q=${qVal.toFixed(4)}V`);
                }
            } else {
                // 🔍 调试：如果ADC解析失败
                if (this.bleDataCount < 10) {
                    console.log(`  ❌ ADC解析失败! adc=null`);
                }
            }
            
            // 解析 IMU 数据（优先陀螺仪）
            const gyr = parseTripletAfterLabel('Gyr:') || parseTripletAfterLabel('GYR:') || parseTripletAfterLabel('GYR_');
            const acc = parseTripletAfterLabel('Acc:') || parseTripletAfterLabel('ACC:');

            // 保存原始Acc值（无论是否用作IMU）
            if (acc) {
                accX = acc[0];
                accY = acc[1];
                accZ = acc[2];
            }

            if (gyr) [imuX, imuY, imuZ] = gyr;
            else if (acc) [imuX, imuY, imuZ] = acc;

            // 解析温度数据 T:23.0
            const tempIdx = trimmed.indexOf('T:');
            if (tempIdx >= 0) {
                const tempSeg = trimmed.slice(tempIdx + 2);
                const tempMatch = tempSeg.match(floatRe);
                if (tempMatch && tempMatch.length > 0) {
                    temperature = parseFloat(tempMatch[0]);
                    if (this.bleDataCount < 10) {
                        console.log(`  解析温度: T=${temperature}°C`);
                    }
                }
            }

            // 可选：解析序号（如果设备未来加了 seq 字段）
            // 例如："...|SEQ:1234|..." 或 JSON {"seq":1234,...}
            let seq = null;
            const seqMatch = trimmed.match(/(?:\bSEQ\b|\bseq\b|\bidx\b|\bindex\b)\s*[:=]\s*(\d+)/);
            if (seqMatch) seq = parseInt(seqMatch[1], 10);

            // 如果还没有从 ADC 字段提取到数据，则尝试其他格式
            if (!Number.isFinite(iVal) || !Number.isFinite(qVal)) {
                if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
                    const obj = JSON.parse(trimmed);
                    ts = obj.ts ?? Date.now();
                    iVal = parseFloat(obj.i);
                    qVal = parseFloat(obj.q);
                    if (seq === null && obj.seq !== undefined) seq = parseInt(obj.seq, 10);
                } else {
                    const parts = trimmed.split(/\s+/);
                    if (parts.length >= 3) {
                        ts = parts[0];
                        iVal = parseFloat(parts[1]);
                        qVal = parseFloat(parts[2]);
                    } else {
                        // 提取该行中的所有浮点（支持 .xxx 形式）并保留索引
                        const matches = [...trimmed.matchAll(floatRe)];
                        if (matches.length >= 2) {
                            let firstStr = matches[0][0];
                            let secondStr = matches[1][0];
                            const secondIdx = matches[1].index;
                            // 修复形如 "1.6421.588" => 第一项去掉最后一位，第二项补上该位："1.642" 与 "1.588"
                            if (secondStr.startsWith('.') && secondIdx > 0) {
                                const prevChar = trimmed[secondIdx - 1];
                                if (prevChar >= '0' && prevChar <= '9' && /\d$/.test(firstStr)) {
                                    // 仅当第一项最后也是数字时进行重组
                                    secondStr = prevChar + secondStr;
                                    firstStr = firstStr.slice(0, -1);
                                }
                            }
                            ts = Date.now();
                            iVal = parseFloat(firstStr);
                            qVal = parseFloat(secondStr);
                        } else if (matches.length === 1) {
                            // 单值：与上一次的单值配对
                            const val = parseFloat(matches[0][0]);
                            if (!Number.isFinite(val)) return;
                            if (this.blePendingFloat === null) {
                                this.blePendingFloat = val;
                                return;
                            } else {
                                ts = Date.now();
                                iVal = this.blePendingFloat;
                                qVal = val;
                                this.blePendingFloat = null;
                            }
                        } else {
                            return;
                        }
                    }
                }
            }

            // 更新丢包/采样率统计（放在 try 内，确保能拿到 seq）
            this._updateBleLossStats(seq);
        } catch (err) { 
            // 🔍 调试：捕获异常
            if (this.bleDataCount < 10) {
                console.log(`  ❌ 解析异常:`, err);
            }
            return; 
        }

        // 🔍 调试：检查最终的 iVal 和 qVal
        if (this.bleDataCount < 10) {
            console.log(`  最终检查: iVal=${iVal}, qVal=${qVal}`);
            console.log(`  iVal有效: ${Number.isFinite(iVal)}, qVal有效: ${Number.isFinite(qVal)}`);
        }

        if (!Number.isFinite(iVal) || !Number.isFinite(qVal)) {
            if (this.bleDataCount < 10) {
                console.log(`  ❌ 数据无效，丢弃此行`);
                console.log(`========================================\n`);
            }
            return;
        }

        // 🔍 调试：确认数据被添加
        if (this.bleDataCount < 10) {
            console.log(`  ✅ 准备添加到buffer: I=${iVal.toFixed(4)}V, Q=${qVal.toFixed(4)}V`);
            console.log(`  当前buffer长度: I=${this.bleBufferI.length}, Q=${this.bleBufferQ.length}`);
        }

        this.bleBufferTimestamps.push(ts);
        this.bleBufferI.push(iVal);
        this.bleBufferQ.push(qVal);

        // 🔍 调试：验证数据确实被添加
        if (this.bleDataCount < 10) {
            const lastI = this.bleBufferI[this.bleBufferI.length - 1];
            const lastQ = this.bleBufferQ[this.bleBufferQ.length - 1];
            console.log(`  ✅ 添加后验证: I数组最后一个=${lastI?.toFixed(4)}, Q数组最后一个=${lastQ?.toFixed(4)}`);
            console.log(`  添加后buffer长度: I=${this.bleBufferI.length}, Q=${this.bleBufferQ.length}`);
            console.log(`========================================\n`);
        }

        // IMU 三轴（gx/gy/gz），保持与 I/Q 同步长度
        this.bleBufferIMU_X.push(Number.isFinite(imuX) ? imuX : 0);
        this.bleBufferIMU_Y.push(Number.isFinite(imuY) ? imuY : 0);
        this.bleBufferIMU_Z.push(Number.isFinite(imuZ) ? imuZ : 0);
        
        // 温度数据：只有当设备发送了温度数据时才更新，否则使用null表示无数据
        if (temperature !== null && Number.isFinite(temperature)) {
            this.bleBufferTemperature.push(temperature);
        } else {
            // 如果没有温度数据，仍然保持数组长度同步，填充null
            this.bleBufferTemperature.push(null);
        }
        
        // 实时保存完整的原始蓝牙数据
        if (this.bleRecordingFlag === 1) {
            const timestamp = new Date().toISOString().replace('T', ' ').slice(0, 19);

            // 保存完整的蓝牙原始数据：时间戳、ADC、Acc、I、Q、IMU(x,y,z)、温度
            // 格式：timestamp ADC_I ADC_Q Acc_X Acc_Y Acc_Z I_voltage Q_voltage IMU_x IMU_y IMU_z temperature
            const imuX = this.bleBufferIMU_X[this.bleBufferIMU_X.length - 1] || 0;
            const imuY = this.bleBufferIMU_Y[this.bleBufferIMU_Y.length - 1] || 0;
            const imuZ = this.bleBufferIMU_Z[this.bleBufferIMU_Z.length - 1] || 0;
            const temp = this.bleBufferTemperature[this.bleBufferTemperature.length - 1];

            // 需要从原始字符串中提取ADC和Acc的值
            // 这里我们需要在handleBLELine函数中保存这些值，或者重新解析
            // 为了简单，我们可以从当前处理的变量中获取（如果可用的话）

            // 临时解决方案：如果能从当前上下文中获取ADC和Acc值就保存，否则用默认值
            let adcI = 0, adcQ = 0, accX = 0, accY = 0, accZ = 0;

            // 尝试从原始字符串重新解析ADC和Acc（简化版本）
            try {
                const trimmed = line.trim();
                const adcMatch = trimmed.match(/ADC:([-\d]+)\s+([-\d]+)/);
                if (adcMatch) {
                    adcI = parseInt(adcMatch[1]);
                    adcQ = parseInt(adcMatch[2]);
                }

                const accMatch = trimmed.match(/Acc:([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)/);
                if (accMatch) {
                    accX = parseFloat(accMatch[1]);
                    accY = parseFloat(accMatch[2]);
                    accZ = parseFloat(accMatch[3]);
                }
            } catch (e) {
                // 解析失败时使用默认值
                console.warn('解析ADC/Acc失败，使用默认值');
            }

            const dataLine = `${timestamp}  ${adcI}  ${adcQ}  ${accX.toFixed(3)}  ${accY.toFixed(3)}  ${accZ.toFixed(3)}  ${iVal.toFixed(6)}  ${qVal.toFixed(6)}  ${imuX.toFixed(3)}  ${imuY.toFixed(3)}  ${imuZ.toFixed(3)}  ${temp !== null ? temp.toFixed(2) : 'N/A'}`;
            this.bleRecordingData.push(dataLine);
        }
        
        // 更新数据计数
        this.bleDataCount++;
        document.getElementById('bleDataCount').textContent = this.bleDataCount;
        document.getElementById('bleTotalDataPoints').textContent = this.bleDataCount;
        
        // 通知静息监测模块（如果存在）
        if (typeof restingMonitor !== 'undefined' && restingMonitor) {
            restingMonitor.update();
        }

        // 🔍 调试：在页面上显示最新的I/Q值（每10条更新一次）
        if (this.bleDataCount % 10 === 0) {
            const lastI = this.bleBufferI[this.bleBufferI.length - 1];
            const lastQ = this.bleBufferQ[this.bleBufferQ.length - 1];
            const debugInfo = `I=${lastI?.toFixed(4)}V, Q=${lastQ?.toFixed(4)}V (共${this.bleBufferI.length}点)`;
            // 单独的调试区域，避免覆盖“当前心率”显示
            const debugEl = document.getElementById('bleCurrentIQ');
            if (debugEl && this.bleDataCount < 100) {
                debugEl.textContent = debugInfo;
            }
        }

        // 滑窗（分块裁剪，显著降低长期运行时的卡顿）
        this._trimBleBuffersIfNeeded();

        // 图表更新节流（每条都刷会卡）
        this._scheduleBleChartUpdate();

        // 每累计一段再做一次完整生理参数估计（降低更新频率以提高稳定性）
        // 参考main.py每收集一定数量数据才计算一次（第72400个计数）
        const fs = (this.processor && Number.isFinite(this.processor.fs)) ? this.processor.fs : 50;
        // 改为每1秒计算一次（50个点），更频繁更新
        if (this.bleBufferI.length % fs === 0 && this.bleBufferI.length >= fs * 5) {
            this.updateBluetoothVitalSigns();
        }
    }

    _trimBleBuffersIfNeeded() {
        const len = this.bleBufferI.length;
        if (len <= this.bleMaxBufferHard) return;
        const removeCount = len - this.bleMaxBuffer;
        if (removeCount <= 0) return;

        // 保持各数组长度一致
        [
            this.bleBufferTimestamps,
            this.bleBufferI,
            this.bleBufferQ,
            this.bleBufferIMU_X,
            this.bleBufferIMU_Y,
            this.bleBufferIMU_Z,
            this.bleBufferTemperature
        ].forEach(arr => {
            if (Array.isArray(arr) && arr.length >= removeCount) arr.splice(0, removeCount);
        });
    }

    _scheduleBleChartUpdate() {
        if (this._bleChartRaf) return;
        this._bleChartRaf = requestAnimationFrame(() => {
            this._bleChartRaf = null;
            const now = performance.now();
            if (now - this._bleChartLastUpdateTs < this._bleChartMinIntervalMs) return;
            this._bleChartLastUpdateTs = now;
            this.updateBluetoothLiveCharts();
        });
    }

    /**
     * 更新 BLE 丢包/实际采样率/抖动（估算）
     * - 默认假设：每调用一次 handleBLELine = 1 个采样点（你的设备目前看起来是这样）
     * - 若提供 seq：使用 seq 计算丢包更准确
     */
    _updateBleLossStats(seq = null) {
        const fs = (this.processor && Number.isFinite(this.processor.fs)) ? this.processor.fs : 100;
        const expectedIntervalMs = 1000 / fs;
        const now = Date.now();

        const s = this.bleStats;
        if (!s.startRxTs) s.startRxTs = now;

        // received 计数（每条 line 视为一个采样点）
        s.received += 1;

        if (Number.isFinite(seq)) {
            if (s.lastSeq !== null) {
                const gap = seq - s.lastSeq - 1;
                if (gap > 0) {
                    s.missed += gap;
                    s.seqBased = true;
                }
            }
            s.lastSeq = seq;
        }

        if (s.lastRxTs > 0) {
            const gapMs = now - s.lastRxTs;
            s.lastGapMs = gapMs;

            // EMA 估计间隔与抖动
            const alpha = 0.1;
            s.gapEmaMs = s.gapEmaMs ? (alpha * gapMs + (1 - alpha) * s.gapEmaMs) : gapMs;
            const jitter = Math.abs(gapMs - expectedIntervalMs);
            s.gapJitterEmaMs = s.gapJitterEmaMs ? (alpha * jitter + (1 - alpha) * s.gapJitterEmaMs) : jitter;
        }
        s.lastRxTs = now;

        // 期望点数：
        // - 有 seq：expected = received + seqMissing（精确）
        // - 无 seq：用累计时间计算 expected，避免“批量送达/主线程卡顿”导致的假丢包
        const elapsedSec = Math.max(0.001, (now - s.startRxTs) / 1000);
        if (s.seqBased) {
            s.expected = s.received + s.missed;
        } else {
            s.expected = Math.round(elapsedSec * fs);
            s.missed = Math.max(0, s.expected - s.received);
        }

        // UI 更新（降频：每约 0.5s 更新一次即可；这里用简单取模）
        if (s.received % Math.max(1, Math.floor(fs / 2)) !== 0) return;

        const actualFs = s.received / elapsedSec;
        const lossRate = s.expected > 0 ? (s.missed / s.expected) : 0;

        const fsEl = document.getElementById('bleActualFs');
        const lossEl = document.getElementById('blePacketLoss');
        const jitterEl = document.getElementById('bleJitter');
        if (fsEl) fsEl.textContent = `${actualFs.toFixed(1)} Hz`;
        if (lossEl) lossEl.textContent = `${(lossRate * 100).toFixed(2)} %`;
        if (jitterEl) jitterEl.textContent = `${(s.gapJitterEmaMs || 0).toFixed(1)} ms`;
    }

    // 这些函数已被蓝牙专用函数取代，保留作为兼容性
    updateLiveCharts() {
        // 现在由 updateBluetoothLiveCharts() 处理蓝牙数据
        // 文件数据由 updateCharts() 处理
        console.log('updateLiveCharts: 已弃用，请使用 updateBluetoothLiveCharts');
    }

    updateLiveVitalFromBuffer() {
        // 现在由 updateBluetoothVitalSigns() 处理蓝牙数据
        // 文件数据由正常的文件处理流程处理
        console.log('updateLiveVitalFromBuffer: 已弃用，请使用 updateBluetoothVitalSigns');
    }

    /**
     * 处理文件选择
     */
    handleFileSelect(event) {
        const files = Array.from(event.target.files);
        const validFiles = files.filter(file =>
            file.name.toLowerCase().endsWith('.txt') ||
            file.name.toLowerCase().endsWith('.json')
        );

        if (validFiles.length === 0) {
            this.showMessage('请选择.txt或.json格式的数据文件', 'warning');
            return;
        }

        this.selectedFiles = validFiles;
        this.displayFileList();
        this.showMessage(`已选择 ${validFiles.length} 个文件`, 'success');
    }

    /**
     * 显示文件列表
     */
    displayFileList() {
        const fileList = document.getElementById('fileList');
        const fileItems = document.getElementById('fileItems');
        
        fileItems.innerHTML = '';
        
        this.selectedFiles.forEach((file, index) => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span>${file.name}</span>
                <span>${this.formatFileSize(file.size)}</span>
            `;
            fileItems.appendChild(li);
        });
        
        fileList.style.display = 'block';
    }

    /**
     * 清空文件列表
     */
    clearFiles() {
        this.selectedFiles = [];
        document.getElementById('fileList').style.display = 'none';
        document.getElementById('fileInput').value = '';
        this.hideResults();
    }

    /**
     * 处理文件
     */
    async processFiles() {
        if (this.selectedFiles.length === 0) {
            this.showMessage('请先选择文件', 'warning');
            return;
        }

        this.showLoading(true);
        this.showStatus(true);
        this.processedResults = [];

        const totalFiles = this.selectedFiles.length;
        let processedCount = 0;

        for (const file of this.selectedFiles) {
            try {
                this.updateProgress(processedCount / totalFiles * 100, 
                    `正在处理: ${file.name}`);
                
                this.addStatusLog(`开始处理文件: ${file.name}`);
                
                // 读取文件内容
                const fileContent = await this.readFileContent(file);

                // 处理数据
                let result;
                if (file.name.toLowerCase().endsWith('.json')) {
                    result = this.processJsonFile(file.name, fileContent);
                } else {
                    result = this.processor.processSingleFile(file.name, fileContent);
                }
                this.processedResults.push(result);
                
                if (result.status === 'success') {
                    if (result.dataType === 'json') {
                        this.addStatusLog(`✓ ${file.name} 处理成功 - 动物: ${result.animal.name}(${result.animal.species}), 心率: ${result.heartRate} bpm, 呼吸: ${result.respiratoryRate} bpm`);
                    } else {
                        this.addStatusLog(`✓ ${file.name} 处理成功 - 心率: ${result.heartRate} bpm, 呼吸: ${result.respiratoryRate} bpm`);
                    }
                } else {
                    this.addStatusLog(`✗ ${file.name} 处理失败: ${result.error}`);
                }
                
                processedCount++;
                
            } catch (error) {
                this.addStatusLog(`✗ ${file.name} 处理出错: ${error.message}`);
                this.processedResults.push({
                    fileName: file.name,
                    error: error.message,
                    status: 'error'
                });
                processedCount++;
            }
        }

        this.updateProgress(100, '处理完成');
        this.showLoading(false);
        
        // 显示结果
        this.displayResults();
        this.showMessage(`处理完成！成功处理 ${this.processedResults.filter(r => r.status === 'success').length} 个文件`, 'success');
    }

    /**
     * 读取文件内容
     */
    readFileContent(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(new Error('文件读取失败'));
            reader.readAsText(file, 'utf-8');
        });
    }

    /**
     * 处理JSON格式的传感器数据文件
     */
    processJsonFile(fileName, jsonContent) {
        try {
            const data = JSON.parse(jsonContent);

            // 验证数据结构
            if (!data.event_id || !data.animal || !data.signals) {
                return {
                    fileName: fileName,
                    status: 'error',
                    error: 'JSON格式不正确，缺少必要字段'
                };
            }

            // 提取动物信息
            const animal = data.animal;
            const device = data.device || {};
            const vitals = data.signals.vitals || { samples: [] };
            const accel = data.signals.accel || { samples: [] };
            const temperature = data.signals.temperature || { samples: [] };

            // 计算统计信息
            const hrValues = vitals.samples.map(s => s.hr).filter(hr => hr && hr > 0);
            const rrValues = vitals.samples.map(s => s.rr).filter(rr => rr && rr > 0);
            const tempValues = temperature.samples.map(s => s.value).filter(temp => temp && temp > 0);

            const avgHeartRate = hrValues.length > 0 ? hrValues.reduce((a, b) => a + b, 0) / hrValues.length : 0;
            const avgRespRate = rrValues.length > 0 ? rrValues.reduce((a, b) => a + b, 0) / rrValues.length : 0;
            const avgTemp = tempValues.length > 0 ? tempValues.reduce((a, b) => a + b, 0) / tempValues.length : 0;

            return {
                fileName: fileName,
                status: 'success',
                dataType: 'json',
                animal: animal,
                device: device,
                heartRate: Math.round(avgHeartRate * 10) / 10,
                respiratoryRate: Math.round(avgRespRate * 10) / 10,
                temperature: Math.round(avgTemp * 10) / 10,
                dataPoints: Math.max(vitals.samples.length, accel.samples.length, temperature.samples.length),
                hrData: hrValues,
                rrData: rrValues,
                tempData: tempValues,
                rawData: data
            };

        } catch (error) {
            return {
                fileName: fileName,
                status: 'error',
                error: `JSON解析失败: ${error.message}`
            };
        }
    }

    /**
     * 显示处理结果
     */
    displayResults() {
        const successResults = this.processedResults.filter(r => r.status === 'success');

        if (successResults.length === 0) {
            this.showMessage('没有成功处理的文件', 'warning');
            return;
        }

        // 检查是否有JSON数据
        const hasJsonData = successResults.some(r => r.dataType === 'json');

        // 更新统计信息
        this.updateStatistics(successResults);

        // 更新图表
        this.updateCharts(successResults);

        // 更新结果表格
        this.updateResultsTable();

        // 显示JSON数据详细信息
        if (hasJsonData) {
            this.displayJsonDataDetails(successResults);
            document.getElementById('jsonDataSection').style.display = 'block';
            document.getElementById('healthAnalysisSection').style.display = 'block';
        } else {
            document.getElementById('jsonDataSection').style.display = 'none';
            document.getElementById('healthAnalysisSection').style.display = 'none';
        }

        // JSON时隐藏部分图表，仅保留心率/呼吸时间序列
        this.setChartVisibilityForJson(hasJsonData);

        // 显示结果区域
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('resultsSection').classList.add('fade-in');
    }

    /**
     * 显示JSON数据的详细信息
     */
    displayJsonDataDetails(results) {
        const jsonResults = results.filter(r => r.dataType === 'json');
        if (jsonResults.length === 0) return;

        // 使用最新的JSON数据（如果有多个，取第一个）
        const latestResult = jsonResults[0];
        const animal = latestResult.animal;
        const device = latestResult.device;
        const rawData = latestResult.rawData;

        // 更新动物信息
        const animalEmoji = animal.species === 'dog' ? '🐕' : animal.species === 'cat' ? '🐱' : '🐾';
        document.getElementById('animalEmoji').textContent = animalEmoji;
        document.getElementById('animalName').textContent = animal.name || '未命名宠物';
        document.getElementById('animalBasicInfo').textContent =
            `${animal.breed || '未知品种'} · ${animal.age_months ? Math.floor(animal.age_months / 12) + '岁' + (animal.age_months % 12) + '个月' : '年龄未知'} · ${animal.sex === 'male' ? '公' : animal.sex === 'female' ? '母' : '性别未知'}`;
        document.getElementById('animalWeight').textContent = animal.weight_kg ? `${animal.weight_kg} kg` : '-- kg';
        document.getElementById('animalId').textContent = animal.animal_id || '--';

        // 更新设备信息
        document.getElementById('deviceId').textContent = device.device_id || '--';
        document.getElementById('deviceFirmware').textContent = device.firmware || '--';

        const samplingInfo = device.sampling_hz ?
            `心率:${device.sampling_hz.vitals || '--'}/秒, 加速度:${device.sampling_hz.accel || '--'}Hz, 温度:${device.sampling_hz.temp || '--'}/秒` : '--';
        document.getElementById('deviceSampling').textContent = samplingInfo;

        // 更新测量信息
        document.getElementById('eventId').textContent = rawData.event_id || '--';

        const eventTime = rawData.ts ? new Date(rawData.ts).toLocaleString('zh-CN') : '--';
        document.getElementById('measurementTime').textContent = eventTime;

        const window = rawData.window;
        if (window && window.start_ts && window.end_ts) {
            const startTime = new Date(window.start_ts);
            const endTime = new Date(window.end_ts);
            const duration = Math.round((endTime - startTime) / 1000);
            document.getElementById('measurementDuration').textContent = `${duration} 秒`;
        } else {
            document.getElementById('measurementDuration').textContent = '--';
        }

        const context = rawData.context || {};
        const location = context.location ?
            `${context.location.lat}, ${context.location.lng}` : '--';
        document.getElementById('measurementLocation').textContent = location;

        document.getElementById('measurementNotes').textContent = context.notes || '--';

        // 设置默认的agent endpoint
        const agentEndpointEl = document.getElementById('agentEndpoint');
        if (agentEndpointEl && !agentEndpointEl.value) {
            agentEndpointEl.value = localStorage.getItem('agentEndpoint') || 'http://localhost:8000';
        }
    }

    /**
     * 执行宠物健康分析
     */
    async performHealthAnalysis() {
        const jsonResults = this.processedResults.filter(r => r.dataType === 'json');
        if (jsonResults.length === 0) {
            this.showMessage('没有找到可分析的JSON数据', 'warning');
            return;
        }

        const agentEndpoint = document.getElementById('agentEndpoint').value.trim();
        if (!agentEndpoint) {
            this.showMessage('请设置Agent API地址', 'warning');
            return;
        }

        // 保存endpoint到localStorage
        localStorage.setItem('agentEndpoint', agentEndpoint);

        const result = jsonResults[0]; // 使用第一个JSON结果
        const analysisBtn = document.getElementById('healthAnalysisBtn');
        const reportContainer = document.getElementById('healthAnalysisReport');
        const reportContent = document.getElementById('analysisReportContent');

        // 显示分析界面
        reportContainer.style.display = 'block';
        analysisBtn.disabled = true;
        analysisBtn.textContent = '🔄 分析中...';

        reportContent.innerHTML = `
            <div class="loading-analysis">
                <div class="loading-spinner"></div>
                <p>正在分析宠物健康状况，请稍候...</p>
            </div>
        `;

        try {
            // 构建健康分析查询
            const query = this.buildHealthAnalysisQuery(result);

            // 调用agent API
            const response = await fetch(`${agentEndpoint}/agent/plan_and_solve`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    llm_base_url: 'https://api.openai.com/v1',
                    llm_api_key:  process.env.OPENAI_API_KEY, // 需要用户配置
                    llm_model: 'deepseek-chat',
                    allowed_tools: ['rag.search'],
                    temperature: 0.7,
                    max_tokens: 2000
                })
            });

            if (!response.ok) {
                throw new Error(`Agent API请求失败: ${response.status}`);
            }

            const analysisResult = await response.json();

            if (!analysisResult.ok) {
                throw new Error(analysisResult.error?.message || '分析失败');
            }

            // 显示分析结果
            this.displayHealthAnalysisResult(analysisResult, result);

        } catch (error) {
            console.error('健康分析失败:', error);
            reportContent.innerHTML = `
                <div class="analysis-error">
                    <h4>❌ 分析失败</h4>
                    <p>错误信息: ${error.message}</p>
                    <p>请检查Agent API地址和配置是否正确。</p>
                </div>
            `;
        } finally {
            analysisBtn.disabled = false;
            analysisBtn.textContent = '🩺 开始健康分析';
        }
    }

    /**
     * 构建健康分析查询
     */
    buildHealthAnalysisQuery(result) {
        const animal = result.animal;
        const vitals = result.rawData.signals.vitals || { samples: [] };
        const context = result.rawData.context || {};

        const avgHR = result.heartRate;
        const avgRR = result.respiratoryRate;
        const temp = result.temperature;

        let query = `请分析这只${animal.species === 'dog' ? '狗狗' : '猫咪'}的健康状况：

宠物信息：
- 姓名: ${animal.name || '未命名'}
- 品种: ${animal.breed || '未知'}
- 年龄: ${animal.age_months ? Math.floor(animal.age_months / 12) + '岁' + (animal.age_months % 12) + '个月' : '未知'}
- 体重: ${animal.weight_kg || '未知'}kg
- 性别: ${animal.sex === 'male' ? '公' : animal.sex === 'female' ? '母' : '未知'}

生理指标：
- 平均心率: ${avgHR} bpm
- 平均呼吸频率: ${avgRR} bpm
- 体温: ${temp}°C

测量情况：
- 位置: ${context.location ? `${context.location.lat}, ${context.location.lng}` : '未知'}
- 备注: ${context.notes || '无'}
- 标签: ${context.tags ? context.tags.join(', ') : '无'}

请基于这些数据分析宠物的健康状况，包括：
1. 心率和呼吸频率是否正常
2. 体温是否正常
3. 整体健康评估
4. 如果有异常，建议采取什么措施
5. 日常护理建议

请提供详细的分析报告。`;

        return query;
    }

    /**
     * 显示健康分析结果
     */
    displayHealthAnalysisResult(analysisResult, originalData) {
        const timestamp = new Date().toLocaleString('zh-CN');
        document.getElementById('analysisTimestamp').textContent = `分析时间: ${timestamp}`;

        const reportContent = document.getElementById('analysisReportContent');

        // 格式化分析结果
        const answer = analysisResult.answer || '暂无分析结果';
        const plan = analysisResult.plan || [];
        const toolResults = analysisResult.tool_results || [];

        reportContent.innerHTML = `
            <div class="analysis-summary">
                <h4>📊 分析总结</h4>
                <div class="analysis-content">${this.formatAnalysisText(answer)}</div>
            </div>

            ${plan.length > 0 ? `
            <div class="analysis-plan" style="margin-top: 20px;">
                <h4>🔍 分析过程</h4>
                <ol>
                    ${plan.map(step => `<li><strong>${step.type === 'tool' ? '工具调用' : '推理'}:</strong> ${step.note || step.tool_name || '未知步骤'}</li>`).join('')}
                </ol>
            </div>
            ` : ''}

            ${toolResults.length > 0 ? `
            <div class="tool-results" style="margin-top: 20px;">
                <h4>📚 参考资料</h4>
                ${toolResults.map((result, index) => `
                    <div class="tool-result-item">
                        <h5>工具 ${index + 1}: ${result.tool_name}</h5>
                        <div class="tool-content">${this.formatToolResult(result)}</div>
                    </div>
                `).join('')}
            </div>
            ` : ''}
        `;

        this.showMessage('健康分析完成！', 'success');
    }

    /**
     * 格式化分析文本
     */
    formatAnalysisText(text) {
        if (!text) return '暂无内容';

        // 简单的文本格式化，转换换行符和列表
        return text
            .replace(/\n/g, '<br/>')
            .replace(/(\d+)\.\s/g, '<br/>$1. ')
            .replace(/^(\d+)\.\s/gm, '<br/>$1. ');
    }

    /**
     * 格式化工具结果
     */
    formatToolResult(result) {
        if (!result || !result.data) return '暂无数据';

        try {
            const data = typeof result.data === 'string' ? JSON.parse(result.data) : result.data;

            if (result.tool_name === 'rag.search' && data.results) {
                return data.results.map(item =>
                    `<div class="rag-item">
                        <strong>相关度: ${item.score ? item.score.toFixed(3) : '未知'}</strong><br/>
                        ${item.content || item.text || '无内容'}
                    </div>`
                ).join('');
            }

            return JSON.stringify(data, null, 2);
        } catch (e) {
            return result.data;
        }
    }

    /**
     * 导出健康报告
     */
    exportHealthReport() {
        const reportContent = document.getElementById('analysisReportContent');
        if (!reportContent) {
            this.showMessage('没有可导出的报告', 'warning');
            return;
        }

        const reportText = reportContent.innerText || reportContent.textContent;
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');

        const blob = new Blob([reportText], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `宠物健康分析报告_${timestamp}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showMessage('报告已导出！', 'success');
    }

    /**
     * 初始化宠物健康对话
     */
    initializeHealthChat() {
        const agentEndpoint = document.getElementById('chatAgentEndpoint').value.trim();
        if (!agentEndpoint) {
            this.showMessage('请设置Agent API地址', 'warning');
            return;
        }

        // 保存endpoint到localStorage
        localStorage.setItem('chatAgentEndpoint', agentEndpoint);

        // 显示对话界面
        document.getElementById('chatContainer').style.display = 'block';
        document.getElementById('initChatBtn').style.display = 'none';
        document.getElementById('clearChatBtn').style.display = 'inline-block';
        document.getElementById('sendChatBtn').disabled = false;

        // 加载历史对话
        this.loadChatHistory();

        this.showMessage('宠物健康对话已启动！', 'success');
    }

    /**
     * 发送对话消息
     */
    async sendChatMessage() {
        const inputEl = document.getElementById('chatInput');
        const message = inputEl.value.trim();
        if (!message) {
            this.showMessage('请输入问题内容', 'warning');
            return;
        }

        const agentEndpoint = document.getElementById('chatAgentEndpoint').value.trim();
        const sendBtn = document.getElementById('sendChatBtn');

        // 添加用户消息到界面
        this.addChatMessage('user', message);
        inputEl.value = '';
        sendBtn.disabled = true;
        sendBtn.textContent = '发送中...';

        // 添加AI思考中消息
        const thinkingMessageId = this.addChatMessage('assistant', '正在思考中...', true);

        try {
            // 构建上下文信息
            const contextInfo = this.buildChatContext();

            // 构建完整查询
            const fullQuery = `${contextInfo}\n\n用户问题: ${message}`;

            // 调用agent API
            const response = await fetch(`${agentEndpoint}/agent/plan_and_solve`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: fullQuery,
                    allowed_tools: ['rag.search'],
                    temperature: 0.7,
                    max_tokens: 1500
                })
            });

            if (!response.ok) {
                throw new Error(`Agent API请求失败: ${response.status}`);
            }

            const result = await response.json();

            if (!result.ok) {
                throw new Error(result.error?.message || '对话失败');
            }

            // 更新AI回复
            this.updateChatMessage(thinkingMessageId, result.answer || '暂无回复');

            // 保存对话历史
            this.saveChatMessage('user', message);
            this.saveChatMessage('assistant', result.answer || '暂无回复');

        } catch (error) {
            console.error('对话失败:', error);
            this.updateChatMessage(thinkingMessageId, `❌ 抱歉，回复失败: ${error.message}`);
        } finally {
            sendBtn.disabled = false;
            sendBtn.textContent = '发送';
        }
    }

    /**
     * 构建对话上下文信息
     */
    buildChatContext() {
        const jsonResults = this.processedResults.filter(r => r.dataType === 'json');
        let context = '您是专业的宠物健康助手，可以解答关于宠物健康、护理、训练等方面的问题。';

        if (jsonResults.length > 0) {
            const result = jsonResults[0];
            const animal = result.animal;

            context += `\n\n当前宠物信息:
- 宠物类型: ${animal.species === 'dog' ? '狗狗' : '猫咪'}
- 姓名: ${animal.name || '未命名'}
- 品种: ${animal.breed || '未知'}
- 年龄: ${animal.age_months ? Math.floor(animal.age_months / 12) + '岁' + (animal.age_months % 12) + '个月' : '未知'}
- 体重: ${animal.weight_kg || '未知'}kg
- 性别: ${animal.sex === 'male' ? '公' : animal.sex === 'female' ? '母' : '未知'}

最近的生理指标:
- 平均心率: ${result.heartRate} bpm
- 平均呼吸频率: ${result.respiratoryRate} bpm
- 体温: ${result.temperature}°C

请基于这些信息提供专业的建议。`;
        }

        return context;
    }

    /**
     * 添加聊天消息到界面
     */
    addChatMessage(role, content, isThinking = false) {
        const messagesEl = document.getElementById('chatMessages');
        const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        const messageHtml = `
            <div class="chat-message ${role}-message ${isThinking ? 'thinking' : ''}" id="${messageId}">
                <div class="message-avatar">${role === 'user' ? '👤' : '🤖'}</div>
                <div class="message-content">
                    <div class="message-text">${this.formatChatMessage(content)}</div>
                    <div class="message-time">${new Date().toLocaleTimeString('zh-CN')}</div>
                </div>
            </div>
        `;

        messagesEl.insertAdjacentHTML('beforeend', messageHtml);

        // 滚动到底部
        messagesEl.scrollTop = messagesEl.scrollHeight;

        return messageId;
    }

    /**
     * 更新聊天消息
     */
    updateChatMessage(messageId, newContent) {
        const messageEl = document.getElementById(messageId);
        if (messageEl) {
            const textEl = messageEl.querySelector('.message-text');
            if (textEl) {
                textEl.innerHTML = this.formatChatMessage(newContent);
                messageEl.classList.remove('thinking');
            }
        }
    }

    /**
     * 格式化聊天消息
     */
    formatChatMessage(text) {
        if (!text) return '';

        return text
            .replace(/\n/g, '<br/>')
            .replace(/(\d+)\.\s/g, '<br/>$1. ')
            .replace(/^(\d+)\.\s/gm, '<br/>$1. ');
    }

    /**
     * 保存聊天消息到本地存储
     */
    saveChatMessage(role, content) {
        const chatHistory = JSON.parse(localStorage.getItem('petHealthChatHistory') || '[]');
        chatHistory.push({
            role: role,
            content: content,
            timestamp: new Date().toISOString()
        });

        // 只保留最近50条消息
        if (chatHistory.length > 50) {
            chatHistory.splice(0, chatHistory.length - 50);
        }

        localStorage.setItem('petHealthChatHistory', JSON.stringify(chatHistory));
    }

    /**
     * 加载聊天历史
     */
    loadChatHistory() {
        const chatHistory = JSON.parse(localStorage.getItem('petHealthChatHistory') || '[]');
        const messagesEl = document.getElementById('chatMessages');

        // 清空现有消息（保留欢迎消息）
        const welcomeMessage = messagesEl.querySelector('.welcome-message');
        messagesEl.innerHTML = '';
        if (welcomeMessage) {
            messagesEl.appendChild(welcomeMessage);
        }

        // 添加历史消息
        chatHistory.forEach(msg => {
            this.addChatMessage(msg.role, msg.content);
        });
    }

    /**
     * 清空聊天历史
     */
    clearChatHistory() {
        localStorage.removeItem('petHealthChatHistory');
        this.loadChatHistory();
        this.showMessage('对话历史已清空', 'info');
    }

    /**
     * 更新统计信息
     */
    updateStatistics(results) {
        const heartRates = results.map(r => r.heartRate).filter(hr => hr > 0);
        const respRates = results.map(r => r.respiratoryRate).filter(rr => rr > 0);
        const totalDataPoints = results.reduce((sum, r) => sum + r.dataPoints, 0);

        document.getElementById('avgHeartRate').textContent = 
            heartRates.length > 0 ? `${(heartRates.reduce((a, b) => a + b, 0) / heartRates.length).toFixed(1)} bpm` : '-- bpm';
        
        document.getElementById('avgRespRate').textContent = 
            respRates.length > 0 ? `${(respRates.reduce((a, b) => a + b, 0) / respRates.length).toFixed(1)} bpm` : '-- bpm';
        
        document.getElementById('processedFiles').textContent = results.length;
        document.getElementById('totalDataPoints').textContent = totalDataPoints.toLocaleString();
    }

    /**
     * 初始化蓝牙图表
     */
    initializeBluetoothCharts() {
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: { display: true }
                },
                y: {
                    display: true,
                    title: { display: true }
                }
            },
            animation: false // 关闭动画以提高实时性能
        };

        // 初始化蓝牙专用图表
        // I 通道 - 放大显示微小变化
        this.bleCharts.iSignal = new Chart(document.getElementById('bleISignalChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                ...chartOptions,
                plugins: { ...chartOptions.plugins, title: { display: true, text: '蓝牙 I 通道实时信号 (自适应放大)' } },
                scales: {
                    x: { display: true, title: { display: true, text: '采样点' } },
                    y: {
                        display: true,
                        title: { display: true, text: '幅度 (V)' },
                        min: 1.2,    // 扩大初始范围，更清楚显示波峰变化
                        max: 2.8,    // 适度范围以突出波峰细节
                        beginAtZero: false
                    }
                }
            }
        });

        // Q 通道 - 放大显示微小变化
        this.bleCharts.qSignal = new Chart(document.getElementById('bleQSignalChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                ...chartOptions,
                plugins: { ...chartOptions.plugins, title: { display: true, text: '蓝牙 Q 通道实时信号 (自适应放大)' } },
                scales: {
                    x: { display: true, title: { display: true, text: '采样点' } },
                    y: {
                        display: true,
                        title: { display: true, text: '幅度 (V)' },
                        min: 1.2,    // 扩大初始范围，更清楚显示波峰变化
                        max: 2.8,    // 适度范围以突出波峰细节
                        beginAtZero: false
                    }
                }
            }
        });

        this.bleCharts.constellation = new Chart(document.getElementById('bleConstellationChart'), {
            type: 'scatter',
            data: { datasets: [] },
            options: { 
                ...chartOptions, 
                plugins: { ...chartOptions.plugins, title: { display: true, text: '蓝牙 I/Q 星座图' } },
                scales: {
                    x: { title: { display: true, text: 'I通道' } },
                    y: { title: { display: true, text: 'Q通道' } }
                }
            }
        });

        this.bleCharts.respiratory = new Chart(document.getElementById('bleRespiratoryChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: '蓝牙呼吸波形' } } }
        });

        this.bleCharts.heartbeat = new Chart(document.getElementById('bleHeartbeatChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: '蓝牙心跳波形' } } }
        });

        // 初始化 IMU(Gx/Gy/Gz) 图表
        const imuCanvas = document.getElementById('bleIMUChart');
        if (imuCanvas) {
            this.bleCharts.imu = new Chart(imuCanvas, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: '蓝牙 Gx/Gy/Gz 三轴变化' } } }
            });
        }

        // 初始化温度图表
        const tempCanvas = document.getElementById('bleTemperatureChart');
        if (tempCanvas) {
            this.bleCharts.temperature = new Chart(tempCanvas, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: { 
                    ...chartOptions, 
                    plugins: { ...chartOptions.plugins, title: { display: true, text: '蓝牙 温度变化 (°C)' } },
                    scales: {
                        x: { display: true, title: { display: true, text: '时间' } },
                        y: { 
                            display: true, 
                            title: { display: true, text: '温度 (°C)' },
                            min: 15, // 最小温度15°C
                            max: 45  // 最大温度45°C
                        }
                    }
                }
            });
        }
    }

    /**
     * 初始化蓝牙动态ECG画布
     */
    initializeBLEECG() {
        const resCanvas = document.getElementById('bleRespiratoryECGCanvas');
        const hbCanvas = document.getElementById('bleHeartbeatECGCanvas');
        if (!resCanvas || !hbCanvas) return;

        const ctxRes = resCanvas.getContext('2d');
        const ctxHb = hbCanvas.getContext('2d');
        this._bleECG = {
            res: { canvas: resCanvas, ctx: ctxRes, data: [], playing: false, cursor: 0 },
            hb:  { canvas: hbCanvas,  ctx: ctxHb,  data: [], playing: false, cursor: 0 },
            raf: null
        };

        const draw = () => {
            const { res, hb } = this._bleECG;
            [res, hb].forEach(track => {
                const { canvas, ctx } = track;
                const w = canvas.width = canvas.clientWidth || 600;
                const h = canvas.height = canvas.clientHeight || 160;
                ctx.clearRect(0, 0, w, h);
                ctx.strokeStyle = '#0aa'; ctx.lineWidth = 2; ctx.beginPath();
                const len = track.data.length;
                const view = 1000;
                const start = Math.max(0, len - view);
                for (let i = start; i < len; i++) {
                    const x = (i - start) / view * w;
                    const y = h/2 - (track.data[i] || 0) * (h*0.4);
                    if (i === start) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();
            });
            if (this._bleECG.res.playing || this._bleECG.hb.playing) {
                this._bleECG.raf = requestAnimationFrame(draw);
            } else {
                cancelAnimationFrame(this._bleECG.raf);
                this._bleECG.raf = null;
            }
        };

        this._bleECG.draw = draw;
    }

    /**
     * 初始化文件数据的动态ECG画布
     */
    initializeFileECG() {
        const resCanvas = document.getElementById('respiratoryECGCanvas');
        const hbCanvas = document.getElementById('heartbeatECGCanvas');
        if (!resCanvas || !hbCanvas) return;

        const ctxRes = resCanvas.getContext('2d');
        const ctxHb = hbCanvas.getContext('2d');

        // 从处理结果中获取数据
        const firstResult = this.processedResults.find(r => r.respiratoryWave && r.heartbeatWave);
        if (!firstResult) return;

        this._fileECG = {
            res: {
                canvas: resCanvas,
                ctx: ctxRes,
                data: Array.from(firstResult.respiratoryWave),
                playing: false,
                cursor: 0
            },
            hb: {
                canvas: hbCanvas,
                ctx: ctxHb,
                data: Array.from(firstResult.heartbeatWave),
                playing: false,
                cursor: 0
            },
            raf: null
        };

        const draw = () => {
            const { res, hb } = this._fileECG;

            // 绘制呼吸波形
            [res, hb].forEach(track => {
                const { canvas, ctx, data, cursor } = track;
                const w = canvas.width = canvas.clientWidth || 600;
                const h = canvas.height = canvas.clientHeight || 160;
                ctx.clearRect(0, 0, w, h);

                // 绘制网格
                ctx.strokeStyle = '#e0e0e0';
                ctx.lineWidth = 1;
                for (let x = 0; x < w; x += 20) {
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, h);
                    ctx.stroke();
                }
                for (let y = 0; y < h; y += 20) {
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(w, y);
                    ctx.stroke();
                }

                // 绘制波形
                if (data.length > 0) {
                    ctx.strokeStyle = track === res ? '#28a745' : '#dc3545';
                    ctx.lineWidth = 2;
                    ctx.beginPath();

                    const displayPoints = Math.min(200, data.length);
                    const startIdx = Math.max(0, cursor - displayPoints);

                    for (let i = 0; i < displayPoints && startIdx + i < data.length; i++) {
                        const x = (i / displayPoints) * w;
                        const value = data[startIdx + i];
                        const y = h/2 - (value * h/4); // 缩放并居中

                        if (i === 0) {
                            ctx.moveTo(x, y);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    }
                    ctx.stroke();
                }
            });

            // 更新游标
            if (res.playing || hb.playing) {
                this._fileECG.res.cursor = (this._fileECG.res.cursor + 1) % Math.max(1, this._fileECG.res.data.length);
                this._fileECG.hb.cursor = (this._fileECG.hb.cursor + 1) % Math.max(1, this._fileECG.hb.data.length);
                this._fileECG.raf = requestAnimationFrame(draw);
            } else {
                cancelAnimationFrame(this._fileECG.raf);
                this._fileECG.raf = null;
            }
        };

        this._fileECG.draw = draw;
    }

    /**
     * 初始化图表
     */
    initializeCharts() {
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true
                    }
                }
            }
        };

        // 初始化所有图表
        // I 通道图表
        this.charts.iSignal = new Chart(document.getElementById('iSignalChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                ...chartOptions,
                plugins: { ...chartOptions.plugins, title: { display: true, text: 'I 通道信号 (放大显示)' } },
                scales: {
                    x: { display: true, title: { display: true, text: '采样点' } },
                    y: {
                        display: true,
                        title: { display: true, text: '幅度 (V)' },
                        beginAtZero: false,
                        // 动态放大范围以显示更多细节
                        ticks: {
                            callback: function(value, index, values) {
                                return value.toFixed(4); // 显示更多小数位以观察微小变化
                            }
                        }
                    }
                }
            }
        });

        // Q 通道图表
        this.charts.qSignal = new Chart(document.getElementById('qSignalChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                ...chartOptions,
                plugins: { ...chartOptions.plugins, title: { display: true, text: 'Q 通道信号 (放大显示)' } },
                scales: {
                    x: { display: true, title: { display: true, text: '采样点' } },
                    y: {
                        display: true,
                        title: { display: true, text: '幅度 (V)' },
                        beginAtZero: false,
                        // 动态放大范围以显示更多细节
                        ticks: {
                            callback: function(value, index, values) {
                                return value.toFixed(4); // 显示更多小数位以观察微小变化
                            }
                        }
                    }
                }
            }
        });

        this.charts.constellation = new Chart(document.getElementById('constellationChart'), {
            type: 'scatter',
            data: { datasets: [] },
            options: { 
                ...chartOptions, 
                plugins: { ...chartOptions.plugins, title: { display: true, text: 'I/Q星座图' } },
                scales: {
                    x: { title: { display: true, text: 'I通道' } },
                    y: { title: { display: true, text: 'Q通道' } }
                }
            }
        });

        this.charts.respiratory = new Chart(document.getElementById('respiratoryChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: '呼吸波形' } } }
        });

        this.charts.heartbeat = new Chart(document.getElementById('heartbeatChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: '心跳波形' } } }
        });

        this.charts.heartRate = new Chart(document.getElementById('heartRateChart'), {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: { 
                ...chartOptions, 
                plugins: { ...chartOptions.plugins, title: { display: true, text: '心率分布' } },
                scales: {
                    x: { title: { display: true, text: '文件' } },
                    y: { title: { display: true, text: '心率 (bpm)' } }
                }
            }
        });

        this.charts.respRate = new Chart(document.getElementById('respRateChart'), {
            type: 'bar',
            data: { labels: [], datasets: [] },
            options: { 
                ...chartOptions, 
                plugins: { ...chartOptions.plugins, title: { display: true, text: '呼吸频率分布' } },
                scales: {
                    x: { title: { display: true, text: '文件' } },
                    y: { title: { display: true, text: '呼吸频率 (bpm)' } }
                }
            }
        });

        // 心率时间序列图表
        this.charts.heartRateTime = new Chart(document.getElementById('heartRateTimeChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: { 
                ...chartOptions, 
                plugins: { ...chartOptions.plugins, title: { display: true, text: '心率随时间变化' } },
                scales: {
                    x: { title: { display: true, text: '文件序号' } },
                    y: { title: { display: true, text: '心率 (bpm)' } }
                }
            }
        });

        // 呼吸频率时间序列图表
        this.charts.respRateTime = new Chart(document.getElementById('respRateTimeChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: { 
                ...chartOptions, 
                plugins: { ...chartOptions.plugins, title: { display: true, text: '呼吸频率随时间变化' } },
                scales: {
                    x: { title: { display: true, text: '文件序号' } },
                    y: { title: { display: true, text: '呼吸频率 (bpm)' } }
                }
            }
        });
    }

    /**
     * 更新图表
     */
    updateCharts(results) {
        if (results.length === 0) return;

        // 使用第一个成功的结果来显示波形
        const firstResult = results[0];

        // 如果是JSON数据，只更新心率和呼吸率时间序列图
        if (firstResult.dataType === 'json') {
            this.updateJsonCharts(results);
            return;
        }

        // 更新I/Q信号图
        const sampleSize = Math.min(1000, firstResult.iData.length);
        const indices = Array.from({length: sampleSize}, (_, i) => i);

        // 计算I通道数据的统计信息以实现动态放大
        const iDataSlice = firstResult.iData.slice(0, sampleSize);
        const iMin = Math.min(...iDataSlice);
        const iMax = Math.max(...iDataSlice);
        const iRange = iMax - iMin;
        const iPadding = iRange * 0.05; // 5% padding

        // 设置I通道Y轴动态范围（放大显示微小变化）
        const iYAxisMin = iMin - iPadding;
        const iYAxisMax = iMax + iPadding;

        // 更新 I 通道
        this.charts.iSignal.data = {
            labels: indices,
            datasets: [{
                label: 'I通道',
                data: Array.from(iDataSlice),
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1,
                pointRadius: 0
            }]
        };

        // 动态调整I通道Y轴范围以放大显示细节
        if (this.charts.iSignal.options.scales.y) {
            this.charts.iSignal.options.scales.y.min = iYAxisMin;
            this.charts.iSignal.options.scales.y.max = iYAxisMax;
        }

        this.charts.iSignal.update();

        // 计算Q通道数据的统计信息以实现动态放大
        const qDataSlice = firstResult.qData.slice(0, sampleSize);
        const qMin = Math.min(...qDataSlice);
        const qMax = Math.max(...qDataSlice);
        const qRange = qMax - qMin;
        const qPadding = qRange * 0.05; // 5% padding

        // 设置Q通道Y轴动态范围（放大显示微小变化）
        const qYAxisMin = qMin - qPadding;
        const qYAxisMax = qMax + qPadding;

        // 更新 Q 通道
        this.charts.qSignal.data = {
            labels: indices,
            datasets: [{
                label: 'Q通道',
                data: Array.from(qDataSlice),
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.1,
                pointRadius: 0
            }]
        };

        // 动态调整Q通道Y轴范围以放大显示细节
        if (this.charts.qSignal.options.scales.y) {
            this.charts.qSignal.options.scales.y.min = qYAxisMin;
            this.charts.qSignal.options.scales.y.max = qYAxisMax;
        }

        this.charts.qSignal.update();

        // 更新星座图
        const constellationSampleSize = Math.min(500, firstResult.iData.length);
        const step = Math.floor(firstResult.iData.length / constellationSampleSize);
        const constellationData = [];
        
        for (let i = 0; i < firstResult.iData.length; i += step) {
            constellationData.push({
                x: firstResult.iData[i],
                y: firstResult.qData[i]
            });
        }

        this.charts.constellation.data = {
            datasets: [
                {
                    label: 'I/Q数据点',
                    data: constellationData,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    pointRadius: 2
                },
                {
                    label: '圆心',
                    data: [{
                        x: firstResult.circleCenter[0],
                        y: firstResult.circleCenter[1]
                    }],
                    backgroundColor: 'red',
                    pointRadius: 8
                }
            ]
        };
        this.charts.constellation.update();

        // 更新呼吸波形（仅当有波形数据时）
        if (firstResult.respiratoryWave) {
            this.charts.respiratory.data = {
                labels: indices,
                datasets: [{
                    label: `呼吸波形 (${firstResult.respiratoryRate} bpm)`,
                    data: Array.from(firstResult.respiratoryWave.slice(0, sampleSize)),
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }]
            };
            this.charts.respiratory.update();
        }

        // 更新心跳波形（仅当有波形数据时）
        if (firstResult.heartbeatWave) {
            this.charts.heartbeat.data = {
                labels: indices,
                datasets: [{
                    label: `心跳波形 (${firstResult.heartRate} bpm)`,
                    data: Array.from(firstResult.heartbeatWave.slice(0, sampleSize)),
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                }]
            };
            this.charts.heartbeat.update();
        }

        // 更新心率分布图
        const fileNames = results.map(r => r.fileName.substring(0, 10) + '...');
        const heartRates = results.map(r => r.heartRate);

        this.charts.heartRate.data = {
            labels: fileNames,
            datasets: [{
                label: '心率 (bpm)',
                data: heartRates,
                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        };
        this.charts.heartRate.update();

        // 更新呼吸频率分布图
        const respRates = results.map(r => r.respiratoryRate);

        this.charts.respRate.data = {
            labels: fileNames,
            datasets: [{
                label: '呼吸频率 (bpm)',
                data: respRates,
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        };
        this.charts.respRate.update();

        // 更新心率时间序列图 - 使用真实的时间序列数据
        this.updateTimeSeriesCharts(results);
    }

    /**
     * 更新时间序列图表
     */
    updateTimeSeriesCharts(results) {
        if (results.length === 0) return;

        // 合并所有文件的时间序列数据
        let allHeartRateData = [];
        let allRespRateData = [];
        let allTimeLabels = [];
        let currentTime = 0;

        results.forEach((result, fileIndex) => {
            if (result.heartRateTimeSeries && result.respiratoryRateTimeSeries && result.timeAxis) {
                // 为每个文件的时间序列数据添加偏移
                const fileTimeOffset = currentTime;
                
                result.timeAxis.forEach((time, i) => {
                    const absoluteTime = fileTimeOffset + time;
                    allTimeLabels.push(`${Math.floor(absoluteTime / 60)}:${String(Math.floor(absoluteTime % 60)).padStart(2, '0')}`);
                    allHeartRateData.push(result.heartRateTimeSeries[i]);
                    allRespRateData.push(result.respiratoryRateTimeSeries[i]);
                });
                
                // 更新当前时间偏移（假设每个文件大约持续时间）
                const fileDuration = result.dataPoints / this.processor.fs;
                currentTime += fileDuration;
            }
        });

        // 如果没有时间序列数据，使用文件级别的数据
        if (allHeartRateData.length === 0) {
            allTimeLabels = results.map((_, index) => `文件${index + 1}`);
            allHeartRateData = results.map(r => r.heartRate);
            allRespRateData = results.map(r => r.respiratoryRate);
        }

        // 更新心率时间序列图
        this.charts.heartRateTime.data = {
            labels: allTimeLabels,
            datasets: [{
                label: '心率变化 (bpm)',
                data: allHeartRateData,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                tension: 0.3,
                fill: true,
                pointRadius: 2,
                pointHoverRadius: 6,
                borderWidth: 2
            }]
        };
        this.charts.heartRateTime.update();

        // 更新呼吸频率时间序列图
        this.charts.respRateTime.data = {
            labels: allTimeLabels,
            datasets: [{
                label: '呼吸频率变化 (bpm)',
                data: allRespRateData,
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.3,
                fill: true,
                pointRadius: 2,
                pointHoverRadius: 6,
                borderWidth: 2
            }]
        };
        this.charts.respRateTime.update();
    }

    /**
     * 更新JSON数据的图表（只显示心率和呼吸率时间序列）
     */
    updateJsonCharts(results) {
        const jsonResults = results.filter(r => r.dataType === 'json');
        if (jsonResults.length === 0) return;

        const firstResult = jsonResults[0];

        // 只更新心率和呼吸率时间序列图
        if (firstResult.hrData && firstResult.hrData.length > 0) {
            const hrTimeLabels = Array.from({length: firstResult.hrData.length}, (_, i) => i + 1);
            this.charts.heartRateTime.data = {
                labels: hrTimeLabels,
                datasets: [{
                    label: '心率 (bpm)',
                    data: firstResult.hrData,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                }]
            };
            this.charts.heartRateTime.update();
        }

        if (firstResult.rrData && firstResult.rrData.length > 0) {
            const rrTimeLabels = Array.from({length: firstResult.rrData.length}, (_, i) => i + 1);
            this.charts.respRateTime.data = {
                labels: rrTimeLabels,
                datasets: [{
                    label: '呼吸频率 (bpm)',
                    data: firstResult.rrData,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }]
            };
            this.charts.respRateTime.update();
        }

        // 清空其他图表（雷达信号相关）
        this.clearRadarCharts();
    }

    /**
     * JSON数据时隐藏雷达相关图表
     */
    setChartVisibilityForJson(hasJsonData) {
        const hideIds = [
            'iqChart',
            'constellationChart',
            'respiratoryChart',
            'heartbeatChart',
            'heartRateChart',
            'respRateChart'
        ];
        const ecgSection = document.querySelector('.ecg-section');

        hideIds.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.parentElement) {
                el.parentElement.style.display = hasJsonData ? 'none' : 'block';
            }
        });

        if (ecgSection) {
            ecgSection.style.display = hasJsonData ? 'none' : 'block';
        }
    }

    /**
     * 清空雷达信号相关的图表（用于JSON数据时）
     */
    clearRadarCharts() {
        // 清空I/Q信号图
        if (this.charts.iq) {
            this.charts.iq.data = { labels: [], datasets: [] };
            this.charts.iq.update();
        }

        // 清空星座图
        if (this.charts.constellation) {
            this.charts.constellation.data = { datasets: [] };
            this.charts.constellation.update();
        }

        // 清空呼吸波形图
        if (this.charts.respiratory) {
            this.charts.respiratory.data = { labels: [], datasets: [] };
            this.charts.respiratory.update();
        }

        // 清空心跳波形图
        if (this.charts.heartbeat) {
            this.charts.heartbeat.data = { labels: [], datasets: [] };
            this.charts.heartbeat.update();
        }

        // 清空心率分布图
        if (this.charts.heartRate) {
            this.charts.heartRate.data = { labels: [], datasets: [] };
            this.charts.heartRate.update();
        }

        // 清空呼吸频率分布图
        if (this.charts.respRate) {
            this.charts.respRate.data = { labels: [], datasets: [] };
            this.charts.respRate.update();
        }
    }

    /**
     * 更新结果表格
     */
    updateResultsTable() {
        const tbody = document.getElementById('resultsTableBody');
        tbody.innerHTML = '';

        this.processedResults.forEach(result => {
            const row = document.createElement('tr');

            if (result.status === 'success') {
                if (result.dataType === 'json') {
                    // JSON数据格式
                    row.innerHTML = `
                        <td>${result.fileName}</td>
                        <td>${result.dataPoints.toLocaleString()}</td>
                        <td>${result.heartRate}</td>
                        <td>${result.respiratoryRate}</td>
                        <td>--</td>
                        <td>--</td>
                        <td>--</td>
                        <td><span class="status-success">JSON数据</span></td>
                    `;
                } else {
                    // TXT数据格式（原始雷达数据）
                    row.innerHTML = `
                        <td>${result.fileName}</td>
                        <td>${result.dataPoints.toLocaleString()}</td>
                        <td>${result.heartRate}</td>
                        <td>${result.respiratoryRate}</td>
                        <td>${result.circleCenter[0].toFixed(4)}</td>
                        <td>${result.circleCenter[1].toFixed(4)}</td>
                        <td>${result.circleRadius.toFixed(4)}</td>
                        <td><span class="status-success">雷达数据</span></td>
                    `;
                }
            } else {
                row.innerHTML = `
                    <td>${result.fileName}</td>
                    <td>--</td>
                    <td>--</td>
                    <td>--</td>
                    <td>--</td>
                    <td>--</td>
                    <td>--</td>
                    <td><span class="status-error">失败: ${result.error}</span></td>
                `;
            }
            
            tbody.appendChild(row);
        });
    }

    /**
     * 导出结果为CSV
     */
    exportResults() {
        if (this.processedResults.length === 0) {
            this.showMessage('没有可导出的结果', 'warning');
            return;
        }

        const headers = ['文件名', '数据点数', '心率(bpm)', '呼吸频率(bpm)', '圆心I', '圆心Q', '圆半径', '状态'];
        const csvContent = [
            headers.join(','),
            ...this.processedResults.map(result => {
                if (result.status === 'success') {
                    return [
                        result.fileName,
                        result.dataPoints,
                        result.heartRate,
                        result.respiratoryRate,
                        result.circleCenter[0].toFixed(4),
                        result.circleCenter[1].toFixed(4),
                        result.circleRadius.toFixed(4),
                        '成功'
                    ].join(',');
                } else {
                    return [
                        result.fileName,
                        '--', '--', '--', '--', '--', '--',
                        `失败: ${result.error}`
                    ].join(',');
                }
            })
        ].join('\n');

        this.downloadFile(csvContent, 'radar_processing_results.csv', 'text/csv');
        this.showMessage('结果已导出为CSV文件', 'success');
    }

    /**
     * 导出图表
     */
    exportCharts() {
        Object.keys(this.charts).forEach(chartName => {
            const canvas = this.charts[chartName].canvas;
            const link = document.createElement('a');
            link.download = `${chartName}_chart.png`;
            link.href = canvas.toDataURL();
            link.click();
        });
        
        this.showMessage('图表已导出为PNG文件', 'success');
    }

    /**
     * 切换设置面板
     */
    toggleSettings() {
        const panel = document.getElementById('settingsPanel');
        panel.classList.toggle('open');
    }

    /**
     * 应用设置（关键：把采样率写回处理器）
     */
    applySettings() {
        const srEl = document.getElementById('samplingRate');
        const sr = srEl ? parseInt(srEl.value, 10) : NaN;
        const samplingRate = Number.isFinite(sr) && sr > 0 ? sr : 100;
        if (this.processor) this.processor.fs = samplingRate;
        
        // 应用心率平滑参数
        const smoothEl = document.getElementById('heartRateSmoothing');
        const smooth = smoothEl ? parseInt(smoothEl.value, 10) : NaN;
        if (Number.isFinite(smooth) && smooth >= 5 && smooth <= 60) {
            this.historyMaxLength = smooth;
        }
        
        const deltaEl = document.getElementById('heartRateDelta');
        const delta = deltaEl ? parseInt(deltaEl.value, 10) : NaN;
        if (Number.isFinite(delta) && delta >= 5 && delta <= 30) {
            this.heartRateDelta = delta;
        }
        
        this.addBLELog(`⚙️ 已应用设置：采样率=${samplingRate}Hz, 平滑长度=${this.historyMaxLength}, 变化阈值=${this.heartRateDelta}bpm`);
        this.showMessage(`已应用设置：采样率${samplingRate}Hz, 心率平滑${this.historyMaxLength}次, 阈值${this.heartRateDelta}bpm`, 'success');
        this.toggleSettings();
    }

    /**
     * 显示/隐藏加载指示器
     */
    showLoading(show) {
        document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
    }

    /**
     * 显示/隐藏状态区域
     */
    showStatus(show) {
        document.getElementById('statusSection').style.display = show ? 'block' : 'none';
    }

    /**
     * 隐藏结果区域
     */
    hideResults() {
        document.getElementById('resultsSection').style.display = 'none';
    }

    /**
     * 更新进度条
     */
    updateProgress(percentage, text) {
        document.getElementById('progressFill').style.width = `${percentage}%`;
        document.getElementById('progressText').textContent = text;
    }

    /**
     * 添加状态日志
     */
    addStatusLog(message) {
        const log = document.getElementById('statusLog');
        const timestamp = new Date().toLocaleTimeString();
        log.innerHTML += `<div>[${timestamp}] ${message}</div>`;
        log.scrollTop = log.scrollHeight;
    }

    /**
     * 显示消息
     */
    showMessage(message, type = 'info') {
        // 简单的消息显示，可以用更复杂的通知系统替换
        const colors = {
            success: '#28a745',
            warning: '#ffc107',
            error: '#dc3545',
            info: '#17a2b8'
        };

        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${colors[type]};
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            z-index: 3000;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        `;
        messageDiv.textContent = message;
        
        document.body.appendChild(messageDiv);
        
        setTimeout(() => {
            messageDiv.remove();
        }, 3000);
    }

    /**
     * 下载文件
     */
    downloadFile(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
    }

    /**
     * 汇总本次蓝牙录制的窗口HR/RR统计与平均值
     */
    _buildBluetoothSessionStats() {
        const history = this._bleWindowHistory || [];
        const startTs = this.bleRecordingStartTime ? this.bleRecordingStartTime.toISOString() : new Date().toISOString();
        const endTs = new Date().toISOString();
        const durationSec = this.bleRecordingStartTime ? Math.round((Date.now() - this.bleRecordingStartTime.getTime())/1000) : 0;

        const hrList = history.map(h => h.heartRate).filter(v => Number.isFinite(v) && v > 0);
        const rrList = history.map(h => h.respiratoryRate).filter(v => Number.isFinite(v) && v > 0);
        const avgHR = hrList.length ? Math.round(hrList.reduce((a,b)=>a+b,0)/hrList.length) : 0;
        const avgRR = rrList.length ? Math.round(rrList.reduce((a,b)=>a+b,0)/rrList.length) : 0;

        return {
            startTime: startTs,
            endTime: endTs,
            durationSeconds: durationSec,
            windowCount: history.length,
            windows: history,
            average: { heartRate: avgHR, respiratoryRate: avgRR }
        };
    }

    /**
     * 格式化文件大小
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // ======= 蓝牙专用函数 =======

    /**
     * 打印原始数据到日志
     */
    printRawData(line) {
        const log = document.getElementById('bleRawDataLog');
        if (!log) return;
        
        const ts = new Date().toLocaleTimeString();
        const trimmed = line.trim();
        this._bleRawLines.push(`[${ts}] ${trimmed}`);
        if (this._bleRawLines.length > 50) this._bleRawLines.splice(0, this._bleRawLines.length - 50);

        // 节流渲染
        if (this._bleRawRenderTimer) return;
        this._bleRawRenderTimer = setTimeout(() => {
            this._bleRawRenderTimer = null;
            log.style.whiteSpace = 'pre-wrap';
            log.textContent = `原始数据:\n${this._bleRawLines.join('\n')}\n`;
            log.scrollTop = log.scrollHeight;
        }, 200);
    }

    /**
     * 开始蓝牙连接计时
     */
    startBluetoothTimer() {
        this.stopBluetoothTimer(); // 防止重复计时器
        
        this.bleConnectTimer = setInterval(() => {
            if (this.bleConnectStartTime) {
                const elapsedMs = Date.now() - this.bleConnectStartTime;
                const minutes = Math.floor(elapsedMs / 60000);
                const seconds = Math.floor((elapsedMs % 60000) / 1000);
                document.getElementById('bleConnectTime').textContent = `${minutes} 分 ${seconds} 秒`;
            }
        }, 1000);
    }

    /**
     * 启动接收看门狗：3秒无数据则尝试判定掉线
     */
    startRxWatchdog() {
        if (this.rxWatchdogTimer) return;
        this.rxWatchdogTimer = setInterval(async () => {
            try {
                if (this.bleConnected) {
                    const idleMs = Date.now() - (this.lastBleRxTs || 0);
                    if (this.lastBleRxTs > 0 && idleMs > 3000) {
                        this.addBLELog('⚠️ 3秒未收到数据，判定连接已中断，正在断开...');
                        if (window.BLE && typeof BLE.disconnect === 'function') {
                            await BLE.disconnect();
                        }
                    }
                }
            } catch (e) {
                // 忽略看门狗异常
            }
        }, 1000);
    }

    stopRxWatchdog() {
        if (this.rxWatchdogTimer) {
            clearInterval(this.rxWatchdogTimer);
            this.rxWatchdogTimer = null;
        }
    }

    /**
     * 停止蓝牙连接计时
     */
    stopBluetoothTimer() {
        if (this.bleConnectTimer) {
            clearInterval(this.bleConnectTimer);
            this.bleConnectTimer = null;
        }
    }

    /**
     * 重置蓝牙数据
     */
    resetBluetoothData() {
        this.bleBufferI = [];
        this.bleBufferQ = [];
        this.bleBufferIMU_X = [];
        this.bleBufferIMU_Y = [];
        this.bleBufferIMU_Z = [];
        this.bleBufferTemperature = [];
        this.bleBufferTimestamps = [];
        this.bleDataCount = 0;
        
        // 重置录制相关数据
        this.bleRecordingFlag = 0;
        this.bleRecordingData = [];
        this.bleRecordingRawData = [];
        this.bleRecordingStartTime = null;
        this._bleWindowHistory = [];
        
        // 重置心率平滑历史（循环数组）
        this.heartRateHistory.fill(70);
        this.respiratoryHistory.fill(18);
        this.historyIndex = 0;
        this.lastStableHeartRate = 70;
        this.lastStableRespRate = 18;

        // 重置自适应Y轴状态
        this.adaptiveSampleCount = 0;
        this.adaptiveLastMinI = Infinity;
        this.adaptiveLastMaxI = -Infinity;
        this.adaptiveLastMinQ = Infinity;
        this.adaptiveLastMaxQ = -Infinity;
        this.adaptiveStabilized = false;

        // 重置图表Y轴到初始范围
        if (this.bleCharts.iSignal) {
            this.bleCharts.iSignal.options.scales.y.min = 0;
            this.bleCharts.iSignal.options.scales.y.max = 4.0;
        }
        if (this.bleCharts.qSignal) {
            this.bleCharts.qSignal.options.scales.y.min = 0;
            this.bleCharts.qSignal.options.scales.y.max = 4.0;
        }

        // 重置丢包统计
        this.bleStats = {
            startRxTs: 0,
            lastRxTs: 0,
            received: 0,
            expected: 0,
            missed: 0,
            lastGapMs: 0,
            gapEmaMs: 0,
            gapJitterEmaMs: 0,
            lastSeq: null,
            seqBased: false
        };

        // 清空统计显示
        const fsEl = document.getElementById('bleActualFs');
        const lossEl = document.getElementById('blePacketLoss');
        const jitterEl = document.getElementById('bleJitter');
        if (fsEl) fsEl.textContent = '-- Hz';
        if (lossEl) lossEl.textContent = '-- %';
        if (jitterEl) jitterEl.textContent = '-- ms';
        
        // 清空显示
        document.getElementById('bleDataCount').textContent = '0';
        document.getElementById('bleTotalDataPoints').textContent = '0';
        document.getElementById('bleCurrentHR').textContent = '-- bpm';
        document.getElementById('bleCurrentResp').textContent = '-- bpm';
        const iqEl = document.getElementById('bleCurrentIQ');
        if (iqEl) iqEl.textContent = '--';
        document.getElementById('bleAvgHeartRate').textContent = '-- bpm';
        document.getElementById('bleAvgRespRate').textContent = '-- bpm';
        const tempEl = document.getElementById('bleCurrentTemp');
        const avgTempEl = document.getElementById('bleAvgTemp');
        if (tempEl) tempEl.textContent = '-- °C';
        if (avgTempEl) avgTempEl.textContent = '-- °C';
        
        // 清空原始数据日志/缓存
        this._bleRawLines = [];
        const rawLog = document.getElementById('bleRawDataLog');
        if (rawLog) {
            rawLog.style.whiteSpace = 'pre-wrap';
            rawLog.textContent = '原始数据:\n';
        }

        // 清空 BLE 事件日志/缓存
        this._bleLogLines = [];
        const bleLog = document.getElementById('bleLog');
        if (bleLog) {
            bleLog.style.whiteSpace = 'pre-line';
            bleLog.textContent = '';
        }
        
        // 更新按钮状态
        this.updateBLEButtons();
    }

    /**
     * 切换蓝牙数据录制状态 (参考main.py的_set_button_att方法)
     */
    toggleBluetoothRecording() {
        // 切换录制状态，类似于 main.py 中的 flag_record = (1 + flag_record) % 2
        this.bleRecordingFlag = (1 + this.bleRecordingFlag) % 2;
        
        if (this.bleRecordingFlag === 1) {
            // 开始录制
            this.bleRecordingData = [];
            this.bleRecordingRawData = [];
            this.bleRecordingStartTime = new Date();

            // 生成录制文件名 (参考main.py的命名规则)
            const timestamp = this.bleRecordingStartTime.toISOString()
                .slice(0, 16).replace('T', '-').replace(/:/g, '-');

            // 记录开始时的心率和呼吸率
            const currentHR = this.currentHeartRate || 0;
            const currentRR = this.currentRespiratoryRate || 0;
            const startTimestamp = new Date().toISOString().replace('T', ' ').slice(0, 19);

            // 在录制数据开头添加元数据信息
            this.bleRecordingData.push(`# 录制开始时间: ${startTimestamp}`);
            this.bleRecordingData.push(`# 开始时心率: ${currentHR} bpm, 呼吸率: ${currentRR} bpm`);
            this.bleRecordingData.push(`# 数据格式: timestamp ADC_I ADC_Q Acc_X Acc_Y Acc_Z I_voltage Q_voltage IMU_x IMU_y IMU_z temperature`);
            this.bleRecordingData.push(`# 原始数据开始`);

            this.addBLELog(`🔴 开始录制数据 - ${timestamp}`);
            this.addBLELog(`💓 开始时心率: ${currentHR} bpm, 呼吸率: ${currentRR} bpm`);
            this.addBLELog('📝 实时保存到内存，结束时将下载处理后数据和原始数据文件');
            
        } else {
            // 结束录制并自动下载文件
            const recordingEndTime = new Date();
            const duration = ((recordingEndTime - this.bleRecordingStartTime) / 1000).toFixed(1);

            // 记录结束时的心率和呼吸率
            const endHR = this.currentHeartRate || 0;
            const endRR = this.currentRespiratoryRate || 0;
            const endTimestamp = new Date().toISOString().replace('T', ' ').slice(0, 19);

            // 在录制数据末尾添加结束信息
            this.bleRecordingData.push(`# 原始数据结束`);
            this.bleRecordingData.push(`# 录制结束时间: ${endTimestamp}`);
            this.bleRecordingData.push(`# 结束时心率: ${endHR} bpm, 呼吸率: ${endRR} bpm`);
            this.bleRecordingData.push(`# 录制统计: 总时长 ${duration}秒, 数据点数 ${this.bleRecordingData.filter(line => !line.startsWith('#')).length}`);

            // 生成文件内容 (参考main.py的数据格式)
            let fileContent = '';
            for (const line of this.bleRecordingData) {
                fileContent += line + '\n';
            }
            
            // 生成文件名 (参考main.py的命名格式)
            const timestamp = this.bleRecordingStartTime.toISOString()
                .slice(0, 16).replace('T', '-').replace(/:/g, '-');
            const filename = `bluetooth_record_${timestamp}.txt`;
            
            // 自动下载处理后数据文件
            this.downloadFile(fileContent, filename, 'text/plain');

            // 生成并下载原始数据文件
            const rawFileContent = this.bleRecordingRawData.join('\n');
            const rawFilename = `bluetooth_raw_${timestamp}.txt`;
            this.downloadFile(rawFileContent, rawFilename, 'text/plain');
            this.addBLELog(`📄 已保存原始数据: ${rawFilename} (${this.bleRecordingRawData.length} 行)`);

            // 保存简化的录制统计（只包含最终结果，不包含详细窗口数据）
            const simplifiedStats = {
                startTime: this.bleRecordingStartTime.toISOString(),
                endTime: new Date().toISOString(),
                durationSeconds: parseFloat(duration),
                finalHeartRate: endHR,
                finalRespiratoryRate: endRR,
                dataPoints: this.bleRecordingData.filter(line => !line.startsWith('#')).length,
                note: '心率呼吸率只保存显示的最终结果'
            };
            const statsJson = JSON.stringify(simplifiedStats, null, 2);
            const statsFilename = `bluetooth_record_${timestamp}_stats.json`;
            this.downloadFile(statsJson, statsFilename, 'application/json');
            this.addBLELog(`📈 已保存录制统计: ${statsFilename}`);

            // 显示录制统计
            this.addBLELog(`🟢 录制结束 - 时长: ${duration}秒`);
            this.addBLELog(`💓 结束时心率: ${endHR} bpm, 呼吸率: ${endRR} bpm`);
            // 计算实际数据点数（排除注释行）
            const dataPointCount = this.bleRecordingData.filter(line => !line.startsWith('#')).length;
            this.addBLELog(`💾 已保存处理后数据: ${filename} (${dataPointCount} 数据点 + 元数据)`);
            this.addBLELog(`📂 总共下载3个文件: 处理后数据、原始数据、统计信息`);
            
            // 清空录制缓存
            this.bleRecordingData = [];
            this.bleRecordingRawData = [];
            this.bleRecordingStartTime = null;
        }
        
        // 更新按钮状态
        this.updateBLEButtons();
    }

    /**
     * 更新蓝牙实时图表
     */
    updateBluetoothLiveCharts() {
        // 检查所有必需的图表是否已初始化
        if (!this.bleCharts.iSignal || !this.bleCharts.qSignal || !this.bleCharts.constellation) {
            console.warn('❌ 蓝牙图表未初始化：', {
                iSignal: !!this.bleCharts.iSignal,
                qSignal: !!this.bleCharts.qSignal,
                constellation: !!this.bleCharts.constellation,
                imu: !!this.bleCharts.imu,
                temperature: !!this.bleCharts.temperature
            });
            return;
        }

        // 调试：检查数据缓冲区状态
        if (this.bleDataCount === 10) {
            console.log('📊 数据缓冲区状态:', {
                I长度: this.bleBufferI.length,
                Q长度: this.bleBufferQ.length,
                IMU_X长度: this.bleBufferIMU_X.length,
                IMU_Y长度: this.bleBufferIMU_Y.length,
                IMU_Z长度: this.bleBufferIMU_Z.length,
                温度长度: this.bleBufferTemperature.length
            });
        }
        const len = this.bleBufferI.length;
        if (len < 10) return;

        // 自适应Y轴调节逻辑（提高实时性：增加检测频率）
        if (this.adaptiveYAxisEnabled && this.bleDataCount % 2 === 0) { // 每2个数据点计算一次，提高响应速度
            this.adaptiveSampleCount++;

            // 收集最近数据的范围
            const recentDataSize = Math.min(len, this.adaptiveStabilizeWindow);
            const startIdx = len - recentDataSize;
            const recentI = this.bleBufferI.slice(startIdx);
            const recentQ = this.bleBufferQ.slice(startIdx);

            const currentMinI = Math.min(...recentI);
            const currentMaxI = Math.max(...recentI);
            const currentMinQ = Math.min(...recentQ);
            const currentMaxQ = Math.max(...recentQ);

            // 检测信号范围是否发生显著变化（需要重新自适应）
            let rangeChanged = false;
            if (this.adaptiveStabilized) {
                const currentRangeI = currentMaxI - currentMinI;
                const currentRangeQ = currentMaxQ - currentMinQ;
                const stabilizedRangeI = this.adaptiveLastMaxI - this.adaptiveLastMinI;
                const stabilizedRangeQ = this.adaptiveLastMaxQ - this.adaptiveLastMinQ;

                // 检查是否处于微小波动状态（Y轴范围≤0.1）
                const isMicroFluctuationMode = (
                    this.bleCharts.iSignal && this.bleCharts.qSignal &&
                    (this.bleCharts.iSignal.options.scales.y.max - this.bleCharts.iSignal.options.scales.y.min) <= 0.1 ||
                    (this.bleCharts.qSignal.options.scales.y.max - this.bleCharts.qSignal.options.scales.y.min) <= 0.1
                );

                if (isMicroFluctuationMode) {
                    // 微小波动模式下，提高重置阈值，避免频繁重置
                    const rangeChangeThreshold = 1.0; // 从0.2提高到1.0，更宽容
                    const offsetThresholdI = Math.max(stabilizedRangeI * 0.5, 0.1); // 从0.15提高到0.5，从0.05提高到0.1
                    const offsetThresholdQ = Math.max(stabilizedRangeQ * 0.5, 0.1);

                    if (Math.abs(currentRangeI - stabilizedRangeI) / Math.max(stabilizedRangeI, 0.01) > rangeChangeThreshold ||
                        Math.abs(currentRangeQ - stabilizedRangeQ) / Math.max(stabilizedRangeQ, 0.01) > rangeChangeThreshold) {
                        rangeChanged = true;
                        console.log(`🔄 [微小模式]信号范围变化: I(${stabilizedRangeI.toFixed(4)}→${currentRangeI.toFixed(4)}), Q(${stabilizedRangeQ.toFixed(4)}→${currentRangeQ.toFixed(4)})`);
                    }

                    if (Math.abs(currentMinI - this.adaptiveLastMinI) > offsetThresholdI ||
                        Math.abs(currentMaxI - this.adaptiveLastMaxI) > offsetThresholdI ||
                        Math.abs(currentMinQ - this.adaptiveLastMinQ) > offsetThresholdQ ||
                        Math.abs(currentMaxQ - this.adaptiveLastMaxQ) > offsetThresholdQ) {
                        rangeChanged = true;
                        console.log(`🔄 [微小模式]信号偏移变化: I(${this.adaptiveLastMinI.toFixed(4)}-${this.adaptiveLastMaxI.toFixed(4)} → ${currentMinI.toFixed(4)}-${currentMaxI.toFixed(4)}), Q(${this.adaptiveLastMinQ.toFixed(4)}-${this.adaptiveLastMaxQ.toFixed(4)} → ${currentMinQ.toFixed(4)}-${currentMaxQ.toFixed(4)})`);
                    }
                } else {
                    // 正常模式下的重置逻辑（保持原有敏感度）
                    const rangeChangeThreshold = 0.2;
                    if (Math.abs(currentRangeI - stabilizedRangeI) / Math.max(stabilizedRangeI, 0.01) > rangeChangeThreshold ||
                        Math.abs(currentRangeQ - stabilizedRangeQ) / Math.max(stabilizedRangeQ, 0.01) > rangeChangeThreshold) {
                        rangeChanged = true;
                        console.log(`🔄 检测到信号范围变化: I(${stabilizedRangeI.toFixed(3)}→${currentRangeI.toFixed(3)}), Q(${stabilizedRangeQ.toFixed(3)}→${currentRangeQ.toFixed(3)})`);
                    }

                    const offsetThresholdI = Math.max(stabilizedRangeI * 0.15, 0.05);
                    const offsetThresholdQ = Math.max(stabilizedRangeQ * 0.15, 0.05);
                    if (Math.abs(currentMinI - this.adaptiveLastMinI) > offsetThresholdI ||
                        Math.abs(currentMaxI - this.adaptiveLastMaxI) > offsetThresholdI ||
                        Math.abs(currentMinQ - this.adaptiveLastMinQ) > offsetThresholdQ ||
                        Math.abs(currentMaxQ - this.adaptiveLastMaxQ) > offsetThresholdQ) {
                        rangeChanged = true;
                        console.log(`🔄 检测到信号偏移变化: I(${this.adaptiveLastMinI.toFixed(3)}-${this.adaptiveLastMaxI.toFixed(3)} → ${currentMinI.toFixed(3)}-${currentMaxI.toFixed(3)}), Q(${this.adaptiveLastMinQ.toFixed(3)}-${this.adaptiveLastMaxQ.toFixed(3)} → ${currentMinQ.toFixed(3)}-${currentMaxQ.toFixed(3)})`);
                    }
                }

                // 检测信号是否完全超出当前显示范围（需要立即响应）
                const currentChartMinI = this.bleCharts.iSignal?.options.scales.y.min || 0;
                const currentChartMaxI = this.bleCharts.iSignal?.options.scales.y.max || 4;
                const currentChartMinQ = this.bleCharts.qSignal?.options.scales.y.min || 0;
                const currentChartMaxQ = this.bleCharts.qSignal?.options.scales.y.max || 4;

                if (currentMinI < currentChartMinI || currentMaxI > currentChartMaxI ||
                    currentMinQ < currentChartMinQ || currentMaxQ > currentChartMaxQ) {
                    rangeChanged = true;
                    console.log('🔄 检测到信号超出当前显示范围，立即重新自适应');
                }
            }

            // 如果检测到范围变化，重置自适应状态
            if (rangeChanged) {
                this.adaptiveSampleCount = 0;
                this.adaptiveLastMinI = Infinity;
                this.adaptiveLastMaxI = -Infinity;
                this.adaptiveLastMinQ = Infinity;
                this.adaptiveLastMaxQ = -Infinity;
                this.adaptiveStabilized = false;

                // 重置图表范围：根据当前状态智能选择范围
                if (this.bleCharts.iSignal && this.bleCharts.qSignal) {
                    // 检查之前是否处于微小波动模式
                    const wasMicroMode = (
                        (this.bleCharts.iSignal.options.scales.y.max - this.bleCharts.iSignal.options.scales.y.min) <= 0.1 ||
                        (this.bleCharts.qSignal.options.scales.y.max - this.bleCharts.qSignal.options.scales.y.min) <= 0.1
                    );

                    if (wasMicroMode) {
                        // 如果之前是微小模式，重置到稍微大一点的范围，但保持相对较小
                        this.bleCharts.iSignal.options.scales.y.min = Math.max(0, currentMinI - 0.1);
                        this.bleCharts.iSignal.options.scales.y.max = currentMaxI + 0.1;
                        this.bleCharts.qSignal.options.scales.y.min = Math.max(0, currentMinQ - 0.1);
                        this.bleCharts.qSignal.options.scales.y.max = currentMaxQ + 0.1;
                        console.log('🔄 微小模式重置：保持相对较小的范围');
                    } else {
                        // 正常重置到稍宽的初始范围
                        this.bleCharts.iSignal.options.scales.y.min = 1.0;
                        this.bleCharts.iSignal.options.scales.y.max = 3.0;
                        this.bleCharts.qSignal.options.scales.y.min = 1.0;
                        this.bleCharts.qSignal.options.scales.y.max = 3.0;
                    }
                }
                console.log('🔄 自适应Y轴已重置，重新开始调节');
            }

            // 如果还没稳定，更新范围
            if (!this.adaptiveStabilized) {
                this.adaptiveLastMinI = Math.min(this.adaptiveLastMinI, currentMinI);
                this.adaptiveLastMaxI = Math.max(this.adaptiveLastMaxI, currentMaxI);
                this.adaptiveLastMinQ = Math.min(this.adaptiveLastMinQ, currentMinQ);
                this.adaptiveLastMaxQ = Math.max(this.adaptiveLastMaxQ, currentMaxQ);

                // 检查是否达到稳定阈值
                if (this.adaptiveSampleCount >= this.adaptiveStabilizeThreshold) {
                    // 全程自适应：稳定后设置极紧凑范围以显示微小细节
                    const rangeI = this.adaptiveLastMaxI - this.adaptiveLastMinI;
                    const rangeQ = this.adaptiveLastMaxQ - this.adaptiveLastMinQ;

                    // 详细调试信息
                    console.log(`🔍 自适应调试: 样本数=${this.adaptiveSampleCount}, I范围=${rangeI.toFixed(4)}V (${this.adaptiveLastMinI.toFixed(3)}-${this.adaptiveLastMaxI.toFixed(3)}), Q范围=${rangeQ.toFixed(4)}V (${this.adaptiveLastMinQ.toFixed(3)}-${this.adaptiveLastMaxQ.toFixed(3)})`);

                    // 简化波动性评估：使用数据范围的简单比例来代替复杂标准差计算
                    const dataRangeI = this.adaptiveLastMaxI - this.adaptiveLastMinI;
                    const dataRangeQ = this.adaptiveLastMaxQ - this.adaptiveLastMinQ;

                    // 使用数据范围的10%作为波动性估计（简化计算）
                    const stdI = dataRangeI * 0.1;
                    const stdQ = dataRangeQ * 0.1;

                    let newMinI, newMaxI, newMinQ, newMaxQ;

                    // 实时微小波动检测：波动小于0.2V时启用0.1单位Y轴控制
                    const microFluctuationThreshold = 0.2; // 微小波动阈值：总范围0.2V
                    if (rangeI <= microFluctuationThreshold || rangeQ <= microFluctuationThreshold) {
                        // 计算信号中心点
                        const centerI = (this.adaptiveLastMinI + this.adaptiveLastMaxI) / 2;
                        const centerQ = (this.adaptiveLastMinQ + this.adaptiveLastMaxQ) / 2;

                        // 设置0.1单位长度的Y轴范围（±0.05），最大化放大微小波动
                        newMinI = Math.max(0, centerI - 0.05);
                        newMaxI = centerI + 0.05;
                        newMinQ = Math.max(0, centerQ - 0.05);
                        newMaxQ = centerQ + 0.05;

                        console.log(`🔬 微小波动检测触发! I范围=${rangeI.toFixed(4)}V, Q范围=${rangeQ.toFixed(4)}V，启用0.1单位Y轴控制`);
                        console.log(`📍 信号中心: I=${centerI.toFixed(4)}V, Q=${centerQ.toFixed(4)}V`);
                        console.log(`🎨 Y轴设置: I=[${newMinI.toFixed(4)}, ${newMaxI.toFixed(4)}], Q=[${newMinQ.toFixed(4)}, ${newMaxQ.toFixed(4)}]`);
                    } else {
                        // 自适应完成后，设置适度紧凑的范围来更清楚显示波峰
                        // 使用标准差的3倍作为余量，但最大不超过数据范围的20%，最小0.01V
                        const detailPaddingI = Math.max(0.01, Math.min(stdI * 3, rangeI * 0.20));
                        const detailPaddingQ = Math.max(0.01, Math.min(stdQ * 3, rangeQ * 0.20));

                        // 设置极紧凑的范围：数据范围 ± 很小的余量
                        newMinI = Math.max(0, this.adaptiveLastMinI - detailPaddingI);
                        newMaxI = this.adaptiveLastMaxI + detailPaddingI;
                        newMinQ = Math.max(0, this.adaptiveLastMinQ - detailPaddingQ);
                        newMaxQ = this.adaptiveLastMaxQ + detailPaddingQ;

                        console.log(`🔄 标准自适应: I余量=${detailPaddingI.toFixed(3)}V, Q余量=${detailPaddingQ.toFixed(3)}V`);
                    }

                    // 更新I通道Y轴
                    if (this.bleCharts.iSignal) {
                        this.bleCharts.iSignal.options.scales.y.min = newMinI;
                        this.bleCharts.iSignal.options.scales.y.max = newMaxI;
                        console.log(`📊 自适应Y轴: I通道范围调整为 ${newMinI.toFixed(3)}-${newMaxI.toFixed(3)}V (标准差:${stdI.toFixed(4)}V)`);
                    }

                    // 更新Q通道Y轴
                    if (this.bleCharts.qSignal) {
                        this.bleCharts.qSignal.options.scales.y.min = newMinQ;
                        this.bleCharts.qSignal.options.scales.y.max = newMaxQ;
                        console.log(`📊 自适应Y轴: Q通道范围调整为 ${newMinQ.toFixed(3)}-${newMaxQ.toFixed(3)}V (标准差:${stdQ.toFixed(4)}V)`);
                    }

                    this.adaptiveStabilized = true;
                    console.log('✅ Y轴自适应调节完成，开始显示细节');
                }
            }
        }

        // 🔍 调试：降低日志频率以提高性能
        if (this.bleDataCount <= 100 && this.bleDataCount % 100 === 0) { // 从50改为100
            console.log(`📊 Buffer统计 (总点数=${len}): I=${Math.min(...this.bleBufferI).toFixed(3)}-${Math.max(...this.bleBufferI).toFixed(3)}V`);
        }

        const sampleSize = Math.min(1000, len);
        const start = len - sampleSize;
        const indices = Array.from({length: sampleSize}, (_, i) => i);

        // 🔍 调试：验证传给图表的数据
        const iDataForChart = this.bleBufferI.slice(start);
        const qDataForChart = this.bleBufferQ.slice(start);

        // 减少调试日志以提高性能
        if (this.bleDataCount === 10) {
            console.log(`🎨 图表初始化完成 - 数据长度:${iDataForChart.length}`);
        }

        // 更新 I 通道
        if (this.bleCharts.iSignal) {
            this.bleCharts.iSignal.data = {
                labels: indices,
                datasets: [
                    { label: 'I通道', data: iDataForChart, borderColor: 'rgb(75, 192, 192)', backgroundColor: 'rgba(75, 192, 192, 0.2)', tension: 0.1, pointRadius: 0 }
                ]
            };
            this.bleCharts.iSignal.update('none');
        }

        // 更新 Q 通道
        if (this.bleCharts.qSignal) {
            this.bleCharts.qSignal.data = {
                labels: indices,
                datasets: [
                    { label: 'Q通道', data: qDataForChart, borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.2)', tension: 0.1, pointRadius: 0 }
                ]
            };
            this.bleCharts.qSignal.update('none');
        }

        const constellationSampleSize = Math.min(500, len);
        const step = Math.max(1, Math.floor(len / constellationSampleSize));
        const data = [];
        for (let i = start; i < len; i += step) data.push({ x: this.bleBufferI[i], y: this.bleBufferQ[i] });
        // 更新星座图
        if (this.bleCharts.constellation) {
            this.bleCharts.constellation.data = { datasets: [ { label: 'I/Q数据点', data, backgroundColor: 'rgba(54, 162, 235, 0.6)', pointRadius: 2 } ] };
            this.bleCharts.constellation.update();
            if (this.bleDataCount === 10) {
                console.log('✅ 星座图已更新');
            }
        } else {
            console.warn('❌ 星座图对象不存在');
        }

        // 更新 IMU 图表（gx/gy/gz）
        if (this.bleCharts.imu && this.bleBufferIMU_X.length > 0) {
            if (this.bleDataCount === 10) {
                console.log(`🎯 IMU更新条件满足: 图表存在=${!!this.bleCharts.imu}, IMU_X长度=${this.bleBufferIMU_X.length}`);
            }
            this.bleCharts.imu.data = {
                labels: indices,
                datasets: [
                    { label: 'gx', data: this.bleBufferIMU_X.slice(start), borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.08)', tension: 0.1, pointRadius: 0 },
                    { label: 'gy', data: this.bleBufferIMU_Y.slice(start), borderColor: 'rgb(54, 162, 235)', backgroundColor: 'rgba(54, 162, 235, 0.08)', tension: 0.1, pointRadius: 0 },
                    { label: 'gz', data: this.bleBufferIMU_Z.slice(start), borderColor: 'rgb(75, 192, 192)', backgroundColor: 'rgba(75, 192, 192, 0.08)', tension: 0.1, pointRadius: 0 }
                ]
            };
            this.bleCharts.imu.update();
            if (this.bleDataCount === 10) {
                console.log('✅ IMU图表已更新');
            }
        } else if (this.bleBufferIMU_X.length > 0) {
            console.warn('❌ IMU图表对象不存在，但有IMU数据');
        }

        // 更新温度图表
        if (this.bleCharts.temperature && this.bleBufferTemperature.length > 0) {
            const tempDataRaw = this.bleBufferTemperature.slice(start);
            // 过滤掉null值，只显示有效温度数据
            const validTempData = tempDataRaw.map((temp, idx) => temp !== null ? temp : null);

            // 计算有效温度数据的统计
            const validTemps = validTempData.filter(temp => temp !== null);
            const hasValidTemp = validTemps.length > 0;

            this.bleCharts.temperature.data = {
                labels: indices,
                datasets: [
                    {
                        label: hasValidTemp ? `温度 (°C) - 最新: ${validTemps[validTemps.length - 1]?.toFixed(1)}°C` : '温度 (°C) - 无数据',
                        data: validTempData,
                        borderColor: hasValidTemp ? 'rgb(255, 159, 64)' : 'rgb(200, 200, 200)',
                        backgroundColor: hasValidTemp ? 'rgba(255, 159, 64, 0.2)' : 'rgba(200, 200, 200, 0.1)',
                        tension: 0.3,
                        pointRadius: 0,
                        fill: true,
                        spanGaps: false // 不连接null值之间的间隙
                    }
                ]
            };
            this.bleCharts.temperature.update();
            if (this.bleDataCount === 10) {
                console.log(`✅ 温度图表已更新 - 有效温度点: ${validTemps.length}/${tempDataRaw.length}`);
            }
        } else if (this.bleBufferTemperature.length > 0) {
            console.warn('❌ 温度图表对象不存在，但有温度数据');
        }

        // 更新当前温度显示
        if (tempData && tempData.length > 0) {
            const currentTemp = tempData[tempData.length - 1];
            const tempEl = document.getElementById('bleCurrentTemp');
            const avgTempEl = document.getElementById('bleAvgTemp');
            if (tempEl) {
                tempEl.textContent = `${currentTemp.toFixed(1)} °C`;
            }
            if (avgTempEl) {
                avgTempEl.textContent = `${currentTemp.toFixed(1)} °C`;
            }
        }
    }

    /**
     * 重置自适应Y轴状态（手动重置为初始范围）
     */
    resetAdaptiveYAxis() {
        console.log('🔄 重置自适应Y轴状态...');

        // 重置状态变量
        this.adaptiveSampleCount = 0;
        this.adaptiveLastMinI = Infinity;
        this.adaptiveLastMaxI = -Infinity;
        this.adaptiveLastMinQ = Infinity;
        this.adaptiveLastMaxQ = -Infinity;
        this.adaptiveStabilized = false;

        // 重置图表Y轴到初始范围
        if (this.bleCharts.iSignal) {
            this.bleCharts.iSignal.options.scales.y.min = 0;
            this.bleCharts.iSignal.options.scales.y.max = 4.0;
            this.bleCharts.iSignal.update();
        }
        if (this.bleCharts.qSignal) {
            this.bleCharts.qSignal.options.scales.y.min = 0;
            this.bleCharts.qSignal.options.scales.y.max = 4.0;
            this.bleCharts.qSignal.update();
        }

        console.log('✅ 自适应Y轴已重置为初始范围 (0-4.0V)');
    }

    /**
     * 强制切换到细节显示模式（极紧凑的Y轴范围）
     */
    forceDetailMode() {
        if (this.bleBufferI.length < 50) {
            console.warn('❌ 数据点不足，无法切换到细节模式');
            return;
        }

        console.log('🔍 强制切换到细节显示模式...');

        // 使用最近50个数据点计算极紧凑的范围
        const detailDataSize = Math.min(this.bleBufferI.length, 50);
        const startIdx = this.bleBufferI.length - detailDataSize;
        const detailI = this.bleBufferI.slice(startIdx);
        const detailQ = this.bleBufferQ.slice(startIdx);

        const minI = Math.min(...detailI);
        const maxI = Math.max(...detailI);
        const minQ = Math.min(...detailQ);
        const maxQ = Math.max(...detailQ);

        const rangeI = maxI - minI;
        const rangeQ = maxQ - minQ;

        // 设置极小的余量：0.02V或数据范围的2%
        const detailPadding = 0.02;
        const rangePaddingI = Math.max(detailPadding, rangeI * 0.02);
        const rangePaddingQ = Math.max(detailPadding, rangeQ * 0.02);

        const detailMinI = Math.max(0, minI - rangePaddingI);
        const detailMaxI = maxI + rangePaddingI;
        const detailMinQ = Math.max(0, minQ - rangePaddingQ);
        const detailMaxQ = maxQ + rangePaddingQ;

        // 更新图表
        if (this.bleCharts.iSignal) {
            this.bleCharts.iSignal.options.scales.y.min = detailMinI;
            this.bleCharts.iSignal.options.scales.y.max = detailMaxI;
            this.bleCharts.iSignal.update();
        }
        if (this.bleCharts.qSignal) {
            this.bleCharts.qSignal.options.scales.y.min = detailMinQ;
            this.bleCharts.qSignal.options.scales.y.max = detailMaxQ;
            this.bleCharts.qSignal.update();
        }

        // 重置自适应状态，防止自动调节覆盖手动设置
        this.adaptiveStabilized = false;

        console.log(`🎯 细节模式已激活: I(${detailMinI.toFixed(4)}-${detailMaxI.toFixed(4)}V), Q(${detailMinQ.toFixed(4)}-${detailMaxQ.toFixed(4)}V)`);
    }

    /**
     * 强制重新初始化所有图表（用于调试蓝牙图表显示问题）
     */
    forceReinitializeCharts() {
        console.log('🔄 强制重新初始化所有图表...');
        console.log('当前图表状态:', {
            iSignal: !!this.charts.iSignal,
            qSignal: !!this.charts.qSignal,
            bleISignal: !!this.bleCharts.iSignal,
            bleQSignal: !!this.bleCharts.qSignal,
            bleConstellation: !!this.bleCharts.constellation,
            bleIMU: !!this.bleCharts.imu,
            bleTemperature: !!this.bleCharts.temperature
        });

        this.initializeCharts();
        this.initializeBluetoothCharts();

        // 延迟刷新所有图表
        setTimeout(() => {
            const allCharts = [
                ...Object.values(this.charts || {}),
                ...Object.values(this.bleCharts || {})
            ];
            allCharts.forEach(chart => {
                if (chart && typeof chart.resize === 'function') chart.resize();
                if (chart && typeof chart.update === 'function') chart.update();
            });
            console.log('✅ 图表重新初始化完成');
        }, 100);
    }

    /**
     * 更新蓝牙生理参数（参考main.py的心率稳定算法）
     */
    updateBluetoothVitalSigns() {
        // 增加窗口长度以提高稳定性（参考main.py使用500-1000点）
        const fs = (this.processor && Number.isFinite(this.processor.fs)) ? this.processor.fs : 50;
        const windowSize = Math.min(this.bleBufferI.length, fs * 30); // 最近30秒（50Hz=>1500点）
        const iData = new Float64Array(this.bleBufferI.slice(-windowSize));
        const qData = new Float64Array(this.bleBufferQ.slice(-windowSize));
        
        // 需要至少5秒数据才能计算
        if (iData.length < fs * 5) {
            return;
        }

        try {
            // 运行时防御：确认方法已加载
            if (!this.processor || typeof this.processor.extractVitalSignsMainPy !== 'function') {
                console.warn('extractVitalSignsMainPy 未加载，回退到旧算法');
                const { center, radius } = this.processor.circleFitting(iData, qData);
                const phaseData = this.processor.arcsinDemodulation(iData, qData, center, radius);
                const vital = this.processor.extractVitalSigns(iData, qData, phaseData);

                // 最少更新显示，避免空白
                const hrElement = document.getElementById('bleCurrentHR');
                const respElement = document.getElementById('bleCurrentResp');
                if (hrElement) hrElement.textContent = `${vital.heartRate} bpm`;
                if (respElement) respElement.textContent = `${vital.respiratoryRate} bpm`;
                return;
            }
            // 完全对齐 main.py：单函数完成相位、波形、HR/RR 提取
            const result = this.processor.extractVitalSignsMainPy(iData, qData);
            const { heartRate, respiratoryRate, phase, respiratoryWave, heartbeatWave } = result;
            // 保存窗口统计：仅在“开始记录”时保存，避免不录制时内存持续增长
            if (this.bleRecordingFlag === 1) {
                if (!this._bleWindowHistory) this._bleWindowHistory = [];
                this._bleWindowHistory.push({ t: Date.now(), heartRate, respiratoryRate });
                if (this._bleWindowHistory.length > 600) this._bleWindowHistory.splice(0, this._bleWindowHistory.length - 600); // 最多保留约10分钟(1Hz)
            }

            // 更新呼吸/心跳波形图表
            const sampleSize = Math.min(1000, iData.length);
            const indices = Array.from({length: sampleSize}, (_, i) => i);

            if (this.bleCharts.respiratory) {
                this.bleCharts.respiratory.data = { labels: indices, datasets: [{ label: '呼吸波形(实时)', data: Array.from(respiratoryWave.slice(-sampleSize)), borderColor: 'rgb(75, 192, 192)', backgroundColor: 'rgba(75, 192, 192, 0.2)', tension: 0.1 }] };
                this.bleCharts.respiratory.update();  // 移除 'none' 让图表真正刷新
            }

            if (this.bleCharts.heartbeat) {
                this.bleCharts.heartbeat.data = { labels: indices, datasets: [{ label: '心跳波形(实时)', data: Array.from(heartbeatWave.slice(-sampleSize)), borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.2)', tension: 0.1 }] };
                this.bleCharts.heartbeat.update();  // 移除 'none' 让图表真正刷新
            }

            // 推动ECG动态画布数据
            if (this._bleECG) {
                const resTrack = this._bleECG.res;
                const hbTrack = this._bleECG.hb;
                const pushLen = Math.min(50, respiratoryWave.length);
                const startIdx = Math.max(0, respiratoryWave.length - pushLen);
                // 归一化尾段，避免幅值漂移导致看不见
                const resSeg = Array.from(respiratoryWave.slice(startIdx));
                const hbSeg = Array.from(heartbeatWave.slice(Math.max(0, heartbeatWave.length - pushLen)));
                const norm = (arr) => {
                    if (arr.length === 0) return arr;
                    const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
                    const std = Math.sqrt(arr.reduce((s,v)=>s+(v-mean)*(v-mean),0)/arr.length) || 1;
                    return arr.map(v => (v-mean)/(std*3)); // 压缩到[-~0.3,0.3]范围，便于显示
                };
                const resNorm = norm(resSeg);
                const hbNorm = norm(hbSeg);
                resNorm.forEach(v => resTrack.data.push(v));
                hbNorm.forEach(v => hbTrack.data.push(v));
                // 裁剪，避免无限增长
                if (resTrack.data.length > 5000) resTrack.data.splice(0, resTrack.data.length - 5000);
                if (hbTrack.data.length > 5000) hbTrack.data.splice(0, hbTrack.data.length - 5000);
                // 始终刷新一次画布，即使不在播放状态
                if (this._bleECG.draw) {
                    this._bleECG.draw();
                }
                // 如果在播放状态，继续动画循环
                if ((resTrack.playing || hbTrack.playing) && !this._bleECG.raf) {
                    this._bleECG.raf = requestAnimationFrame(this._bleECG.draw);
                }
            }

            // ===== 心率平滑处理（参考main.py第332-340行）=====
            
            // 1. 更新循环历史记录（类似Python端的固定长度数组）
            this.heartRateHistory[this.historyIndex] = heartRate;
            this.respiratoryHistory[this.historyIndex] = respiratoryRate;
            this.historyIndex = (this.historyIndex + 1) % this.historyMaxLength;

            // 2. 计算移动平均（参考main.py第333行的np.mean(heart_history_short)）
            const avgHeartRate = Math.round(
                this.heartRateHistory.reduce((a, b) => a + b, 0) / this.historyMaxLength
            );
            const avgRespRate = Math.round(
                this.respiratoryHistory.reduce((a, b) => a + b, 0) / this.historyMaxLength
            );
            
            // 4. 心率稳定控制（参考main.py第353-360行的逻辑）
            let displayHeartRate = avgHeartRate;
            let displayRespRate = avgRespRate;
            
            // 始终应用心率变化限制（数组已填满历史数据）
            const delta = avgHeartRate - this.lastStableHeartRate;
            if (Math.abs(delta) > this.heartRateDelta) {
                // 限制变化：只允许每次改变heartRateDelta的幅度
                displayHeartRate = this.lastStableHeartRate + Math.sign(delta) * this.heartRateDelta;
                console.log(`心率限制: ${avgHeartRate} → ${displayHeartRate} (变化${delta}bpm超过阈值${this.heartRateDelta}bpm)`);
            }
            
            // 5. 更新稳定值
            this.lastStableHeartRate = displayHeartRate;
            this.lastStableRespRate = displayRespRate;
            
            // 6. 使用平滑后的值
            const vital = { 
                heartRate: displayHeartRate, 
                respiratoryRate: displayRespRate 
            };
            
            // 更新当前心率和呼吸率（供静息监测模块使用）
            this.currentHeartRate = displayHeartRate;
            this.currentRespiratoryRate = displayRespRate;
            
            console.log(`生理参数: 原始HR=${heartRate}bpm, 平滑后HR=${displayHeartRate}bpm, RR=${displayRespRate}bpm (历史${this.heartRateHistory.length}次)`);
            
            // 更新显示
            const hrElement = document.getElementById('bleCurrentHR');
            const respElement = document.getElementById('bleCurrentResp');
            const avgHrElement = document.getElementById('bleAvgHeartRate');
            const avgRespElement = document.getElementById('bleAvgRespRate');
            
            if (hrElement) hrElement.textContent = `${vital.heartRate} bpm`;
            if (respElement) respElement.textContent = `${vital.respiratoryRate} bpm`;
            if (avgHrElement) avgHrElement.textContent = `${vital.heartRate} bpm`;
            if (avgRespElement) avgRespElement.textContent = `${vital.respiratoryRate} bpm`;
            // 同步更新蓝牙ECG区块显示数值
            const bleHrEl = document.getElementById('bleCurrentHeartRate');
            const bleRespEl = document.getElementById('bleCurrentRespRate');
            if (bleHrEl) bleHrEl.textContent = `${vital.heartRate} bpm`;
            if (bleRespEl) bleRespEl.textContent = `${vital.respiratoryRate} bpm`;

            // 同时更新动态心电图的显示
            if (document.getElementById('currentHeartRate')) {
                document.getElementById('currentHeartRate').textContent = `${vital.heartRate} bpm`;
            }
            if (document.getElementById('currentRespRate')) {
                document.getElementById('currentRespRate').textContent = `${vital.respiratoryRate} bpm`;
            }
            
            // 避免日志刷屏导致卡顿：最多每10秒记一次（且仅在录制时）
            const now = Date.now();
            if (this.bleRecordingFlag === 1 && now - (this._bleVitalLogLastTs || 0) > 10000) {
                this._bleVitalLogLastTs = now;
                this.addBLELog(`📊 生理参数: HR=${vital.heartRate}bpm, RR=${vital.respiratoryRate}bpm`);
            }
            
        } catch (e) {
            console.error('更新生理参数错误:', e);
            this.addBLELog(`❌ 处理错误: ${e.message}`);
        }
    }
}

// 全局函数供HTML调用
let app;

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    app = new RadarWebApp();
});

// 供HTML按钮调用的全局函数
function processFiles() {
    app.processFiles();
}

function clearFiles() {
    app.clearFiles();
}

function exportResults() {
    app.exportResults();
}

function exportCharts() {
    app.exportCharts();
}

function toggleSettings() {
    app.toggleSettings();
}

function applySettings() {
    if (app && typeof app.applySettings === 'function') app.applySettings();
}

// 连接诊断：生成诊断JSON并复制到剪贴板（方便你粘贴给我分析）
async function bleQuickDiagnose() {
    if (!app) return;
    const diag = app.buildBleDiagnostics ? app.buildBleDiagnostics() : { error: 'buildBleDiagnostics not available' };
    const text = JSON.stringify(diag, null, 2);
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
            app.addBLELog('🩺 连接诊断已复制到剪贴板，请直接粘贴给我。');
        } else {
            // fallback
            prompt('复制下面的诊断信息发给我：', text);
        }
    } catch (e) {
        prompt('复制下面的诊断信息发给我：', text);
    }
    // 同时也打印到控制台（便于开发者工具查看）
    console.log('[BLE_DIAG]', diag);
}

// ===== Azure 配置/Prompt/RAG UI 逻辑 =====
function showAIConfig() {
    const modal = document.getElementById('aiConfigModal');
    if (!modal) return;
    // 预填本地保存的配置
    document.getElementById('azureEndpoint').value = localStorage.getItem('azureEndpoint') || '';
    document.getElementById('azureApiKey').value = localStorage.getItem('azureApiKey') || '';
    document.getElementById('azureDeployment').value = localStorage.getItem('azureDeployment') || 'gpt-4';
    modal.style.display = 'block';
}

function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.style.display = 'none';
}

function saveAIConfig() {
    const endpoint = document.getElementById('azureEndpoint').value.trim();
    const apiKey = document.getElementById('azureApiKey').value.trim();
    const deployment = document.getElementById('azureDeployment').value.trim();
    if (!endpoint || !apiKey || !deployment) {
        alert('请完整填写 Endpoint / API Key / Deployment');
        return;
    }
    localStorage.setItem('azureEndpoint', endpoint);
    localStorage.setItem('azureApiKey', apiKey);
    localStorage.setItem('azureDeployment', deployment);
    alert('已保存 Azure OpenAI 配置');
    closeModal('aiConfigModal');
}

async function testAIConnection() {
    try {
        const endpoint = document.getElementById('azureEndpoint').value.trim();
        const apiKey = document.getElementById('azureApiKey').value.trim();
        const deployment = document.getElementById('azureDeployment').value.trim();
        if (!endpoint || !apiKey || !deployment) {
            alert('请先填写所有配置');
            return;
        }
        const analyzer = new AzureGPTAnalyzer();
        analyzer.configure(endpoint, apiKey, deployment);
        // 用极小提示测试
        const response = await analyzer.callAzureOpenAI('Test connection');
        alert('连接成功');
    } catch (e) {
        alert('连接失败: ' + e.message);
    }
}

function showPromptEditor() {
    const modal = document.getElementById('promptEditorModal');
    if (modal) modal.style.display = 'block';
}

function showRAGEditor() {
    const modal = document.getElementById('ragEditorModal');
    if (modal) modal.style.display = 'block';
}

// BLE 控制按钮回调
async function bleConnect() {
    if (!window.BLE) {
        app.showMessage('此浏览器不支持Web Bluetooth', 'error');
        return;
    }
    try {
        await BLE.connect();
    } catch (e) {
        app.showMessage(`连接失败: ${e.message}`, 'error');
    }
}

async function bleDisconnect() {
    if (!window.BLE) return;
    try {
        await BLE.disconnect();
    } catch (e) {
        // ignore
    }
}

// 蓝牙录制控制函数 (参考main.py的按钮响应)
function toggleBluetoothRecording() {
    if (app && app.bleConnected) {
        app.toggleBluetoothRecording();
    } else {
        alert('请先连接蓝牙设备');
    }
}

// 分离的开始/结束录制按钮事件
function bleStartRecording() {
    if (!app || !app.bleConnected) {
        alert('请先连接蓝牙设备');
        return;
    }
    if (app.bleRecordingFlag !== 1) {
        app.toggleBluetoothRecording();
    }
}

function bleStopRecording() {
    if (!app || !app.bleConnected) {
        alert('请先连接蓝牙设备');
        return;
    }
    if (app.bleRecordingFlag === 1) {
        app.toggleBluetoothRecording();
    }
}

// 蓝牙上报控制
function bleStartUpload() {
    if (!app) return;
    app.startBleUpload();
}

function bleStopUpload() {
    if (!app) return;
    app.stopBleUpload();
}

// 蓝牙图表控制函数
function showBluetoothCharts() {
    const section = document.getElementById('bluetoothChartsSection');
    if (section) {
        section.style.display = 'block';
        section.scrollIntoView({ behavior: 'smooth' });
    }
    // 展开后强制刷新图表尺寸（避免之前隐藏导致的空白/不刷新）
    if (window.app && app.bleCharts) {
        setTimeout(() => {
            try {
                Object.values(app.bleCharts).forEach(ch => {
                    if (ch && typeof ch.resize === 'function') ch.resize();
                    if (ch && typeof ch.update === 'function') ch.update('none');
                });
            } catch (_) {}
        }, 50);
    }
}

function hideBluetoothCharts() {
    const section = document.getElementById('bluetoothChartsSection');
    if (section) {
        section.style.display = 'none';
    }
}

// 文件数据ECG播放控制
function toggleECGPlayback() {
    if (!app) return;

    // 初始化ECG播放器（如果还没有初始化）
    if (!app._fileECG) {
        app.initializeFileECG();
    }

    if (!app._fileECG) return;

    const playing = app._fileECG.res.playing || app._fileECG.hb.playing;
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');

    if (playing) {
        // 暂停播放
        app._fileECG.res.playing = false;
        app._fileECG.hb.playing = false;
        pauseBtn.style.display = 'none';
        playBtn.style.display = 'inline-block';
    } else {
        // 开始播放
        app._fileECG.res.playing = true;
        app._fileECG.hb.playing = true;
        playBtn.style.display = 'none';
        pauseBtn.style.display = 'inline-block';
        if (!app._fileECG.raf) app._fileECG.draw();
    }
}

// BLE ECG 控制
function toggleBLEECGPlayback() {
    if (!app || !app._bleECG) return;
    const playing = app._bleECG.res.playing || app._bleECG.hb.playing;
    const playBtn = document.getElementById('blePlayBtn');
    const pauseBtn = document.getElementById('blePauseBtn');
    if (playing) {
        app._bleECG.res.playing = false;
        app._bleECG.hb.playing = false;
        pauseBtn.style.display = 'none';
        playBtn.style.display = 'inline-block';
    } else {
        app._bleECG.res.playing = true;
        app._bleECG.hb.playing = true;
        playBtn.style.display = 'none';
        pauseBtn.style.display = 'inline-block';
        if (!app._bleECG.raf) app._bleECG.draw();
    }
}

function resetECG() {
    if (!app || !app._fileECG) return;
    if (app._fileECG) {
        app._fileECG.res.cursor = 0;
        app._fileECG.hb.cursor = 0;
        app._fileECG.res.playing = false;
        app._fileECG.hb.playing = false;

        const playBtn = document.getElementById('playBtn');
        const pauseBtn = document.getElementById('pauseBtn');
        if (playBtn && pauseBtn) {
            pauseBtn.style.display = 'none';
            playBtn.style.display = 'inline-block';
        }
    }
}

function testECG() {
    if (!app) return;

    // 确保有处理结果
    if (app.processedResults.length === 0) {
        app.showMessage('请先上传并处理数据文件', 'warning');
        return;
    }

    // 初始化并测试ECG播放
    app.initializeFileECG();
    if (app._fileECG) {
        // 自动开始播放
        toggleECGPlayback();
        app.showMessage('ECG测试播放已启动', 'success');
    } else {
        app.showMessage('没有可播放的ECG数据', 'warning');
    }
}

function resetBLEECG() {
    if (!app || !app._bleECG) return;
    app._bleECG.res.data = [];
    app._bleECG.hb.data = [];
}

function clearBluetoothData() {
    if (app && confirm('确定要清空蓝牙数据吗？这将重置所有实时数据。')) {
        app.resetBluetoothData();
        app.addBLELog('🔄 已清空蓝牙数据');
    }
}

function saveBluetoothData() {
    if (!app || !app.bleConnected) {
        alert('请先连接蓝牙设备');
        return;
    }
    
    if (app.bleBufferI.length === 0) {
        alert('没有可保存的数据');
        return;
    }

    // 生成文件内容
    let content = '';
    for (let i = 0; i < app.bleBufferI.length; i++) {
        const ts = app.bleBufferTimestamps[i] || `${Date.now()}-${i}`;
        content += `${ts}\t${app.bleBufferI[i]}\t${app.bleBufferQ[i]}\n`;
    }
    
    // 下载文件
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `bluetooth_data_${timestamp}.txt`;
    app.downloadFile(content, filename, 'text/plain');
    app.addBLELog(`💾 已保存数据: ${filename} (${app.bleBufferI.length} 数据点)`);
}

// 模拟测试功能
function startSimulationTest() {
    if (!app) return;
    
    app.addBLELog('🧪 开始模拟数据测试...');
    
    // 模拟蓝牙连接
    app.bleConnected = true;
    app.bleConnectStartTime = Date.now();
    app.startBluetoothTimer();
    
    // 显示实时数据区域
    document.getElementById('bleRealTimeData').style.display = 'block';

    // 自动展开图表区域并刷新（确保图表可见且尺寸正确）
    const chartsSection = document.getElementById('bluetoothChartsSection');
    if (chartsSection) {
        chartsSection.style.display = 'block';
        chartsSection.scrollIntoView({ behavior: 'smooth' });
        console.log('✅ 模拟测试：蓝牙图表区域已展开');
    }

    // 确保图表已初始化
    if (!app.bleCharts.iSignal || !app.bleCharts.qSignal) {
        console.log('🔄 模拟测试：重新初始化蓝牙图表...');
        app.initializeBluetoothCharts();
    }

    // 延迟触发布局更新，确保Canvas尺寸正确
    setTimeout(() => {
        if (app.bleCharts) {
            console.log('📊 模拟测试：刷新所有蓝牙图表...');
            Object.values(app.bleCharts).forEach(chart => {
                if (chart && typeof chart.resize === 'function') chart.resize();
                if (chart && typeof chart.update === 'function') chart.update('none');
            });
        }
    }, 200);

    app.updateBLEButtons();
    
    // 生成模拟数据 (模拟心率75bpm，呼吸18bpm)
    let dataCount = 0;
    app.stopSimulation();
    app._simInterval = setInterval(() => {
        if (dataCount >= 2000) {
            app.stopSimulation();
            app.addBLELog('🏁 模拟测试完成');
            return;
        }
        
        const fs = (app.processor && Number.isFinite(app.processor.fs)) ? app.processor.fs : 50;
        const t = dataCount / fs;
        // 模拟信号: 呼吸(0.3Hz=18bpm) + 心率(1.25Hz=75bpm) + 噪声
        const respiratorySignal = 0.5 * Math.sin(2 * Math.PI * 0.3 * t);
        const heartSignal = 0.2 * Math.sin(2 * Math.PI * 1.25 * t);
        const noise = 0.1 * (Math.random() - 0.5);
        
        // 模拟I/Q电压数据（0~3.3V范围）
        const voltageI = 1.65 + respiratorySignal + heartSignal + noise;
        const voltageQ = 1.55 + respiratorySignal * 0.8 + heartSignal * 1.2 + noise * 0.8;
        
        // 将电压转换为ADC值（-32768~32767）
        // 反向公式：adc = (voltage * 2 / 3.3 - 1) * 32767
        const adcI = Math.round((voltageI * 2 / 3.3 - 1) * 32767);
        const adcQ = Math.round((voltageQ * 2 / 3.3 - 1) * 32767);
        
        // 模拟蓝牙数据接收（接近实际设备格式：包含 Gyr 三轴和温度）
        const gx = 10 * Math.sin(2 * Math.PI * 0.5 * t);
        const gy = 5 * Math.cos(2 * Math.PI * 0.2 * t);
        const gz = 2 * Math.sin(2 * Math.PI * 1.0 * t);
        // 模拟温度缓慢变化（34-36°C之间波动）
        const temp = 35 + Math.sin(2 * Math.PI * 0.01 * t) + 0.1 * (Math.random() - 0.5);
        const simulatedLine = `ADC:${adcI} ${adcQ}|Gyr:${gx.toFixed(2)} ${gy.toFixed(2)} ${gz.toFixed(2)}|T:${temp.toFixed(1)}`;
        app.handleBLELine(simulatedLine);
        
        dataCount++;
    }, 20); // 50Hz采样率 = 20ms间隔

    app.addBLELog(`📡 正在生成模拟心率75bpm、呼吸18bpm的数据（${app.processor.fs}Hz采样率）...`);
}

// 停止模拟
RadarWebApp.prototype.stopSimulation = function() {
    if (this._simInterval) {
        clearInterval(this._simInterval);
        this._simInterval = null;
    }
};

// 触发Azure诊断：基于本次录制窗口统计
async function bleAzureDiagnose() {
    if (!window.AzureGPTAnalyzer) {
        alert('Azure模块未加载');
        return;
    }
    if (!app || !app._bleWindowHistory || app._bleWindowHistory.length === 0) {
        alert('暂无可用的录制窗口统计，请先完成一次录制');
        return;
    }

    try {
        const analyzer = new AzureGPTAnalyzer();
        // 读取页面配置（如果已在右侧设置面板中配置，则可扩展从localStorage读取）
        const endpoint = localStorage.getItem('azureEndpoint') || '';
        const apiKey = localStorage.getItem('azureApiKey') || '';
        const deployment = localStorage.getItem('azureDeployment') || 'gpt-4';
        analyzer.configure(endpoint, apiKey, deployment);

        // 将本次录制窗口统计转换为 processedResults 结构的最小集合
        const session = app._buildBluetoothSessionStats();
        const processedResults = [
            {
                status: 'success',
                heartRate: session.average.heartRate,
                respiratoryRate: session.average.respiratoryRate,
                heartRateTimeSeries: session.windows.map(w => w.heartRate),
                respiratoryRateTimeSeries: session.windows.map(w => w.respiratoryRate),
                timeAxis: session.windows.map((w, i) => i),
                dataPoints: app.bleBufferI.length,
                fileName: 'bluetooth_session'
            }
        ];

        const result = await analyzer.generateDiagnosticReport(processedResults, 'detailed_medical');
        if (result.success) {
            const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0,19);
            app.downloadFile(result.report, `bluetooth_session_report_${ts}.txt`, 'text/plain');
            app.addBLELog('🤖 已生成并下载AI诊断报告');
        } else {
            alert('生成诊断失败: ' + result.error);
        }
    } catch (e) {
        alert('AI诊断出错: ' + e.message);
    }
}

// 宠物健康分析相关全局函数
function performHealthAnalysis() {
    app.performHealthAnalysis();
}

function exportHealthReport() {
    app.exportHealthReport();
}

// 宠物健康对话相关全局函数
function initializeHealthChat() {
    app.initializeHealthChat();
}

function sendChatMessage() {
    app.sendChatMessage();
}

function clearChatHistory() {
    app.clearChatHistory();
}
