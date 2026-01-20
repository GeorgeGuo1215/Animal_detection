/**
 * 毫米波雷达数据处理Web应用主控制器
 */

class RadarWebApp {
    constructor() {
        // 采样率以右侧面板为准；修改为 50Hz
        const srEl = document.getElementById('samplingRate');
        const sr = srEl ? parseInt(srEl.value, 10) : NaN;
        const samplingRate = Number.isFinite(sr) && sr > 0 ? sr : 50;
        this.processor = new RadarDataProcessor(samplingRate);
        this.selectedFiles = [];
        this.processedResults = [];
        this.charts = {}; // 文件数据图表
        
        // 蓝牙数据相关
        this.bleConnected = false;
        this.bleCharts = {}; // 蓝牙数据图表
        this.bleBufferI = [];
        this.bleBufferQ = [];
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
        this.heartRateHistory = [];  // 心率历史记录
        this.respiratoryHistory = []; // 呼吸频率历史记录
        this.historyMaxLength = 30;  // 保留最近30次的结果（约30秒）
        this.heartRateDelta = 10;    // 心率最大变化幅度（bpm）参考main.py第51行
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
        this.bleRecordingData = []; // 记录的数据缓存
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

        // 初始化BLE事件
        this.initializeBLE();
        
        // 测试FFT是否正常工作
        this.testFFT();

        // 启动接收看门狗：若长时间无数据则判定断连
        this.startRxWatchdog();
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
            if (chartsSection) chartsSection.style.display = 'block';
            // 触发一次 resize/update，解决 display:none 时 Chart.js 尺寸为0的问题
            setTimeout(() => {
                try {
                    Object.values(this.bleCharts || {}).forEach(ch => {
                        if (ch && typeof ch.resize === 'function') ch.resize();
                        if (ch && typeof ch.update === 'function') ch.update('none');
                    });
                } catch (_) {}
            }, 50);

            // 自动初始化并启动蓝牙ECG播放
            this.initializeBLEECG();
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
        // 打印原始数据
        this.printRawData(line);
        this.lastBleRxTs = Date.now();
        // 允许 JSON 格式 {ts:..., i:..., q:...}；也兼容无空格双小数如 "1.6421.588"
        let ts, iVal, qVal;
        let imuX = 0, imuY = 0, imuZ = 0; // gx/gy/gz（优先取 Gyr:）
        let temperature = null; // 温度数据
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
        
        // 温度数据，保持同步（如果没有温度数据，使用上一次的值或默认25°C）
        const lastTemp = this.bleBufferTemperature.length > 0 ? this.bleBufferTemperature[this.bleBufferTemperature.length - 1] : 25;
        this.bleBufferTemperature.push(Number.isFinite(temperature) ? temperature : lastTemp);
        
        // 实时保存数据 (参考main.py的记录逻辑)
        if (this.bleRecordingFlag === 1) {
            const timestamp = new Date().toISOString().replace('T', ' ').slice(0, 19);
            const dataLine = `${timestamp}  ${iVal}  ${qVal}`;
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
        const txtFiles = files.filter(file => file.name.toLowerCase().endsWith('.txt'));
        
        if (txtFiles.length === 0) {
            this.showMessage('请选择.txt格式的数据文件', 'warning');
            return;
        }

        this.selectedFiles = txtFiles;
        this.displayFileList();
        this.showMessage(`已选择 ${txtFiles.length} 个文件`, 'success');
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
                const result = this.processor.processSingleFile(file.name, fileContent);
                this.processedResults.push(result);
                
                if (result.status === 'success') {
                    this.addStatusLog(`✓ ${file.name} 处理成功 - 心率: ${result.heartRate} bpm, 呼吸: ${result.respiratoryRate} bpm`);
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
     * 显示处理结果
     */
    displayResults() {
        const successResults = this.processedResults.filter(r => r.status === 'success');
        
        if (successResults.length === 0) {
            this.showMessage('没有成功处理的文件', 'warning');
            return;
        }

        // 更新统计信息
        this.updateStatistics(successResults);
        
        // 更新图表
        this.updateCharts(successResults);
        
        // 更新结果表格
        this.updateResultsTable();
        
        // 显示结果区域
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('resultsSection').classList.add('fade-in');
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
        this.bleCharts.iq = new Chart(document.getElementById('bleIQChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: '蓝牙 I/Q 实时信号' } } }
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
        this.charts.iq = new Chart(document.getElementById('iqChart'), {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: { ...chartOptions, plugins: { ...chartOptions.plugins, title: { display: true, text: '原始I/Q信号' } } }
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
        
        // 更新I/Q信号图
        const sampleSize = Math.min(1000, firstResult.iData.length);
        const indices = Array.from({length: sampleSize}, (_, i) => i);
        
        this.charts.iq.data = {
            labels: indices,
            datasets: [
                {
                    label: 'I通道',
                    data: Array.from(firstResult.iData.slice(0, sampleSize)),
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'Q通道',
                    data: Array.from(firstResult.qData.slice(0, sampleSize)),
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                }
            ]
        };
        this.charts.iq.update();

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

        // 更新呼吸波形
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

        // 更新心跳波形
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
     * 更新结果表格
     */
    updateResultsTable() {
        const tbody = document.getElementById('resultsTableBody');
        tbody.innerHTML = '';

        this.processedResults.forEach(result => {
            const row = document.createElement('tr');
            
            if (result.status === 'success') {
                row.innerHTML = `
                    <td>${result.fileName}</td>
                    <td>${result.dataPoints.toLocaleString()}</td>
                    <td>${result.heartRate}</td>
                    <td>${result.respiratoryRate}</td>
                    <td>${result.circleCenter[0].toFixed(4)}</td>
                    <td>${result.circleCenter[1].toFixed(4)}</td>
                    <td>${result.circleRadius.toFixed(4)}</td>
                    <td><span class="status-success">成功</span></td>
                `;
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
        this.bleRecordingStartTime = null;
        this._bleWindowHistory = [];
        
        // 重置心率平滑历史
        this.heartRateHistory = [];
        this.respiratoryHistory = [];
        this.lastStableHeartRate = 70;
        this.lastStableRespRate = 18;

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
            this.bleRecordingStartTime = new Date();
            
            // 生成录制文件名 (参考main.py的命名规则)
            const timestamp = this.bleRecordingStartTime.toISOString()
                .slice(0, 16).replace('T', '-').replace(/:/g, '-');
            
            this.addBLELog(`🔴 开始录制数据 - ${timestamp}`);
            this.addBLELog('📝 实时保存到内存，结束时将下载文件');
            
        } else {
            // 结束录制并自动下载文件
            const recordingEndTime = new Date();
            const duration = ((recordingEndTime - this.bleRecordingStartTime) / 1000).toFixed(1);
            
            // 生成文件内容 (参考main.py的数据格式)
            let fileContent = '';
            for (const line of this.bleRecordingData) {
                fileContent += line + '\n';
            }
            
            // 生成文件名 (参考main.py的命名格式)
            const timestamp = this.bleRecordingStartTime.toISOString()
                .slice(0, 16).replace('T', '-').replace(/:/g, '-');
            const filename = `bluetooth_record_${timestamp}.txt`;
            
            // 自动下载文件
            this.downloadFile(fileContent, filename, 'text/plain');
            
            // 汇总本次录制的窗口HR/RR用于Azure（按更新周期采样记录）
            const sessionStats = this._buildBluetoothSessionStats();
            const statsJson = JSON.stringify(sessionStats);
            const statsFilename = `bluetooth_record_${timestamp}_stats.json`;
            this.downloadFile(statsJson, statsFilename, 'application/json');
            this.addBLELog(`📈 已保存本次窗口统计: ${statsFilename}`);

            // 显示录制统计
            this.addBLELog(`🟢 录制结束 - 时长: ${duration}秒`);
            this.addBLELog(`💾 已保存: ${filename} (${this.bleRecordingData.length} 数据点)`);
            
            // 清空录制缓存
            this.bleRecordingData = [];
            this.bleRecordingStartTime = null;
        }
        
        // 更新按钮状态
        this.updateBLEButtons();
    }

    /**
     * 更新蓝牙实时图表
     */
    updateBluetoothLiveCharts() {
        if (!this.bleCharts.iq || !this.bleCharts.constellation) return;
        const len = this.bleBufferI.length;
        if (len < 10) return;

        // 🔍 调试：打印buffer统计
        if (this.bleDataCount <= 100 && this.bleDataCount % 50 === 0) {
            console.log(`\n📊 Buffer统计 (总点数=${len}):`);
            console.log(`  I通道: 最小=${Math.min(...this.bleBufferI).toFixed(4)}, 最大=${Math.max(...this.bleBufferI).toFixed(4)}, 平均=${(this.bleBufferI.reduce((a,b)=>a+b,0)/len).toFixed(4)}`);
            console.log(`  Q通道: 最小=${Math.min(...this.bleBufferQ).toFixed(4)}, 最大=${Math.max(...this.bleBufferQ).toFixed(4)}, 平均=${(this.bleBufferQ.reduce((a,b)=>a+b,0)/len).toFixed(4)}`);
            console.log(`  最后5个I值:`, this.bleBufferI.slice(-5).map(v => v.toFixed(4)));
            console.log(`  最后5个Q值:`, this.bleBufferQ.slice(-5).map(v => v.toFixed(4)));
            if (this.bleBufferTemperature.length > 0) {
                console.log(`  温度: 最小=${Math.min(...this.bleBufferTemperature).toFixed(1)}°C, 最大=${Math.max(...this.bleBufferTemperature).toFixed(1)}°C, 当前=${this.bleBufferTemperature[this.bleBufferTemperature.length - 1].toFixed(1)}°C`);
            }
        }

        const sampleSize = Math.min(1000, len);
        const start = len - sampleSize;
        const indices = Array.from({length: sampleSize}, (_, i) => i);

        // 🔍 调试：验证传给图表的数据
        const iDataForChart = this.bleBufferI.slice(start);
        const qDataForChart = this.bleBufferQ.slice(start);
        
        if (this.bleDataCount === 10) {
            console.log(`\n🎨 图表数据检查 (首次更新):`);
            console.log(`  start=${start}, sampleSize=${sampleSize}`);
            console.log(`  I数据长度=${iDataForChart.length}, 前5个:`, iDataForChart.slice(0, 5).map(v => v?.toFixed(4)));
            console.log(`  Q数据长度=${qDataForChart.length}, 前5个:`, qDataForChart.slice(0, 5).map(v => v?.toFixed(4)));
            console.log(`  Q数据包含0的数量: ${qDataForChart.filter(v => v === 0).length}`);
        }

        this.bleCharts.iq.data = {
            labels: indices,
            datasets: [
                { label: 'I通道', data: iDataForChart, borderColor: 'rgb(75, 192, 192)', backgroundColor: 'rgba(75, 192, 192, 0.2)', tension: 0.1 },
                { label: 'Q通道', data: qDataForChart, borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.2)', tension: 0.1 }
            ]
        };
        this.bleCharts.iq.update();  // 移除 'none'，让图表真正刷新

        const constellationSampleSize = Math.min(500, len);
        const step = Math.max(1, Math.floor(len / constellationSampleSize));
        const data = [];
        for (let i = start; i < len; i += step) data.push({ x: this.bleBufferI[i], y: this.bleBufferQ[i] });
        this.bleCharts.constellation.data = { datasets: [ { label: 'I/Q数据点', data, backgroundColor: 'rgba(54, 162, 235, 0.6)', pointRadius: 2 } ] };
        this.bleCharts.constellation.update();  // 移除 'none'，让图表真正刷新

        // 更新 IMU 图表（gx/gy/gz）
        if (this.bleCharts.imu && this.bleBufferIMU_X.length > 0) {
            this.bleCharts.imu.data = {
                labels: indices,
                datasets: [
                    { label: 'gx', data: this.bleBufferIMU_X.slice(start), borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.08)', tension: 0.1, pointRadius: 0 },
                    { label: 'gy', data: this.bleBufferIMU_Y.slice(start), borderColor: 'rgb(54, 162, 235)', backgroundColor: 'rgba(54, 162, 235, 0.08)', tension: 0.1, pointRadius: 0 },
                    { label: 'gz', data: this.bleBufferIMU_Z.slice(start), borderColor: 'rgb(75, 192, 192)', backgroundColor: 'rgba(75, 192, 192, 0.08)', tension: 0.1, pointRadius: 0 }
                ]
            };
            this.bleCharts.imu.update();  // 移除 'none'，让图表真正刷新
        }

        // 更新温度图表
        if (this.bleCharts.temperature && this.bleBufferTemperature.length > 0) {
            const tempData = this.bleBufferTemperature.slice(start);
            this.bleCharts.temperature.data = {
                labels: indices,
                datasets: [
                    { 
                        label: '温度 (°C)', 
                        data: tempData, 
                        borderColor: 'rgb(255, 159, 64)', 
                        backgroundColor: 'rgba(255, 159, 64, 0.2)', 
                        tension: 0.3,
                        pointRadius: 0,
                        fill: true
                    }
                ]
            };
            this.bleCharts.temperature.update();  // 移除 'none'，让图表真正刷新
            
            // 更新当前温度显示
            if (tempData.length > 0) {
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
            
            // 1. 添加到历史记录
            this.heartRateHistory.push(heartRate);
            this.respiratoryHistory.push(respiratoryRate);
            
            // 2. 保持历史记录长度
            if (this.heartRateHistory.length > this.historyMaxLength) {
                this.heartRateHistory.shift();
                this.respiratoryHistory.shift();
            }
            
            // 3. 计算移动平均（参考main.py第333行的np.mean(heart_history_short)）
            const avgHeartRate = Math.round(
                this.heartRateHistory.reduce((a, b) => a + b, 0) / this.heartRateHistory.length
            );
            const avgRespRate = Math.round(
                this.respiratoryHistory.reduce((a, b) => a + b, 0) / this.respiratoryHistory.length
            );
            
            // 4. 心率稳定控制（参考main.py第353-360行的逻辑）
            let displayHeartRate = avgHeartRate;
            let displayRespRate = avgRespRate;
            
            // 如果心率变化过大，限制变化幅度
            if (this.heartRateHistory.length >= 5) {
                const delta = avgHeartRate - this.lastStableHeartRate;
                if (Math.abs(delta) > this.heartRateDelta) {
                    // 限制变化：只允许每次改变heartRateDelta的幅度
                    displayHeartRate = this.lastStableHeartRate + Math.sign(delta) * this.heartRateDelta;
                    console.log(`心率限制: ${avgHeartRate} → ${displayHeartRate} (变化${delta}bpm超过阈值${this.heartRateDelta}bpm)`);
                }
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
        
        const fs = 50;
        const t = dataCount / fs; // 采样率50Hz
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

    app.addBLELog('📡 正在生成模拟心率75bpm、呼吸18bpm的数据（50Hz采样率）...');
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
