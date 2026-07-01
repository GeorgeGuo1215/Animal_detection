/**
 * 静息心率呼吸率监测模块 - 独立模块
 * 通过 IMU 数据判断用户是否处于静息状态，只在静息时记录心率和呼吸率
 */

class RestingMonitor {
    constructor() {
        // 监测状态
        this.enabled = false;              // 是否启用监测
        this.isResting = false;            // 当前是否处于静息状态
        this.restingStartTime = null;      // 当前静息开始时间
        this.restingDuration = 0;          // 当前静息持续时间（秒）
        this.monitorStartTime = null;      // 监测开始时间

        // 数据存储
        this.restingRecords = [];          // 静息心率呼吸率记录

        // 实时数据追踪
        this.currentRestingHR = null;      // 当前静息心率
        this.currentRestingRR = null;      // 当前静息呼吸率
        this.lastRestingUpdateTime = null; // 最后一次静息数据更新时间

        // 每分钟数据
        this.minuteData = [];              // {timestamp, avgHR, avgRR, count}
        this.currentMinuteHR = [];         // 当前分钟内的心率数据
        this.currentMinuteRR = [];         // 当前分钟内的呼吸率数据
        this.lastMinuteTimestamp = null;   // 上一分钟的时间戳

        // 每小时数据
        this.hourlyData = Array(24).fill(null).map(() => ({
            avgHR: 0,
            avgRR: 0,
            count: 0,
            samples: []
        }));

        // IMU 判断参数（保留作为备用）
        this.imuThreshold = 2.0;           // IMU 稳定阈值
        this.imuWindowSize = 50;           // 判断窗口大小（数据点）
        this.stableDuration = 5;           // 需要稳定多少秒才算静息
        this.minRestingDuration = 10;      // 最小静息时长（秒）

        // 临时稳定计时
        this.stableStartTime = null;       // 开始稳定的时间

        // 图表对象
        this.charts = {
            realtimeHR: null,              // 实时静息心率图
            minuteHR: null,                // 每分钟静息心率图
            hourlyHR: null                 // 每小时静息心率图
        };

        // 绑定到全局 app
        this.app = null;
    }
    
    /**
     * 绑定到主应用
     */
    bindToApp(app) {
        this.app = app;
    }
    
    /**
     * 计算数组的标准差
     */
    calculateStd(arr) {
        if (!arr || arr.length === 0) return 0;
        const mean = arr.reduce((sum, val) => sum + val, 0) / arr.length;
        const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
        return Math.sqrt(variance);
    }
    
    /**
     * 检查 IMU 是否稳定
     */
    checkIMUStable() {
        if (!this.app || !this.app.bleBufferIMU_X || this.app.bleBufferIMU_X.length < this.imuWindowSize) {
            return false;
        }
        
        // 获取最近的 IMU 数据
        const recentGX = this.app.bleBufferIMU_X.slice(-this.imuWindowSize);
        const recentGY = this.app.bleBufferIMU_Y.slice(-this.imuWindowSize);
        const recentGZ = this.app.bleBufferIMU_Z.slice(-this.imuWindowSize);
        
        // 计算每个轴的标准差
        const stdGX = this.calculateStd(recentGX);
        const stdGY = this.calculateStd(recentGY);
        const stdGZ = this.calculateStd(recentGZ);
        
        // 总体运动幅度（三轴平方和的平方根）
        const totalMotion = Math.sqrt(stdGX * stdGX + stdGY * stdGY + stdGZ * stdGZ);
        
        // 判断是否稳定
        return totalMotion < this.imuThreshold;
    }

    /**
     * 更新每分钟数据
     */
    updateMinuteData(now) {
        const currentMinute = Math.floor(now / 60000); // 当前分钟的时间戳（分钟级别）

        if (this.lastMinuteTimestamp === null) {
            this.lastMinuteTimestamp = currentMinute;
        }

        // 检查是否进入新的一分钟
        if (currentMinute > this.lastMinuteTimestamp) {
            // 计算上一分钟的平均值
            if (this.currentMinuteHR.length > 0) {
                const avgHR = this.currentMinuteHR.reduce((sum, val) => sum + val, 0) / this.currentMinuteHR.length;
                const avgRR = this.currentMinuteRR.reduce((sum, val) => sum + val, 0) / this.currentMinuteRR.length;

                this.minuteData.push({
                    timestamp: this.lastMinuteTimestamp * 60000, // 转换回毫秒
                    avgHR: avgHR,
                    avgRR: avgRR,
                    count: this.currentMinuteHR.length
                });

                // 限制历史数据长度（保留最近60分钟）
                if (this.minuteData.length > 60) {
                    this.minuteData.shift();
                }
            }

            // 重置当前分钟数据
            this.currentMinuteHR = [];
            this.currentMinuteRR = [];
            this.lastMinuteTimestamp = currentMinute;
        }
    }

    /**
     * 更新每小时数据
     */
    updateHourlyData(now) {
        const hour = new Date(now).getHours();

        if (this.currentRestingHR && this.currentRestingRR) {
            this.hourlyData[hour].samples.push({
                hr: this.currentRestingHR,
                rr: this.currentRestingRR
            });

            // 重新计算平均值
            const samples = this.hourlyData[hour].samples;
            if (samples.length > 0) {
                this.hourlyData[hour].avgHR = samples.reduce((sum, s) => sum + s.hr, 0) / samples.length;
                this.hourlyData[hour].avgRR = samples.reduce((sum, s) => sum + s.rr, 0) / samples.length;
                this.hourlyData[hour].count = samples.length;
            }
        }
    }

    /**
     * 更新监测状态（使用活动监测的强度判断）
     */
    update() {
        if (!this.enabled || !this.app) return;

        const now = Date.now();

        // 检查是否有活动监测模块
        const hasActivityMonitor = this.app.activityMonitor && this.app.activityMonitorEnabled;

        // 判断是否处于静息状态
        let isCurrentlyResting = false;

        if (hasActivityMonitor) {
            // 使用活动监测的强度判断（优先）
            isCurrentlyResting = this.app.activityMonitor.currentIntensity === 'resting';
        } else {
            // 降级到IMU稳定性判断
            isCurrentlyResting = this.checkIMUStable();
        }

        if (isCurrentlyResting) {
            // 处于静息状态
            if (!this.isResting) {
                // 刚进入静息状态
                if (!this.stableStartTime) {
                    // 刚开始稳定
                    this.stableStartTime = now;
                } else {
                    // 检查是否持续稳定足够久
                    const stableDuration = (now - this.stableStartTime) / 1000;
                    if (stableDuration >= this.stableDuration) {
                        // 确认进入静息状态
                        this.isResting = true;
                        this.restingStartTime = now;
                        const source = hasActivityMonitor ? '活动监测' : 'IMU';
                        this.log(`🛌 进入静息状态 (${source}判定，已稳定${stableDuration.toFixed(1)}秒)`);
                    }
                }
            } else {
                // 已在静息状态，更新持续时间
                this.restingDuration = (now - this.restingStartTime) / 1000;

                // 记录静息心率和呼吸率
                if (this.app.currentHeartRate && this.app.currentRespiratoryRate) {
                    // 更新当前静息数据
                    this.currentRestingHR = this.app.currentHeartRate;
                    this.currentRestingRR = this.app.currentRespiratoryRate;
                    this.lastRestingUpdateTime = now;

                    // 添加到当前分钟数据
                    this.currentMinuteHR.push(this.currentRestingHR);
                    this.currentMinuteRR.push(this.currentRestingRR);

                    // 记录详细数据
                    const record = {
                        timestamp: new Date().toISOString(),
                        heartRate: this.currentRestingHR,
                        respiratoryRate: this.currentRestingRR,
                        restingDuration: this.restingDuration,
                        source: hasActivityMonitor ? 'activity' : 'imu'
                    };
                    this.restingRecords.push(record);

                    // 检查是否需要更新分钟数据
                    this.updateMinuteData(now);

                    // 更新每小时数据
                    this.updateHourlyData(now);
                }
            }
        } else {
            // 不处于静息状态
            if (this.isResting) {
                // 退出静息状态
                this.log(`🚶 退出静息状态 (持续${this.restingDuration.toFixed(1)}秒)`);

                // 如果静息时间足够长，标记为有效
                if (this.restingDuration >= this.minRestingDuration) {
                    this.log(`✅ 有效静息记录`);
                }

                this.isResting = false;
                this.restingDuration = 0;
            }

            // 重置稳定计时
            this.stableStartTime = null;
        }

        // 更新 UI
        this.updateUI();
    }
    
    /**
     * 开始监测
     */
    start() {
        if (this.enabled) {
            this.log('⚠️ 监测已在运行中');
            return;
        }
        
        // 重置数据
        this.restingRecords = [];
        this.isResting = false;
        this.restingStartTime = null;
        this.restingDuration = 0;
        this.stableStartTime = null;
        this.monitorStartTime = new Date();

        // 重置实时数据
        this.currentRestingHR = null;
        this.currentRestingRR = null;
        this.lastRestingUpdateTime = null;
        this.minuteData = [];
        this.currentMinuteHR = [];
        this.currentMinuteRR = [];
        this.lastMinuteTimestamp = null;
        this.hourlyData = Array(24).fill(null).map(() => ({
            avgHR: 0,
            avgRR: 0,
            count: 0,
            samples: []
        }));

        this.enabled = true;

        this.log('🎯 开始静息监测 - 系统将自动识别静息状态');
        if (this.app && this.app.activityMonitor && this.app.activityMonitorEnabled) {
            this.log('✅ 使用活动监测模块判定静息状态');
        } else {
            this.log('⚠️ 活动监测未启用，使用IMU稳定性判定');
        }
        this.log(`⚙️ 参数: 稳定阈值=${this.imuThreshold}, 稳定时长=${this.stableDuration}秒, 最小时长=${this.minRestingDuration}秒`);

        // 初始化图表
        this.initializeCharts();

        // 更新 UI
        document.getElementById('restingStartBtn').style.display = 'none';
        document.getElementById('restingStopBtn').style.display = 'inline-block';
        document.getElementById('restingSaveBtn').style.display = 'inline-block';
        document.getElementById('restingConfigBtn').style.display = 'inline-block';
        document.getElementById('restingStatusPanel').style.display = 'block';
    }
    
    /**
     * 停止监测
     */
    stop() {
        if (!this.enabled) return;
        
        this.enabled = false;
        const duration = (Date.now() - this.monitorStartTime.getTime()) / 1000;
        
        this.log(`📊 静息监测结束 - 总时长: ${(duration/60).toFixed(1)}分钟, 记录: ${this.restingRecords.length}条`);
        
        // 更新 UI
        document.getElementById('restingStartBtn').style.display = 'inline-block';
        document.getElementById('restingStopBtn').style.display = 'none';
    }
    
    /**
     * 保存数据
     */
    save() {
        if (this.restingRecords.length === 0) {
            alert('没有静息数据可保存');
            return;
        }
        
        const timestamp = this.monitorStartTime.toISOString()
            .slice(0, 16).replace('T', '-').replace(/:/g, '-');
        
        // 1. 保存详细数据 (CSV)
        let csvContent = '# 静息心率和呼吸率监测数据\n';
        csvContent += `# 监测开始时间: ${this.monitorStartTime.toISOString()}\n`;
        csvContent += `# 总记录数: ${this.restingRecords.length}\n`;
        csvContent += 'Timestamp,HeartRate(bpm),RespiratoryRate(bpm),RestingDuration(s),IMUStability\n';
        
        for (const record of this.restingRecords) {
            csvContent += `${record.timestamp},${record.heartRate},${record.respiratoryRate},${record.restingDuration.toFixed(1)},${record.imuStability.toFixed(4)}\n`;
        }
        
        const csvFilename = `resting_vitals_${timestamp}.csv`;
        this.downloadFile(csvContent, csvFilename, 'text/csv');
        
        // 2. 生成统计报告
        const stats = this.generateStats();
        const statsJson = JSON.stringify(stats, null, 2);
        const statsFilename = `resting_vitals_${timestamp}_stats.json`;
        this.downloadFile(statsJson, statsFilename, 'application/json');
        
        this.log(`💾 已保存: ${csvFilename}`);
        this.log(`📈 已保存统计: ${statsFilename}`);
    }
    
    /**
     * 生成统计报告
     */
    generateStats() {
        const data = this.restingRecords;
        
        if (data.length === 0) {
            return { error: '无数据' };
        }
        
        const heartRates = data.map(r => r.heartRate);
        const respRates = data.map(r => r.respiratoryRate);
        
        const avgHR = heartRates.reduce((sum, hr) => sum + hr, 0) / heartRates.length;
        const avgRR = respRates.reduce((sum, rr) => sum + rr, 0) / respRates.length;
        
        const minHR = Math.min(...heartRates);
        const maxHR = Math.max(...heartRates);
        const minRR = Math.min(...respRates);
        const maxRR = Math.max(...respRates);
        
        const stdHR = this.calculateStd(heartRates);
        const stdRR = this.calculateStd(respRates);
        
        return {
            monitorStartTime: this.monitorStartTime.toISOString(),
            monitorDuration: (Date.now() - this.monitorStartTime.getTime()) / 1000,
            recordCount: data.length,
            heartRate: {
                average: parseFloat(avgHR.toFixed(2)),
                min: minHR,
                max: maxHR,
                std: parseFloat(stdHR.toFixed(2)),
                range: `${minHR}-${maxHR}`
            },
            respiratoryRate: {
                average: parseFloat(avgRR.toFixed(2)),
                min: minRR,
                max: maxRR,
                std: parseFloat(stdRR.toFixed(2)),
                range: `${minRR}-${maxRR}`
            },
            settings: {
                imuThreshold: this.imuThreshold,
                imuWindowSize: this.imuWindowSize,
                stableDuration: this.stableDuration,
                minRestingDuration: this.minRestingDuration
            }
        };
    }
    
    /**
     * 配置参数
     */
    config() {
        const threshold = prompt(
            '设置 IMU 稳定阈值 (当前: ' + this.imuThreshold + ')\n' +
            '数值越小越严格，建议范围: 0.5-5.0',
            this.imuThreshold
        );
        if (threshold !== null && !isNaN(threshold) && threshold > 0) {
            this.imuThreshold = parseFloat(threshold);
        }
        
        const stableDur = prompt(
            '设置判定静息所需稳定时长(秒) (当前: ' + this.stableDuration + ')\n' +
            '需要连续稳定多久才算进入静息，建议: 3-10秒',
            this.stableDuration
        );
        if (stableDur !== null && !isNaN(stableDur) && stableDur > 0) {
            this.stableDuration = parseFloat(stableDur);
        }
        
        const minDur = prompt(
            '设置最小静息时长(秒) (当前: ' + this.minRestingDuration + ')\n' +
            '低于此时长的静息不会被保存，建议: 10-30秒',
            this.minRestingDuration
        );
        if (minDur !== null && !isNaN(minDur) && minDur > 0) {
            this.minRestingDuration = parseFloat(minDur);
        }
        
        this.log(`⚙️ 参数已更新: 阈值=${this.imuThreshold}, 稳定=${this.stableDuration}秒, 最小=${this.minRestingDuration}秒`);
        this.updateUI();
    }

    /**
     * 初始化图表
     */
    initializeCharts() {
        console.log('🎨 初始化静息心率图表...');

        // 实时静息心率图
        const realtimeCanvas = document.getElementById('restingRealtimeHRChart');
        if (realtimeCanvas) {
            const ctx = realtimeCanvas.getContext('2d');
            this.charts.realtimeHR = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '静息心率',
                        data: [],
                        borderColor: '#FF2D55',
                        backgroundColor: 'rgba(255, 45, 85, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: '心率 (bpm)'
                            }
                        },
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: '时间'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
            console.log('✅ 实时静息心率图初始化成功');
        }

        // 每分钟静息心率图
        const minuteCanvas = document.getElementById('restingMinuteHRChart');
        if (minuteCanvas) {
            const ctx = minuteCanvas.getContext('2d');
            this.charts.minuteHR = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: '每分钟平均心率',
                        data: [],
                        backgroundColor: '#007AFF',
                        borderColor: '#007AFF',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: '心率 (bpm)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
            console.log('✅ 每分钟静息心率图初始化成功');
        }

        // 每小时静息心率图
        const hourlyCanvas = document.getElementById('restingHourlyHRChart');
        if (hourlyCanvas) {
            const ctx = hourlyCanvas.getContext('2d');
            this.charts.hourlyHR = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: Array.from({length: 24}, (_, i) => `${i}:00`),
                    datasets: [{
                        label: '每小时平均心率',
                        data: Array(24).fill(0),
                        backgroundColor: '#34C759',
                        borderColor: '#34C759',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            title: {
                                display: true,
                                text: '心率 (bpm)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
            console.log('✅ 每小时静息心率图初始化成功');
        }

        console.log('🎨 静息心率图表初始化完成');
    }

    /**
     * 更新图表
     */
    updateCharts() {
        // 更新实时心率图（显示最近的记录）
        if (this.charts.realtimeHR && this.restingRecords.length > 0) {
            const recentRecords = this.restingRecords.slice(-60); // 最近60条记录
            const labels = recentRecords.map(r => {
                const date = new Date(r.timestamp);
                return `${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
            });
            const data = recentRecords.map(r => r.heartRate);

            this.charts.realtimeHR.data.labels = labels;
            this.charts.realtimeHR.data.datasets[0].data = data;
            this.charts.realtimeHR.update('none');
        }

        // 更新每分钟心率图
        if (this.charts.minuteHR && this.minuteData.length > 0) {
            const labels = this.minuteData.map(d => {
                const date = new Date(d.timestamp);
                return `${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`;
            });
            const data = this.minuteData.map(d => d.avgHR);

            this.charts.minuteHR.data.labels = labels;
            this.charts.minuteHR.data.datasets[0].data = data;
            this.charts.minuteHR.update('none');
        }

        // 更新每小时心率图
        if (this.charts.hourlyHR) {
            const data = this.hourlyData.map(h => h.avgHR || 0);
            this.charts.hourlyHR.data.datasets[0].data = data;
            this.charts.hourlyHR.update('none');
        }
    }

    /**
     * 更新 UI 显示
     */
    updateUI() {
        // 更新状态显示
        const statusEl = document.getElementById('restingStateText');
        if (statusEl) {
            statusEl.textContent = this.isResting ? '🛌 静息中' : '🚶 活动中';
            statusEl.style.color = this.isResting ? '#28a745' : '#6c757d';
            statusEl.style.fontWeight = 'bold';
        }
        
        // 更新静息持续时间
        const durationEl = document.getElementById('restingCurrentDuration');
        if (durationEl) {
            durationEl.textContent = this.restingDuration > 0 ? 
                `${this.restingDuration.toFixed(1)} 秒` : '-- 秒';
        }
        
        // 更新记录数
        const countEl = document.getElementById('restingTotalRecords');
        if (countEl) {
            countEl.textContent = this.restingRecords.length;
        }
        
        // 计算并显示平均值
        if (this.restingRecords.length > 0) {
            const avgHR = this.restingRecords.reduce((sum, r) => sum + r.heartRate, 0) / this.restingRecords.length;
            const avgRR = this.restingRecords.reduce((sum, r) => sum + r.respiratoryRate, 0) / this.restingRecords.length;
            
            const avgHREl = document.getElementById('restingAvgHR');
            const avgRREl = document.getElementById('restingAvgRR');
            
            if (avgHREl) avgHREl.textContent = `${avgHR.toFixed(1)} bpm`;
            if (avgRREl) avgRREl.textContent = `${avgRR.toFixed(1)} bpm`;
        } else {
            const avgHREl = document.getElementById('restingAvgHR');
            const avgRREl = document.getElementById('restingAvgRR');
            if (avgHREl) avgHREl.textContent = '-- bpm';
            if (avgRREl) avgRREl.textContent = '-- bpm';
        }

        // 更新当前静息心率和呼吸率
        const currentHREl = document.getElementById('restingCurrentHR');
        const currentRREl = document.getElementById('restingCurrentRR');
        if (currentHREl) {
            currentHREl.textContent = this.currentRestingHR ? `${this.currentRestingHR.toFixed(1)} bpm` : '-- bpm';
        }
        if (currentRREl) {
            currentRREl.textContent = this.currentRestingRR ? `${this.currentRestingRR.toFixed(1)} bpm` : '-- bpm';
        }

        // 更新最后更新时间
        const lastUpdateEl = document.getElementById('restingLastUpdate');
        if (lastUpdateEl && this.lastRestingUpdateTime) {
            const date = new Date(this.lastRestingUpdateTime);
            lastUpdateEl.textContent = `${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
        }

        // 更新图表
        this.updateCharts();

        // 更新参数显示
        const thresholdEl = document.getElementById('restingThresholdDisplay');
        const stableDurEl = document.getElementById('restingStableDurDisplay');
        const minDurEl = document.getElementById('restingMinDurDisplay');
        
        if (thresholdEl) thresholdEl.textContent = this.imuThreshold;
        if (stableDurEl) stableDurEl.textContent = this.stableDuration;
        if (minDurEl) minDurEl.textContent = this.minRestingDuration;
    }
    
    /**
     * 下载文件
     */
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    /**
     * 添加日志
     */
    log(message) {
        const logEl = document.getElementById('restingLog');
        if (logEl) {
            const time = new Date().toLocaleTimeString();
            logEl.textContent += `[${time}] ${message}\n`;
            logEl.scrollTop = logEl.scrollHeight;
        }
        console.log(`[RestingMonitor] ${message}`);
    }
    
    /**
     * 清空数据
     */
    clear() {
        if (confirm('确定要清空静息监测数据吗？')) {
            this.restingRecords = [];
            this.isResting = false;
            this.restingStartTime = null;
            this.restingDuration = 0;
            this.stableStartTime = null;
            
            const logEl = document.getElementById('restingLog');
            if (logEl) logEl.textContent = '';
            
            this.updateUI();
            this.log('🔄 数据已清空');
        }
    }
}

const restingMonitor = new RestingMonitor();

export default restingMonitor;
export { RestingMonitor, restingMonitor };

if (typeof window !== 'undefined') {
    window.RestingMonitor = RestingMonitor;
    window.restingMonitor = restingMonitor;
}
