/**
 * 活动量与步数监测模块（ESM 版本）
 * 基于 IMU 加速度计数据，实现 ENMO/MAD 活动量与步数计数。
 *
 * 注：相比原 activity-monitor.js + activity-fix-patch.js，全局样本计数已合入主体，无需补丁。
 */

class ActivityMonitor {
  constructor(samplingRate = 50, petWeight = 10.0) {
    this.fs = samplingRate;
    this.petWeight = petWeight;

    this.rerDaily = 70 * Math.pow(this.petWeight, 0.75);
    this.bmrPerSec = this.rerDaily / 86400.0;

    this.accBufferX = []; this.accBufferY = []; this.accBufferZ = [];
    this.timestamps = [];

    this.activityHistory = [];
    this.stepHistory = [];

    this.totalENMO = 0;
    this.totalSteps = 0;
    this.totalCalories = 0;
    this.dailyGoal = 1000;
    this.activityGoal = 2.0;
    this.calorieGoal = 100;

    this.currentIntensity = 'resting';
    this.currentMETs = 1.0;
    this.lastUpdateTime = Date.now();

    this.globalSampleCount = 0;
    this.lastPeakGlobalIndex = -1;
    this.minPeakDistance = Math.floor(this.fs / 3);

    this.charts = {};
    this.filterWindowSize = Math.floor(this.fs / 2);

    this.hourlyData = new Array(24).fill(null).map(() => ({
      steps: 0, activity: 0, calories: 0, intensity: 'resting',
    }));

    this.lastChartUpdate = 0;
    this.chartUpdateInterval = 1000;
  }

  addAccelerometerData(ax, ay, az, timestamp) {
    this.accBufferX.push(ax);
    this.accBufferY.push(ay);
    this.accBufferZ.push(az);
    this.timestamps.push(timestamp);
    this.globalSampleCount++;

    const maxBufferSize = this.fs * 10;
    if (this.accBufferX.length > maxBufferSize) {
      this.accBufferX.shift(); this.accBufferY.shift(); this.accBufferZ.shift();
      this.timestamps.shift();
    }

    if (this.globalSampleCount % this.fs === 0 && this.accBufferX.length >= this.fs) {
      this.processActivityMetrics();
    }
  }

  calculateActivityMetrics(accX, accY, accZ) {
    const n = accX.length;
    if (n === 0) return { enmo: 0, mad: 0, intensity: 'resting' };
    const svm = [];
    for (let i = 0; i < n; i++) {
      svm.push(Math.sqrt(accX[i] ** 2 + accY[i] ** 2 + accZ[i] ** 2));
    }
    let enmoSum = 0;
    for (let i = 0; i < n; i++) enmoSum += Math.max(0, svm[i] - 1.0);
    const enmo = enmoSum / this.fs;
    const svmMean = svm.reduce((a, b) => a + b, 0) / n;
    let madSum = 0;
    for (let i = 0; i < n; i++) madSum += Math.abs(svm[i] - svmMean);
    const mad = madSum / n;
    let intensity = 'resting';
    if (mad > 0.20) intensity = 'vigorous';
    else if (mad > 0.12) intensity = 'moderate';
    else if (mad > 0.05) intensity = 'light';
    let mets = 1.0;
    if (enmo > 0.50) mets = 6.0;
    else if (enmo > 0.20) mets = 4.0;
    else if (enmo > 0.05) mets = 2.0;
    const duration = n / this.fs;
    const calories = this.bmrPerSec * mets * duration;
    return { enmo, mad, intensity, mets, calories, svm };
  }

  countStepsInWindow(accX, accY, accZ, startGlobalIndex) {
    const n = accX.length;
    if (n < 15) return 0;
    const svm = [];
    for (let i = 0; i < n; i++) {
      svm.push(Math.sqrt(accX[i] ** 2 + accY[i] ** 2 + accZ[i] ** 2));
    }
    const filtered = this.simpleBandpassFilter(svm);
    const minPeakHeight = 0.10;
    let newSteps = 0;
    for (let i = 1; i < filtered.length - 1; i++) {
      const globalIndex = startGlobalIndex + i;
      if (filtered[i] > filtered[i - 1] && filtered[i] > filtered[i + 1] && filtered[i] > minPeakHeight) {
        if (globalIndex - this.lastPeakGlobalIndex >= this.minPeakDistance) {
          newSteps++;
          this.lastPeakGlobalIndex = globalIndex;
        }
      }
    }
    return newSteps;
  }

  simpleBandpassFilter(signal) {
    const n = signal.length;
    if (n < 5) return signal;
    const smoothed = [];
    const win = 3;
    for (let i = 0; i < n; i++) {
      let s = 0, c = 0;
      for (let j = Math.max(0, i - win); j <= Math.min(n - 1, i + win); j++) { s += signal[j]; c++; }
      smoothed.push(s / c);
    }
    const mean = smoothed.reduce((a, b) => a + b, 0) / n;
    return smoothed.map(v => v - mean);
  }

  processActivityMetrics() {
    const now = Date.now();
    const cur = this.accBufferX.length;
    if (cur < this.fs) return;
    const newDataX = this.accBufferX.slice(-this.fs);
    const newDataY = this.accBufferY.slice(-this.fs);
    const newDataZ = this.accBufferZ.slice(-this.fs);
    const metrics = this.calculateActivityMetrics(newDataX, newDataY, newDataZ);

    let newSteps = 0;
    const winSize = Math.min(cur, this.fs * 2);
    const startIdx = cur - winSize;
    const startGlobal = this.globalSampleCount - winSize;
    newSteps = this.countStepsInWindow(
      this.accBufferX.slice(startIdx),
      this.accBufferY.slice(startIdx),
      this.accBufferZ.slice(startIdx),
      startGlobal,
    );

    if (metrics.intensity !== 'resting') this.totalENMO += metrics.enmo;
    this.totalCalories += metrics.calories;
    this.currentMETs = metrics.mets;
    if (newSteps > 0) this.totalSteps += newSteps;

    this.activityHistory.push({
      timestamp: now, enmo: metrics.enmo, mad: metrics.mad,
      intensity: metrics.intensity, mets: metrics.mets, calories: metrics.calories,
    });
    if (newSteps > 0) this.stepHistory.push({ timestamp: now, steps: newSteps });

    const maxAge = 3600 * 1000;
    this.activityHistory = this.activityHistory.filter(it => now - it.timestamp < maxAge);
    this.stepHistory = this.stepHistory.filter(it => now - it.timestamp < maxAge);
    this.updateHourlyStats(now, metrics.enmo, newSteps, metrics.intensity, metrics.calories);
    this.currentIntensity = metrics.intensity;
    this.lastUpdateTime = now;

    if (now - this.lastChartUpdate > this.chartUpdateInterval) {
      this.updateCharts();
      this.lastChartUpdate = now;
    }
  }

  updateHourlyStats(timestamp, enmo, steps, intensity, calories) {
    const hour = new Date(timestamp).getHours();
    this.hourlyData[hour].steps += steps;
    this.hourlyData[hour].activity += enmo;
    this.hourlyData[hour].calories = (this.hourlyData[hour].calories || 0) + calories;
    const lvl = { resting: 0, light: 1, moderate: 2, vigorous: 3 };
    if ((lvl[intensity] || 0) > (lvl[this.hourlyData[hour].intensity] || 0)) {
      this.hourlyData[hour].intensity = intensity;
    }
  }

  initializeCharts() {
    if (typeof Chart === 'undefined') return;
    const mkRing = (id, color) => {
      const el = document.getElementById(id);
      if (!el) return null;
      return new Chart(el, {
        type: 'doughnut',
        data: { labels: ['完成', '剩余'], datasets: [{ data: [0, 100], backgroundColor: [color, '#E5E5EA'], borderWidth: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, cutout: '76%', plugins: { legend: { display: false }, tooltip: { enabled: false } } },
      });
    };
    this.charts.stepsRing    = mkRing('activityStepsRingChart', '#1D1D1F');
    this.charts.activityRing = mkRing('activityENMORingChart',  '#1D1D1F');
    this.charts.calorieRing  = mkRing('activityCalorieRingChart','#1D1D1F');

    const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);
    const mkBar = (id, label, color) => {
      const el = document.getElementById(id);
      if (!el) return null;
      return new Chart(el, {
        type: 'bar',
        data: { labels: hours, datasets: [{ label, data: new Array(24).fill(0), backgroundColor: color, borderRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } },
      });
    };
    this.charts.hourlySteps    = mkBar('activityHourlyStepsChart',   '步数',          '#1D1D1F');
    this.charts.hourlyActivity = mkBar('activityHourlyENMOChart',    '活动量',        '#1D1D1F');
    this.charts.hourlyCalorie  = mkBar('activityHourlyCalorieChart', '卡路里 (kcal)', '#1D1D1F');

    const trendEl = document.getElementById('activityIntensityTrendChart');
    if (trendEl) {
      this.charts.intensityTrend = new Chart(trendEl, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'MAD 强度', data: [], borderColor: '#1D1D1F', backgroundColor: 'rgba(0,0,0,0.06)', fill: true, tension: 0.4, pointRadius: 0 }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } },
      });
    }
  }

  updateCharts() {
    if (Object.keys(this.charts).length === 0) return;
    this.updateRingCharts();
    this.updateHourlyCharts();
    this.updateIntensityTrend();
    this.updateStatistics();
  }

  updateRingCharts() {
    const upd = (chart, total, goal) => {
      if (!chart) return;
      const p = Math.min(100, (total / goal) * 100);
      chart.data.datasets[0].data = [p, 100 - p];
      chart.update('none');
    };
    upd(this.charts.stepsRing, this.totalSteps, this.dailyGoal);
    upd(this.charts.activityRing, this.totalENMO, this.activityGoal);
    upd(this.charts.calorieRing, this.totalCalories, this.calorieGoal);
  }

  updateHourlyCharts() {
    const setDS = (chart, data) => {
      if (!chart) return;
      chart.data.datasets[0].data = data;
      chart.update('none');
    };
    setDS(this.charts.hourlySteps, this.hourlyData.map(h => h.steps));
    setDS(this.charts.hourlyActivity, this.hourlyData.map(h => +h.activity.toFixed(2)));
    setDS(this.charts.hourlyCalorie, this.hourlyData.map(h => +(h.calories || 0).toFixed(2)));
  }

  updateIntensityTrend() {
    if (!this.charts.intensityTrend) return;
    const tenMinAgo = Date.now() - 10 * 60 * 1000;
    const recent = this.activityHistory.filter(it => it.timestamp >= tenMinAgo);
    if (recent.length === 0) return;
    const sampled = [];
    for (let i = 0; i < recent.length; i += 5) sampled.push(recent[i]);
    const labels = sampled.map(it => {
      const d = new Date(it.timestamp);
      return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
    });
    this.charts.intensityTrend.data.labels = labels;
    this.charts.intensityTrend.data.datasets[0].data = sampled.map(it => +it.mad.toFixed(4));
    this.charts.intensityTrend.update('none');
  }

  updateStatistics() {
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('activityTotalSteps', this.totalSteps.toLocaleString());
    set('activityStepsPercent', `${Math.round((this.totalSteps / this.dailyGoal) * 100)}%`);
    set('activityTotalENMO', this.totalENMO.toFixed(2));
    set('activityENMOPercent', `${Math.round((this.totalENMO / this.activityGoal) * 100)}%`);
    set('activityTotalCalories', this.totalCalories.toFixed(2));
    set('activityCaloriePercent', `${Math.round((this.totalCalories / this.calorieGoal) * 100)}%`);
    set('activityCurrentMETs', this.currentMETs.toFixed(1));
    set('activityPetWeight', `${this.petWeight.toFixed(1)} kg`);
    set('activityRER', `${this.rerDaily.toFixed(0)} kcal/day`);

    const intensityText = { resting: '静息', light: '轻度活动', moderate: '中度活动', vigorous: '剧烈活动' };
    set('activityCurrentIntensity', intensityText[this.currentIntensity] || '未知');

    const lastEl = document.getElementById('activityLastUpdate');
    if (lastEl) {
      const d = new Date(this.lastUpdateTime);
      lastEl.textContent = `${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
    }
  }

  resetDailyData() {
    this.totalSteps = 0; this.totalENMO = 0; this.totalCalories = 0;
    this.lastPeakGlobalIndex = -1;
    this.hourlyData = new Array(24).fill(null).map(() => ({ steps: 0, activity: 0, calories: 0, intensity: 'resting' }));
    this.activityHistory = []; this.stepHistory = [];
    this.updateCharts();
  }

  getSummary() {
    return {
      totalSteps: this.totalSteps, totalENMO: this.totalENMO, totalCalories: this.totalCalories,
      petWeight: this.petWeight, rerDaily: this.rerDaily, currentMETs: this.currentMETs,
      stepsGoalPercent: Math.round((this.totalSteps / this.dailyGoal) * 100),
      activityGoalPercent: Math.round((this.totalENMO / this.activityGoal) * 100),
      calorieGoalPercent: Math.round((this.totalCalories / this.calorieGoal) * 100),
      currentIntensity: this.currentIntensity, lastUpdate: this.lastUpdateTime,
    };
  }
}

export default ActivityMonitor;
export { ActivityMonitor };

if (typeof window !== 'undefined') window.ActivityMonitor = ActivityMonitor;
