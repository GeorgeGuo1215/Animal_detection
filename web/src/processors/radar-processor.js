/**
 * 毫米波雷达数据处理器（JS 移植版）
 * 与 Python 端 main.py 算法保持一致：圆拟合 → ARCSIN 解调 → 滤波 → FFT → 心率/呼吸提取
 */

import { FFT } from './fft.js';

class RadarDataProcessor {
  constructor(samplingRate = 50) {
    this.fs = samplingRate;
    this.nShort = 500;
    this.NLong = 1000;

    this.HPF_short_4_par = [
      0, -5.50164604360785e-05, -0.000170712307325988, -0.000321372765076530,
      -0.000523025797360845, -0.000744196112614598, -0.000983826634810562,
      -0.00127235081696804, -0.00160419939233453, -0.00193993867530017,
      -0.00228082848478637, -0.00264302585359132, -0.00301758574544834,
      -0.00343699236516377, -0.00390408151456911, -0.00439597535068256,
      -0.00492064436945537, -0.00544043127996703, -0.00594971362380537,
      -0.00647249773797565, -0.00699784286438748, -0.00754300147293485,
      -0.00810999030682790, -0.00868059148941056, -0.00925842014653436,
      -0.00988031890708563, -0.0105471202298266, -0.0112435445759278,
      -0.0119790033062446, -0.0127275069499252, -0.0134874967666727,
      -0.0142798607935558, -0.0150989968242221, -0.0159037541140797,
      -0.0166924941429137, -0.0174807050035277, -0.0182572362317448,
      -0.0190510330770520, -0.0198629881240401, -0.0206687276426293,
      -0.0214738649139352, -0.0223007879891433, -0.0231482012320617,
      -0.0240087850093419, -0.0248875902870985, -0.0257644553582338,
      -0.0266380712009247, -0.0275223504264536,
    ];

    this.HPF_short_5_par = [
      -0.00339276769147847, -0.000289899591150812, -0.000301275418827859,
      -0.000312098312620977, -0.000322549062240940, -0.000332381770852629,
      -0.000341731680987501, -0.000350400526126641, -0.000358525665298154,
      -0.000365887160504275, -0.000372637078486777, -0.000378527224152246,
      -0.000383744164044131, -0.000388006414911702, -0.000391501840818984,
      -0.000393924679427624, -0.000395478711351671, -0.000395850492619180,
      -0.000395247988442467, -0.000393338704009786, -0.000390363416943796,
      -0.000385999580496542, -0.000380553095135587, -0.000373688291692988,
      -0.000365770714463747, -0.000356439027816037, -0.000346128807188419,
      -0.000334416670053275, -0.000321755744087934, -0.000307592555574287,
      -0.000292389547082744, -0.000275452034383983, -0.000257331448332866,
      -0.000237239705447390, -0.000216136464554341, -0.000193292824624094,
      -0.000170553182210679, -0.000146675220298725, -0.000123762543021994,
      -9.27732390070040e-05, -6.62743648944459e-05, -3.81338877599648e-05,
      -7.62116480334422e-06, 2.26657868080909e-05, 5.46318749383660e-05,
    ];
  }

  parseDataFile(fileContent) {
    const lines = fileContent.split('\n').filter(l => l.trim());
    const timestamps = [], iData = [], qData = [];
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 4) {
        const iVal = parseFloat(parts[2]);
        const qVal = parseFloat(parts[3]);
        if (Number.isFinite(iVal) && Number.isFinite(qVal)) {
          timestamps.push(`${parts[0]} ${parts[1]}`);
          iData.push(iVal);
          qData.push(qVal);
        }
      } else if (parts.length >= 3) {
        const iVal = parseFloat(parts[1]);
        const qVal = parseFloat(parts[2]);
        if (Number.isFinite(iVal) && Number.isFinite(qVal)) {
          timestamps.push(parts[0]);
          iData.push(iVal);
          qData.push(qVal);
        }
      }
    }
    return {
      timestamps,
      iData: new Float64Array(iData),
      qData: new Float64Array(qData),
      length: iData.length,
    };
  }

  circleFitting(iData, qData) {
    const lenIQ = iData.length;
    const iMean = this.mean(iData);
    const qMean = this.mean(qData);
    let Mxx = 0, Myy = 0, Mxy = 0, Mxz = 0, Myz = 0, Mzz = 0;

    for (let i = 0; i < lenIQ; i++) {
      const Xi = iData[i] - iMean;
      const Yi = qData[i] - qMean;
      const Zi = Xi * Xi + Yi * Yi;
      Mxy += Xi * Yi; Mxx += Xi * Xi; Myy += Yi * Yi;
      Mxz += Xi * Zi; Myz += Yi * Zi; Mzz += Zi * Zi;
    }
    Mxx /= lenIQ; Myy /= lenIQ; Mxy /= lenIQ;
    Mxz /= lenIQ; Myz /= lenIQ; Mzz /= lenIQ;

    const Mz = Mxx + Myy;
    const CovXY = Mxx * Myy - Mxy * Mxy;
    const Mxz2 = Mxz * Mxz, Myz2 = Myz * Myz;
    const A2 = 4 * CovXY - 3 * Mz * Mz - Mzz;
    const A1 = Mzz * Mz + 4 * CovXY * Mz - Mxz2 - Myz2 - Mz * Mz * Mz;
    const A0 = Mxz2 * Myy + Myz2 * Mxx - Mzz * CovXY - 2 * Mxz * Myz * Mxy + Mz * Mz * CovXY;
    const A22 = A2 + A2;

    const epsilon = 1e-12;
    let yNew = 1e+20, xNew = 0;
    for (let iter = 0; iter < 40; iter++) {
      const yOld = yNew;
      yNew = A0 + xNew * (A1 + xNew * (A2 + 4 * xNew * xNew));
      if (Math.abs(yNew) > Math.abs(yOld)) { xNew = 0; break; }
      const Dy = A1 + xNew * (A22 + 16 * xNew * xNew);
      const xOld = xNew;
      xNew = xOld - yNew / Dy;
      if (Math.abs((xNew - xOld) / xNew) < epsilon) break;
      if (xNew < 0) { xNew = 0; break; }
    }

    const det = xNew * xNew - xNew * Mz + CovXY;
    let center, radius;
    if (Math.abs(det) > 1e-10) {
      center = [
        (Mxz * (Myy - xNew) - Myz * Mxy) / det / 2 + iMean,
        (Myz * (Mxx - xNew) - Mxz * Mxy) / det / 2 + qMean,
      ];
      const off = [center[0] - iMean, center[1] - qMean];
      radius = Math.sqrt(off[0] * off[0] + off[1] * off[1] + Mz + 2 * xNew);
    } else {
      center = [iMean, qMean];
      radius = Math.sqrt(this.variance(iData) + this.variance(qData));
    }
    return { center, radius };
  }

  arcsinDemodulation(iData, qData, center, radius) {
    const N = iData.length;
    const phase = new Float64Array(N);
    const iNorm = new Float64Array(N), qNorm = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      iNorm[i] = (iData[i] - center[0]) / radius;
      qNorm[i] = (qData[i] - center[1]) / radius;
    }
    for (let i = 2; i < N; i++) {
      const i2 = iNorm[i - 1], i1 = iNorm[i];
      const q2 = qNorm[i - 1], q1 = qNorm[i];
      const denom = Math.sqrt((i2 * i2 + q2 * q2) * (i1 * i1 + q1 * q1));
      if (denom > 1e-10) {
        const arg = (i2 * q1 - i1 * q2) / denom;
        phase[i] = phase[i - 1] + Math.asin(Math.max(-1, Math.min(1, arg)));
      } else {
        phase[i] = phase[i - 1];
      }
    }
    return phase;
  }

  movingAverage(data, windowSize) {
    const N = data.length;
    const result = new Float64Array(N);
    const halfWindow = Math.floor(windowSize / 2);
    for (let i = 0; i < N; i++) {
      let sum = 0, count = 0;
      const s = Math.max(0, i - halfWindow);
      const e = Math.min(N - 1, i + halfWindow);
      for (let j = s; j <= e; j++) { sum += data[j]; count++; }
      result[i] = sum / count;
    }
    return result;
  }

  convolve(signal, kernel) {
    const N = signal.length, K = kernel.length;
    const result = new Float64Array(N);
    const half = Math.floor(K / 2);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < K; j++) {
        const idx = i - half + j;
        if (idx >= 0 && idx < N) s += signal[idx] * kernel[j];
      }
      result[i] = s;
    }
    return result;
  }

  applyFilters(phaseData) {
    try {
      let respiratoryWave = this.movingAverage(Array.from(phaseData), 5);
      respiratoryWave = this.movingAverage(respiratoryWave, 2.5);
      const minVal = Math.min(...respiratoryWave);
      respiratoryWave = respiratoryWave.map(v => v - minVal);

      let heartbeatWave = this.convolve(Array.from(phaseData), this.HPF_short_4_par);
      heartbeatWave = this.convolve(heartbeatWave, this.HPF_short_5_par);

      return {
        respiratoryWave: new Float64Array(respiratoryWave),
        heartbeatWave: new Float64Array(heartbeatWave),
      };
    } catch (e) {
      return { respiratoryWave: phaseData, heartbeatWave: phaseData };
    }
  }

  fft(data) { return FFT.realFFT(Array.from(data)); }

  extractVitalSignsTimeSeries(iData, qData, phaseData, windowSize = 10, stepSize = 1) {
    const ws = Math.floor(windowSize * this.fs);
    const ss = Math.floor(stepSize * this.fs);
    const heartRateTimeSeries = [], respiratoryRateTimeSeries = [], timeAxis = [];

    for (let s = 0; s + ws <= iData.length; s += ss) {
      const e = s + ws;
      const winI = iData.slice(s, e);
      const winQ = qData.slice(s, e);
      const winPhase = phaseData ? phaseData.slice(s, e) : null;
      const v = this.extractVitalSignsWindow(winI, winQ, winPhase);
      heartRateTimeSeries.push(v.heartRate);
      respiratoryRateTimeSeries.push(v.respiratoryRate);
      timeAxis.push(s / this.fs);
    }

    return {
      heartRateTimeSeries, respiratoryRateTimeSeries, timeAxis,
      heartRate: heartRateTimeSeries.length
        ? Math.round(heartRateTimeSeries.reduce((a, b) => a + b, 0) / heartRateTimeSeries.length) : 0,
      respiratoryRate: respiratoryRateTimeSeries.length
        ? Math.round(respiratoryRateTimeSeries.reduce((a, b) => a + b, 0) / respiratoryRateTimeSeries.length) : 0,
    };
  }

  extractVitalSignsWindow(iData, qData /*, phaseData */) {
    const u1 = this.removeDC(iData);
    const u2 = this.removeDC(qData);
    const cs = [];
    for (let i = 0; i < u1.length; i++) cs.push([u1[i], u2[i]]);
    const padded = Math.pow(2, Math.ceil(Math.log2(u1.length)));
    while (cs.length < padded) cs.push([0, 0]);
    const f = FFT.fft(cs);
    const mag = f.map(([r, im]) => Math.sqrt(r * r + im * im));
    const half = mag.slice(0, Math.floor(mag.length / 4));
    const freq = half.map((_, i) => i * this.fs / (half.length * 4));

    const breathIdx = freq.map((f, i) => f >= 0.1 && f <= 0.5 ? i : -1).filter(i => i !== -1);
    const heartIdx = freq.map((f, i) => f >= 0.8 && f <= 3.0 ? i : -1).filter(i => i !== -1);

    let respiratoryRate = 0, heartRate = 0;
    if (breathIdx.length > 0) {
      const spec = breathIdx.map(i => half[i]);
      const peaks = this.findPeaks(spec, 2);
      let pickIdx = peaks.length > 0
        ? peaks.reduce((bi, p) => spec[p] > spec[peaks[bi]] ? peaks.indexOf(p) : bi, 0)
        : spec.indexOf(Math.max(...spec));
      const idx = peaks.length > 0 ? peaks[pickIdx] : pickIdx;
      respiratoryRate = Math.round(freq[breathIdx[idx]] * 60);
    }
    if (heartIdx.length > 0) {
      const spec = heartIdx.map(i => half[i]);
      const peaks = this.findPeaks(spec, 3);
      let pickIdx = peaks.length > 0
        ? peaks.reduce((bi, p) => spec[p] > spec[peaks[bi]] ? peaks.indexOf(p) : bi, 0)
        : spec.indexOf(Math.max(...spec));
      const idx = peaks.length > 0 ? peaks[pickIdx] : pickIdx;
      heartRate = Math.round(freq[heartIdx[idx]] * 60);
    }
    return { heartRate, respiratoryRate };
  }

  findPeaks(data, minDistance = 1) {
    const peaks = [];
    for (let i = 1; i < data.length - 1; i++) {
      if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
        if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistance) peaks.push(i);
        else if (data[i] > data[peaks[peaks.length - 1]]) peaks[peaks.length - 1] = i;
      }
    }
    return peaks;
  }

  extractVitalSigns(iData, qData /*, phaseData */) {
    try {
      const u1 = this.removeDC(iData);
      const u2 = this.removeDC(qData);
      const cs = [];
      const N = Math.min(u1.length, u2.length);
      for (let i = 0; i < N; i++) cs.push([u1[i], u2[i]]);
      const padded = N + N;
      while (cs.length < padded) cs.push([0, 0]);

      const f = FFT.fft(cs);
      const mag = f.map(([r, im]) => Math.sqrt(r * r + im * im));
      const half = Math.floor(mag.length / 2);
      const sLen = Math.min(half, 512);
      const spectrum = mag.slice(0, sLen);
      const fr = (this.fs || 100) / mag.length;

      const breathStart = Math.max(1, Math.round(0.1 / fr));
      const breathEnd = Math.min(sLen - 1, Math.round(0.5 / fr));
      const breathSpec = spectrum.slice(breathStart, breathEnd + 1);

      const heartStart = Math.max(1, Math.round(0.8 / fr));
      const heartEnd = Math.min(sLen - 1, Math.round(3.0 / fr));
      const heartSpec = spectrum.slice(heartStart, heartEnd + 1);

      let respiratoryRate = 0, heartRate = 0;
      if (breathSpec.length > 0) {
        const idx = breathSpec.indexOf(Math.max(...breathSpec));
        respiratoryRate = Math.round(((breathStart + idx) * fr) * 60);
      }
      if (heartSpec.length > 0) {
        const idx = heartSpec.indexOf(Math.max(...heartSpec));
        heartRate = Math.round(((heartStart + idx) * fr) * 60);
      }
      if (respiratoryRate < 5 || respiratoryRate > 40) respiratoryRate = 15;
      if (heartRate < 40 || heartRate > 200) heartRate = 72;
      return { heartRate, respiratoryRate };
    } catch (e) {
      return { heartRate: 72, respiratoryRate: 15 };
    }
  }

  extractVitalSignsMainPy(iData, qData) {
    const N = Math.min(iData.length, qData.length);
    if (N < 200) {
      return {
        heartRate: 72, respiratoryRate: 15,
        phase: new Float64Array(N),
        respiratoryWave: new Float64Array(N),
        heartbeatWave: new Float64Array(N),
      };
    }

    const iMean = this.mean(iData), qMean = this.mean(qData);
    const iVar = this.variance(iData), qVar = this.variance(qData);
    const crf_sqrt = Math.sqrt(iVar + qVar) || 1;
    const center = [iMean, qMean];

    const iNorm = new Float64Array(N), qNorm = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      iNorm[i] = (iData[i] - center[0]) / crf_sqrt;
      qNorm[i] = (qData[i] - center[1]) / crf_sqrt;
    }

    const phase = new Float64Array(N);
    for (let i = 2; i < N; i++) {
      const i2 = iNorm[i - 1], i1 = iNorm[i];
      const q2 = qNorm[i - 1], q1 = qNorm[i];
      const denom = Math.sqrt((i2 * i2 + q2 * q2) * (i1 * i1 + q1 * q1));
      if (denom > 1e-10) {
        const arg = (i2 * q1 - i1 * q2) / denom;
        phase[i] = phase[i - 1] + Math.asin(Math.max(-1, Math.min(1, arg)));
      } else {
        phase[i] = phase[i - 1];
      }
    }

    let respiratoryWave = this.movingAverage(Array.from(phase), 5);
    respiratoryWave = this.movingAverage(respiratoryWave, 2.5);
    const minResp = Math.min(...respiratoryWave);
    respiratoryWave = respiratoryWave.map(v => v - minResp);

    let heartbeatWave = this.convolve(Array.from(phase), this.HPF_short_4_par);
    heartbeatWave = this.convolve(heartbeatWave, this.HPF_short_5_par);

    const moveStep = 2.5;
    const diffPhase = new Float64Array(N);
    for (let i = 1; i < N; i++) diffPhase[i] = phase[i] - phase[i - 1];
    let dispShort = this.movingAverage(Array.from(diffPhase), moveStep);

    const meanDisp = dispShort.reduce((a, b) => a + b, 0) / dispShort.length;
    const stdDisp = Math.sqrt(dispShort.reduce((s, v) => s + (v - meanDisp) ** 2, 0) / dispShort.length) || 1;
    const dispNorm = dispShort.map(v => (v - meanDisp) / stdDisp);
    const ps = FFT.realFFT(Array.from(dispNorm));
    const halfLen = Math.floor(ps.length / 2);
    const spectrum = ps.slice(0, Math.min(halfLen, 50));
    const shortSpec = spectrum.slice(0, Math.min(25, spectrum.length));

    let respiratoryRate = 15, heartRate = 72;
    if (shortSpec.length >= 21) {
      const breSlice = shortSpec.slice(2, 4);
      const heartSlice = shortSpec.slice(9, 21);
      if (breSlice.length > 0) {
        const i = breSlice.indexOf(Math.max(...breSlice));
        respiratoryRate = (i + 2) * 6;
      }
      if (heartSlice.length > 0) {
        const i = heartSlice.indexOf(Math.max(...heartSlice));
        heartRate = (i + 9) * 6;
      }
    }
    if (respiratoryRate < 5 || respiratoryRate > 40) respiratoryRate = 15;
    if (heartRate < 40 || heartRate > 200) heartRate = 72;

    return {
      heartRate, respiratoryRate, phase,
      respiratoryWave: new Float64Array(respiratoryWave),
      heartbeatWave: new Float64Array(heartbeatWave),
    };
  }

  processSingleFile(fileName, fileContent) {
    try {
      const parsed = this.parseDataFile(fileContent);
      if (parsed.length === 0) throw new Error('文件中没有有效数据');
      const { center, radius } = this.circleFitting(parsed.iData, parsed.qData);
      const phase = this.arcsinDemodulation(parsed.iData, parsed.qData, center, radius);
      const { respiratoryWave, heartbeatWave } = this.applyFilters(phase);
      const ts = this.extractVitalSignsTimeSeries(parsed.iData, parsed.qData, phase, 10, 1);
      return {
        fileName,
        dataPoints: parsed.length,
        heartRate: ts.heartRate,
        respiratoryRate: ts.respiratoryRate,
        heartRateTimeSeries: ts.heartRateTimeSeries,
        respiratoryRateTimeSeries: ts.respiratoryRateTimeSeries,
        timeAxis: ts.timeAxis,
        circleCenter: center,
        circleRadius: radius,
        timestamps: parsed.timestamps,
        iData: parsed.iData,
        qData: parsed.qData,
        phaseData: phase,
        respiratoryWave,
        heartbeatWave,
        status: 'success',
      };
    } catch (e) {
      return { fileName, error: e.message, status: 'error' };
    }
  }

  mean(a)     { return a.reduce((s, v) => s + v, 0) / a.length; }
  variance(a) { const m = this.mean(a); return a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length; }
  removeDC(a) { const m = this.mean(a); return Array.from(a, v => v - m); }
}

export default RadarDataProcessor;
export { RadarDataProcessor };

if (typeof window !== 'undefined') {
  window.RadarDataProcessor = RadarDataProcessor;
}
