/**
 * 简化的 FFT 实现（Cooley-Tukey），用于雷达 I/Q 信号处理。
 */

class SimpleFFT {
  /**
   * 复数 FFT。
   * @param {Array<[number,number]>} x - [[real, imag], ...]
   * @returns {Array<[number,number]>}
   */
  static fft(x) {
    const N = x.length;
    if (N <= 1) return x;
    if (N & (N - 1)) {
      const nextPow2 = Math.pow(2, Math.ceil(Math.log2(N)));
      const padded = [...x];
      while (padded.length < nextPow2) padded.push([0, 0]);
      return SimpleFFT.fft(padded);
    }

    const even = [], odd = [];
    for (let i = 0; i < N; i++) (i % 2 === 0 ? even : odd).push(x[i]);
    const evenFFT = SimpleFFT.fft(even);
    const oddFFT = SimpleFFT.fft(odd);

    const result = new Array(N);
    for (let k = 0; k < N / 2; k++) {
      const angle = -2 * Math.PI * k / N;
      const wr = Math.cos(angle), wi = Math.sin(angle);
      const or = oddFFT[k][0], oi = oddFFT[k][1];
      const tr = or * wr - oi * wi;
      const ti = or * wi + oi * wr;
      result[k] = [evenFFT[k][0] + tr, evenFFT[k][1] + ti];
      result[k + N / 2] = [evenFFT[k][0] - tr, evenFFT[k][1] - ti];
    }
    return result;
  }

  static realFFT(realData) {
    const complexData = realData.map(v => [v, 0]);
    const r = SimpleFFT.fft(complexData);
    return r.map(([re, im]) => Math.sqrt(re * re + im * im));
  }

  static powerSpectrum(data) {
    const r = SimpleFFT.realFFT(data);
    return r.map(m => m * m);
  }
}

export default SimpleFFT;
export const FFT = {
  fft: SimpleFFT.fft.bind(SimpleFFT),
  realFFT: SimpleFFT.realFFT.bind(SimpleFFT),
  powerSpectrum: SimpleFFT.powerSpectrum.bind(SimpleFFT),
};

if (typeof window !== 'undefined') {
  window.FFT = FFT;
}
