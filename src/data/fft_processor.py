"""
FFT 频谱处理器
对应固件: src/service/signal_process/imu_data_process.c
处理流程: 去直流 → 去线性趋势 → Hann窗 → FFT → 幅度谱(dB)
"""

import numpy as np
from scipy.signal import windows as sig_windows

from src.config import (
    SP_FFT_SIZE, SP_HOP_SIZE, SP_FREQ_BINS,
    SP_SAMPLE_RATE, SP_FREQ_RES,
)


def make_hann() -> np.ndarray:
    """生成 Hann 窗, float32"""
    return sig_windows.hann(SP_FFT_SIZE, sym=False).astype(np.float32)


def process_frame(frame: np.ndarray, hann: np.ndarray) -> np.ndarray:
    """
    单帧处理 (对应 C 代码的 process_axis):
      1. 去直流偏移
      2. 去线性趋势
      3. Hann 窗
      4. FFT
      5. 提取幅度谱 (前 FREQ_BINS 个 bin), 转 dB
    """
    buf = frame.astype(np.float32).copy()
    buf -= buf.mean()
    slope = (buf[-1] - buf[0]) / (len(buf) - 1)
    buf -= slope * np.arange(len(buf), dtype=np.float32)
    buf *= hann
    spectrum = np.fft.rfft(buf, n=SP_FFT_SIZE)
    mag = np.abs(spectrum[:SP_FREQ_BINS])
    return 20 * np.log10(mag + 1e-10)


def sliding_window_fft(
    signal: np.ndarray, hann: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    滑动窗口 FFT (对应 C 代码的 compute_fft + sp_process_task 循环):
      - FFT_SIZE=1024, HOP=256
    返回:
      times: 每帧中心时刻 (秒)
      spectrogram: shape=(n_frames, FREQ_BINS)
    """
    n = len(signal)
    starts = list(range(0, n - SP_FFT_SIZE + 1, SP_HOP_SIZE))
    n_frames = len(starts)
    spectrogram = np.empty((n_frames, SP_FREQ_BINS), dtype=np.float32)
    times = np.empty(n_frames, dtype=np.float64)

    for idx, start in enumerate(starts):
        frame = signal[start: start + SP_FFT_SIZE]
        spectrogram[idx] = process_frame(frame, hann)
        times[idx] = (start + SP_FFT_SIZE / 2) / SP_SAMPLE_RATE

    return times, spectrogram


def process_3axis(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray
) -> dict:
    """
    处理 3 轴加速度, 返回完整结果字典:
      {
        "hann": ndarray,
        "x": (times, spectrogram),
        "y": (times, spectrogram),
        "z": (times, spectrogram),
        "freqs": ndarray,
        "sample_rate": float,
        "n_samples": int,
      }
    """
    hann = make_hann()
    print(f"FFT 参数: size={SP_FFT_SIZE}, hop={SP_HOP_SIZE}, "
          f"bins={SP_FREQ_BINS}, df={SP_FREQ_RES:.2f}Hz")

    result = {
        "hann": hann,
        "freqs": np.arange(SP_FREQ_BINS) * SP_FREQ_RES,
        "sample_rate": SP_SAMPLE_RATE,
        "n_samples": len(ax),
    }

    for label, sig in [("x", ax), ("y", ay), ("z", az)]:
        print(f"  处理 {label.upper()} 轴 ({len(sig)} 采样)...")
        result[label] = sliding_window_fft(sig, hann)

    return result
