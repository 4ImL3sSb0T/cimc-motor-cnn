"""
从 FFT 频谱图生成 CNN 训练样本
输出形状: (3, n_frames, freq_bins) 即 (channels, frames, freq_bins)
"""

import numpy as np
from pathlib import Path

from src.config import CNN_SAMPLE_FRAMES, SP_FREQ_BINS, SP_HOP_SIZE, SP_SAMPLE_RATE


def extract_sample(
    spectrograms: dict[str, np.ndarray],
    start_frame: int,
    n_frames: int = CNN_SAMPLE_FRAMES,
) -> np.ndarray:
    """
    从 3 轴频谱图中提取一个 CNN 样本。
    spectrograms: {"x": (n_total, 512), "y": ..., "z": ...}
    返回: shape=(3, n_frames, 512), float32
    """
    end = start_frame + n_frames
    return np.stack([
        spectrograms["x"][start_frame:end],
        spectrograms["y"][start_frame:end],
        spectrograms["z"][start_frame:end],
    ], axis=0).astype(np.float32)


def generate_samples(
    spectrograms: dict[str, np.ndarray],
    n_frames: int = CNN_SAMPLE_FRAMES,
    stride: int = 1,
) -> np.ndarray:
    """
    滑动窗口生成所有 CNN 样本。
    spectrograms: {"x": (n_total, 512), ...}
    stride: 帧步长 (1 = 逐帧滑动)
    返回: shape=(N, 3, n_frames, 512), float32
    """
    n_total = spectrograms["x"].shape[0]
    if n_total < n_frames:
        raise ValueError(f"总帧数 {n_total} < 样本帧数 {n_frames}")

    starts = list(range(0, n_total - n_frames + 1, stride))
    n_samples = len(starts)
    samples = np.empty((n_samples, 3, n_frames, SP_FREQ_BINS), dtype=np.float32)

    for i, s in enumerate(starts):
        samples[i] = extract_sample(spectrograms, s, n_frames)

    print(f"生成样本: {n_samples} 个, shape={samples.shape}, "
          f"dtype={samples.dtype}, range=[{samples.min():.1f}, {samples.max():.1f}]")
    return samples


def save_samples(samples: np.ndarray, out_path: str | Path) -> None:
    """保存样本到 .npz"""
    np.savez_compressed(str(out_path), samples=samples)
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"已保存: {out_path} ({size_mb:.1f} MB)")


def load_samples(npz_path: str | Path) -> np.ndarray:
    """从 .npz 加载样本"""
    data = np.load(str(npz_path))
    return data["samples"]
