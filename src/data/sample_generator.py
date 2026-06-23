"""
从 FFT 频谱图生成 CNN 训练样本
输出形状: (4, n_frames, freq_bins) 即 (channels, frames, freq_bins)
通道: X/Y/Z + magnitude (sqrt(x²+y²+z²))
"""

import json
import numpy as np
from pathlib import Path

from src.config import (
    CNN_SAMPLE_FRAMES, SP_FREQ_BINS, CNN_FREQ_BINS, SP_HOP_SIZE, SP_SAMPLE_RATE,
    SP_FFT_SIZE, SP_FREQ_RES, CLASS_NAMES, NUM_CHANNELS,
)


def extract_sample(
    spectrograms: dict[str, np.ndarray],
    start_frame: int,
    n_frames: int = CNN_SAMPLE_FRAMES,
    n_freq_bins: int = CNN_FREQ_BINS,
) -> np.ndarray:
    """
    从 4 通道频谱图中提取一个 CNN 样本。
    spectrograms: {"x": (n_total, 512), "y": ..., "z": ..., "magnitude": ...}
    n_freq_bins: 保留的频率 bin 数 (默认 CNN_FREQ_BINS=128，裁剪低频段)
    返回: shape=(4, n_frames, n_freq_bins), float32
    """
    end = start_frame + n_frames
    return np.stack([
        spectrograms["x"][start_frame:end, :n_freq_bins],
        spectrograms["y"][start_frame:end, :n_freq_bins],
        spectrograms["z"][start_frame:end, :n_freq_bins],
        spectrograms["magnitude"][start_frame:end, :n_freq_bins],
    ], axis=0).astype(np.float32)


def generate_samples(
    spectrograms: dict[str, np.ndarray],
    n_frames: int = CNN_SAMPLE_FRAMES,
    stride: int = 1,
    n_freq_bins: int = CNN_FREQ_BINS,
) -> np.ndarray:
    """
    滑动窗口生成所有 CNN 样本。
    spectrograms: {"x": (n_total, 512), "y": ..., "z": ..., "magnitude": ...}
    stride: 帧步长 (1 = 逐帧滑动)
    n_freq_bins: 保留的频率 bin 数 (默认 CNN_FREQ_BINS=128)
    返回: shape=(N, 4, n_frames, n_freq_bins), float32
    """
    n_total = spectrograms["x"].shape[0]
    if n_total < n_frames:
        raise ValueError(f"总帧数 {n_total} < 样本帧数 {n_frames}")

    starts = list(range(0, n_total - n_frames + 1, stride))
    n_samples = len(starts)
    samples = np.empty((n_samples, NUM_CHANNELS, n_frames, n_freq_bins), dtype=np.float32)

    for i, s in enumerate(starts):
        samples[i] = extract_sample(spectrograms, s, n_frames, n_freq_bins)

    print(f"生成样本: {n_samples} 个, shape={samples.shape}, "
          f"dtype={samples.dtype}, range=[{samples.min():.1f}, {samples.max():.1f}]")
    print(f"频率裁剪: {SP_FREQ_BINS} → {n_freq_bins} bin "
          f"(0-{n_freq_bins * SP_FREQ_RES:.0f} Hz)")
    return samples


def generate_labels(
    n_samples: int,
    label_config: dict,
) -> np.ndarray:
    """
    根据配置文件生成标签数组。

    CNN 样本 i 的中心时间:
      center_time(i) = ((i + 7.5) * HOP_SIZE + FFT_SIZE / 2) / SAMPLE_RATE

    Args:
        n_samples: CNN 样本总数
        label_config: 从 JSON 解析的配置字典，格式:
            {
              "labels": [
                {"start": 0.0, "end": 5.0, "class": "idle"},
                ...
              ]
            }

    Returns:
        labels: shape=(N,), int32 — 每个样本对应的类别索引，未匹配的为 -1
    """
    segments = label_config.get("labels", [])
    # 预处理: 将 class 名转为索引
    seg_idx = []
    for seg in segments:
        cls_name = seg["class"]
        if cls_name not in CLASS_NAMES:
            print(f"警告: 未知类别 '{cls_name}'，跳过该时间段 [{seg['start']}, {seg['end']}]")
            continue
        seg_idx.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "cls": CLASS_NAMES.index(cls_name),
        })

    labels = np.full(n_samples, -1, dtype=np.int32)
    matched = 0

    for i in range(n_samples):
        # CNN 样本 i 的中心时间
        t = ((i + (CNN_SAMPLE_FRAMES - 1) / 2) * SP_HOP_SIZE + SP_FFT_SIZE / 2) / SP_SAMPLE_RATE
        for seg in seg_idx:
            if seg["start"] <= t < seg["end"]:
                labels[i] = seg["cls"]
                matched += 1
                break

    print(f"标签生成: {n_samples} 个样本, 匹配 {matched}, 无标签 {n_samples - matched}")
    # 统计各类别数量
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        count = (labels == cls_idx).sum()
        if count > 0:
            print(f"  {cls_name}: {count}")
    return labels


def filter_labeled(
    samples: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """过滤掉没有标签 (label=-1) 的样本"""
    mask = labels >= 0
    n_before = len(labels)
    n_after = mask.sum()
    n_dropped = n_before - n_after
    if n_dropped > 0:
        print(f"过滤: 丢弃 {n_dropped} 个无标签样本, 保留 {n_after}")
    return samples[mask], labels[mask]


def save_samples(
    samples: np.ndarray,
    out_path: str | Path,
    labels: np.ndarray | None = None,
) -> None:
    """保存样本到 .npz，可选附带标签"""
    if labels is not None:
        np.savez_compressed(str(out_path), samples=samples, labels=labels)
    else:
        np.savez_compressed(str(out_path), samples=samples)
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"已保存: {out_path} ({size_mb:.1f} MB)")


def load_samples(npz_path: str | Path) -> np.ndarray:
    """从 .npz 加载样本"""
    data = np.load(str(npz_path))
    return data["samples"]
