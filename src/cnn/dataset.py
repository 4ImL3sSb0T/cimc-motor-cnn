"""
数据集加载与增强
从 .npz 文件加载 CNN 样本，进行训练/验证划分和数据增强
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

from src.config import (
    BATCH_SIZE, VALIDATION_SPLIT, CNN_SAMPLE_FRAMES, SP_FREQ_BINS,
)


def load_npz(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    加载 .npz 样本文件。
    返回 (samples, labels):
      samples: shape=(N, 3, 16, 512) → 转置为 (N, 16, 512, 3) channels_last
      labels: 若 npz 中有 'labels' 则返回，否则返回 None
    """
    data = np.load(str(npz_path))
    samples = data["samples"]
    # (N, 3, 16, 512) → (N, 16, 512, 3)  channels_last
    samples = np.transpose(samples, (0, 2, 3, 1)).astype(np.float32)

    labels = None
    if "labels" in data:
        labels = data["labels"].astype(np.int32)

    print(f"加载样本: {samples.shape}, dtype={samples.dtype}")
    return samples, labels


def normalize(samples: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    标准化: 减均值除标准差，逐通道统计。
    返回 (normalized, stats) 以便推理时复用相同归一化参数。
    """
    # samples: (N, 16, 512, 3)
    mean = samples.mean(axis=(0, 1, 2), keepdims=True)  # (1,1,1,3)
    std = samples.std(axis=(0, 1, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)  # 防止除零
    normalized = (samples - mean) / std
    stats = {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}
    print(f"归一化: mean={stats['mean']}, std={stats['std']}")
    return normalized, stats


def make_tf_dataset(
    samples: np.ndarray,
    labels: np.ndarray | None = None,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    """
    创建 tf.data.Dataset，支持数据增强。
    若 labels 为 None，则生成虚拟标签 (用于无标签数据的自监督场景)。
    """
    if labels is None:
        labels = np.zeros(len(samples), dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((samples, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(samples), reshuffle_each_iteration=True)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _augment(sample: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    数据增强 (仅训练时使用):
      - 时间轴随机偏移 (±2 帧)
      - 频率轴随机偏移 (±16 bin)
      - 加高斯噪声
      - 随机频率遮蔽 (SpecAugment)
    """
    # 时间轴偏移: 在 frame 维度 pad 后随机裁剪回原尺寸
    sample = tf.pad(sample, [[2, 2], [0, 0], [0, 0]], mode="REFLECT")
    offset = tf.random.uniform([], 0, 4, dtype=tf.int32)
    sample = sample[offset:offset + CNN_SAMPLE_FRAMES]

    # 频率轴偏移
    sample = tf.pad(sample, [[0, 0], [16, 16], [0, 0]], mode="REFLECT")
    offset_f = tf.random.uniform([], 0, 32, dtype=tf.int32)
    sample = sample[:, offset_f:offset_f + SP_FREQ_BINS]

    # 加高斯噪声
    noise = tf.random.normal(tf.shape(sample), stddev=0.1)
    sample = sample + noise

    # SpecAugment: 随机遮蔽连续 2~4 个频率 bin
    mask_len = tf.random.uniform([], 2, 5, dtype=tf.int32)
    mask_start = tf.random.uniform([], 0, SP_FREQ_BINS - 4, dtype=tf.int32)
    mask = tf.ones((CNN_SAMPLE_FRAMES, SP_FREQ_BINS, 3))
    mask_update = tf.zeros((CNN_SAMPLE_FRAMES, mask_len, 3))
    indices_frame = tf.range(CNN_SAMPLE_FRAMES)
    indices_freq = tf.range(mask_start, mask_start + mask_len)
    idx = tf.stack(tf.meshgrid(indices_frame, indices_freq, indexing="ij"), axis=-1)
    idx = tf.reshape(idx, [-1, 2])
    mask = tf.tensor_scatter_nd_update(
        mask, idx, tf.reshape(mask_update, [-1, 3])
    )
    sample = sample * mask

    return sample, label


def train_val_split(
    samples: np.ndarray,
    labels: np.ndarray | None = None,
    val_ratio: float = VALIDATION_SPLIT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """按比例划分训练/验证集 (保持类别比例)"""
    n = len(samples)
    n_val = max(1, int(n * val_ratio))

    if labels is not None:
        # 分层抽样
        rng = np.random.RandomState(42)
        val_idx = []
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            n_cls_val = max(1, int(len(cls_idx) * val_ratio))
            val_idx.extend(cls_idx[:n_cls_val])
        val_idx = np.array(val_idx)
        train_idx = np.setdiff1d(np.arange(n), val_idx)
    else:
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

    x_train, x_val = samples[train_idx], samples[val_idx]
    y_train = labels[train_idx] if labels is not None else None
    y_val = labels[val_idx] if labels is not None else None

    print(f"训练集: {len(x_train)}  验证集: {len(x_val)}")
    return x_train, x_val, y_train, y_val
