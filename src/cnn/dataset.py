"""
数据集加载与增强

=== 本文件做了什么? ===
1. 从 .npz 文件加载 CNN 样本和标签
2. 对数据做标准化 (减均值除标准差)，让模型训练更容易
3. 划分训练集和验证集
4. 提供数据增强 (增加样本多样性，防止过拟合)
5. 创建 tf.data.Dataset 对象供训练使用

=== 什么是数据增强? ===
训练数据太少时，模型容易"死记硬背" (过拟合)。
数据增强通过对已有样本做微小变换 (偏移、加噪声、遮蔽)，
人为制造更多"看起来不同但本质一样"的训练样本。
例如: 同一张频谱图，向右偏移2帧，还是同一类 → 模型要学会识别"偏移后的"
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

from src.config import (
    BATCH_SIZE, VALIDATION_SPLIT, CNN_SAMPLE_FRAMES, SP_FREQ_BINS,
    CNN_FREQ_BINS, NUM_CHANNELS,
)


def load_npz(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """
    从 .npz 文件加载样本和标签。

    .npz 文件是 NumPy 的压缩格式，里面存了:
      - "samples": CNN 样本数据, shape=(N, 4, 16, 512)
      - "labels" (可选): 分类标签, shape=(N,), 值为 0,1,2,3...

    重要: 存储时用的格式是 (N, 4, 16, 512)，即 channels_first
          但 TensorFlow/Keras 默认用 channels_last 格式
          所以加载后需要转置为 (N, 16, 512, 4)

    Args:
        npz_path: .npz 文件的路径

    Returns:
        (samples, labels):
          - samples: shape=(N, 16, 512, 4), float32 — 转置后的样本
          - labels: shape=(N,), int32 — 标签 (如果没有则返回 None)

    === 转置说明 ===
    存储格式: (N, 4, 16, 512)  →  第1维是通道(X=0, Y=1, Z=2, magnitude=3)
    使用格式: (N, 16, 512, 4)  →  最后一维是通道 (TensorFlow 默认)

    np.transpose 的 (0, 2, 3, 1) 含义:
      维度0(N)     → 保持在位置0
      维度1(4)     → 移到位置3 (最后)
      维度2(16)    → 移到位置1
      维度3(512)   → 移到位置2
    """
    # 读取 npz 文件
    data = np.load(str(npz_path))

    # 取出样本数据
    samples = data["samples"]

    # 转置: (N, 4, 16, 512) → (N, 16, 512, 4)
    # 同时确保数据类型是 float32 (TensorFlow 的默认类型)
    samples = np.transpose(samples, (0, 2, 3, 1)).astype(np.float32)

    # 向后兼容: 旧版 .npz 存储 512 bin，裁剪到 CNN_FREQ_BINS
    if samples.shape[2] > CNN_FREQ_BINS:
        print(f"频率裁剪: {samples.shape[2]} → {CNN_FREQ_BINS} bin (向后兼容)")
        samples = samples[:, :, :CNN_FREQ_BINS, :]

    # 尝试读取标签 (可能没有)
    labels = None
    if "labels" in data:
        labels = data["labels"].astype(np.int32)

    print(f"加载样本: {samples.shape}, dtype={samples.dtype}")
    return samples, labels


def normalize(samples: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """
    标准化 (Standardization): 让数据分布更规整，加速模型训练。

    标准化公式: normalized = (x - mean) / std

    === 逐通道标准化 ===
    X/Y/Z/magnitude 四通道的数值范围可能不同 (比如 X 轴均值=8.6，Y 轴均值=4.1)
    所以每个通道单独计算 mean 和 std

    === 按文件划分时 ===
    如果传入 stats=None: 从 samples 计算 mean/std (用于训练集)
    如果传入 stats:   直接用传入的 mean/std (用于验证集/推理，防止数据泄漏)

    Args:
        samples: shape=(N, H, W, C)
        stats: 预计算的归一化参数 {"mean": [...], "std": [...]}，None 则自动计算

    Returns:
        (normalized, stats):
          - normalized: 标准化后的样本, shape 相同
          - stats: 归一化参数
    """
    if stats is None:
        mean = samples.mean(axis=(0, 1, 2), keepdims=True)
        std = samples.std(axis=(0, 1, 2), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        stats = {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}
        print(f"归一化 (从训练集计算): mean={stats['mean']}, std={stats['std']}")
    else:
        n_ch = samples.shape[-1]
        mean = np.array(stats["mean"], dtype=np.float32).reshape(1, 1, 1, n_ch)
        std = np.array(stats["std"], dtype=np.float32).reshape(1, 1, 1, n_ch)

    normalized = (samples - mean) / std
    return normalized, stats


def make_tf_dataset(
    samples: np.ndarray,
    labels: np.ndarray | None = None,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    """
    创建 tf.data.Dataset 对象 — TensorFlow 的高效数据加载管道。

    === 什么是 tf.data.Dataset? ===
    它是 TensorFlow 官方推荐的数据加载方式，好处:
    1. 自动打乱数据 (shuffle)
    2. 自动分批 (batch)
    3. 数据增强 (augment)
    4. 自动预取 (prefetch) — GPU 训练当前 batch 时，CPU 准备下一个 batch

    === batch 是什么? ===
    不是一次把所有样本都给模型，而是每次给 batch_size 个 (如16个)
    这样:
    - GPU 可以并行处理多个样本，速度快
    - 梯度更稳定 (多个样本的梯度取平均)

    Args:
        samples: shape=(N, 16, 512, 4) — 标准化后的样本
        labels: shape=(N,) — 标签 (None 则生成虚拟标签)
        batch_size: 每批多少个样本 (默认16)
        shuffle: 是否打乱顺序 (训练时打乱，验证时不打乱)
        augment: 是否做数据增强 (训练时增强，验证时不增强)

    Returns:
        tf.data.Dataset 对象，可以直接传给 model.fit() 使用
    """
    # 如果没有标签，生成全 0 的虚拟标签 (仅用于演示)
    if labels is None:
        labels = np.zeros(len(samples), dtype=np.int32)

    # 从 numpy 数组创建 Dataset
    # 每个元素是一个 (sample, label) 对
    ds = tf.data.Dataset.from_tensor_slices((samples, labels))

    # 打乱顺序: 每个 epoch (遍历完所有数据) 重新打乱一次
    # 为什么打乱? 防止模型学到数据的顺序 (比如先学idle再学normal)
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(samples),              # 打乱缓冲区大小
            reshuffle_each_iteration=True,         # 每个 epoch 重新打乱
        )

    # 数据增强: 只在训练时做 (验证时不做，保证评估结果可复现)
    if augment:
        # num_parallel_calls=AUTOTUNE 让 TensorFlow 自动决定用多少 CPU 核心
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    # 分批: 把连续的 batch_size 个样本打包成一个 batch
    # 例如 267 个样本, batch_size=16 → 17 个 batch (最后一个不满)
    ds = ds.batch(batch_size)

    # 预取: 提前准备下一个 batch 的数据
    # GPU 训练当前 batch 时，CPU 同时准备下一个 batch，提高效率
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def _augment(sample: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    数据增强函数 — 对单个样本做随机变换。

    === 为什么要做数据增强? ===
    我们只有 332 个样本，太少了！模型容易"死记硬背" (过拟合)。
    数据增强通过对样本做微小变换，人为增加训练数据的多样性。

    === 四种增强方式 ===

    1. 时间轴偏移 (±2 帧):
       频谱图在时间方向上左右移动一点
       原因: 同样的振动模式，采集时间可能略有不同

    2. 频率轴偏移 (±16 bin):
       频谱图在频率方向上上下移动一点
       原因: 同样的振动模式，频率可能略有偏移

    3. 高斯噪声:
       给每个像素加一点随机噪声
       原因: 真实传感器数据本身就有噪声

    4. SpecAugment (频率遮蔽):
       随机遮住连续 2~4 个频率 bin (设为0)
       原因: 增强模型对部分频率缺失的鲁棒性

    Args:
        sample: shape=(16, 512, 4) — 单个样本
        label: 标量 — 类别标签

    Returns:
        (augmented_sample, label) — 增强后的样本和原始标签
        注意: 标签不变！只是样本被变换，类别没变
    """
    # ── 增强 1: 时间轴随机偏移 ──────────────────────────────────────
    # 思路: 先在时间轴两端各补 2 帧 (padding)，然后随机裁剪回 16 帧
    # 这样相当于把频谱图在时间方向上随机移动了 0~4 帧
    #
    # pad 格式: [[上,下], [左,右], [前,后]] 对应 (帧, 频率, 通道) 三个维度
    # mode="REFLECT": 镜像填充 (比零填充更自然)
    sample = tf.pad(sample, [[2, 2], [0, 0], [0, 0]], mode="REFLECT")
    # 随机选一个裁剪起点 (0, 1, 2, 3 中选一个)
    offset = tf.random.uniform([], 0, 4, dtype=tf.int32)
    # 裁剪回原来的 16 帧
    sample = sample[offset:offset + CNN_SAMPLE_FRAMES]

    # ── 增强 2: 频率轴随机偏移 ──────────────────────────────────────
    # 和时间轴偏移同理，在频率方向两端各补 16 bin，然后随机裁剪
    sample = tf.pad(sample, [[0, 0], [16, 16], [0, 0]], mode="REFLECT")
    offset_f = tf.random.uniform([], 0, 32, dtype=tf.int32)
    sample = sample[:, offset_f:offset_f + CNN_FREQ_BINS]

    # ── 增强 3: 加高斯噪声 ──────────────────────────────────────────
    # 生成和 sample 同形状的随机噪声，标准差=0.1
    # 噪声很小，不会改变样本的类别，但能让模型更鲁棒
    noise = tf.random.normal(tf.shape(sample), stddev=0.1)
    sample = sample + noise

    # ── 增强 4: SpecAugment 频率遮蔽 ────────────────────────────────
    # 随机选一段连续的 2~4 个频率 bin，把它们全部设为 0
    # 效果: 模型要学会即使某些频率信息缺失也能正确分类

    # 随机决定遮蔽长度 (2, 3, 4 中选一个)
    mask_len = tf.random.uniform([], 2, 5, dtype=tf.int32)
    # 随机决定遮蔽起始位置
    mask_start = tf.random.uniform([], 0, CNN_FREQ_BINS - 4, dtype=tf.int32)

    # 创建全 1 的遮蔽矩阵 (16帧 × CNN_FREQ_BINS频率 × 4通道)
    mask = tf.ones((CNN_SAMPLE_FRAMES, CNN_FREQ_BINS, NUM_CHANNELS))
    # 创建遮蔽区域 (全 0)
    mask_update = tf.zeros((CNN_SAMPLE_FRAMES, mask_len, NUM_CHANNELS))

    # 计算遮蔽区域的索引
    # meshgrid 生成二维索引网格: (帧号, 频率号) 的所有组合
    indices_frame = tf.range(CNN_SAMPLE_FRAMES)           # [0, 1, ..., 15]
    indices_freq = tf.range(mask_start, mask_start + mask_len)  # 如 [100, 101, 102]
    idx = tf.stack(tf.meshgrid(indices_frame, indices_freq, indexing="ij"), axis=-1)
    idx = tf.reshape(idx, [-1, 2])  # 展平成 (帧数×遮蔽长度, 2) 的索引数组

    # 把遮蔽区域设为 0
    mask = tf.tensor_scatter_nd_update(
        mask, idx, tf.reshape(mask_update, [-1, NUM_CHANNELS])
    )

    # 应用遮蔽: 原始数据 × 遮蔽矩阵 (遮蔽区域变成 0)
    sample = sample * mask

    return sample, label


def train_val_split(
    samples: np.ndarray,
    labels: np.ndarray | None = None,
    val_ratio: float = VALIDATION_SPLIT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    划分训练集和验证集。

    === 为什么需要验证集? ===
    训练集: 用来训练模型 (模型看这些数据来学习)
    验证集: 用来评估模型效果 (模型不看这些数据，只用来测试)

    如果没有验证集，模型可能在训练集上准确率 100%，但遇到新数据就不行了
    (这就是过拟合)。验证集帮我们及时发现这个问题。

    === 分层抽样 ===
    如果类别分布不均匀 (比如 idle 有 200 个，loose 只有 30 个)，
    简单随机抽样可能导致验证集里某个类别一个都没有。
    分层抽样保证每个类别都按比例分到训练集和验证集。

    Args:
        samples: shape=(N, 16, 512, 4) — 全部样本
        labels: shape=(N,) — 标签 (None 则随机划分)
        val_ratio: 验证集比例 (默认 0.2 = 20%)

    Returns:
        (x_train, x_val, y_train, y_val):
          - x_train: 训练样本
          - x_val: 验证样本
          - y_train: 训练标签
          - y_val: 验证标签
    """
    n = len(samples)
    n_val = max(1, int(n * val_ratio))  # 至少 1 个验证样本

    if labels is not None:
        # ── 有标签: 分层抽样 ──────────────────────────────────────
        # 保证每个类别在训练集和验证集中的比例一致
        rng = np.random.RandomState(42)  # 固定随机种子，结果可复现
        val_idx = []

        # 对每个类别分别处理
        for cls in np.unique(labels):
            # 找到属于这个类别的所有样本索引
            cls_idx = np.where(labels == cls)[0]
            # 打乱顺序
            rng.shuffle(cls_idx)
            # 取前 val_ratio 比例的样本作为验证集
            n_cls_val = max(1, int(len(cls_idx) * val_ratio))
            val_idx.extend(cls_idx[:n_cls_val])

        val_idx = np.array(val_idx)
        # 剩下的作为训练集
        train_idx = np.setdiff1d(np.arange(n), val_idx)
    else:
        # ── 无标签: 简单随机划分 ──────────────────────────────────
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)  # 随机排列 0~n-1
        val_idx = idx[:n_val]     # 前 20% 做验证
        train_idx = idx[n_val:]   # 后 80% 做训练

    # 按索引取出数据
    x_train, x_val = samples[train_idx], samples[val_idx]
    y_train = labels[train_idx] if labels is not None else None
    y_val = labels[val_idx] if labels is not None else None

    print(f"训练集: {len(x_train)}  验证集: {len(x_val)}")
    return x_train, x_val, y_train, y_val


def train_val_split_by_file(
    file_data: list[tuple[np.ndarray, np.ndarray, str]],
    val_ratio: float = VALIDATION_SPLIT,
    seed: int = 42,
) -> tuple[list, list]:
    """
    按采集文件划分训练集和验证集 — 消除滑动窗口带来的数据泄漏。

    === 为什么按文件划分? ===
    当前数据: 7 个独立的采集文件，每个文件只含 1-2 个类别的工况。
    滑动窗口 stride=1 (75% 重叠) 导致相邻窗口几乎一模一样。
    如果随机切窗口 → 同一工况的窗口分散到 train/val → 验证指标虚高。

    按文件划分 = 每个采集文件作为一个整体归入 train 或 val，
    相邻窗口不会被拆分 → 验证集真正测试"未见过的工况"。

    === 划分策略 ===
    1. 确保 train/val 都覆盖所有存在的类别
    2. 每个类别: 最大文件 → train，最小文件 → val
    3. 剩余文件: 按大小分配到 train/val，使比例接近 val_ratio

    Args:
        file_data: [(samples, labels, filename), ...] — 每个元素是一个文件
        val_ratio: 验证集目标比例 (默认 0.2)
        seed: 随机种子 (固定保证可复现)

    Returns:
        (train_data, val_data): 各自的文件列表，格式同 file_data
    """
    rng = np.random.RandomState(seed)
    file_indices = rng.permutation(len(file_data))
    file_data_shuffled = [file_data[i] for i in file_indices]

    all_classes = set()
    for _, labels, _ in file_data_shuffled:
        all_classes.update(np.unique(labels))
    all_classes = sorted(all_classes)

    train_files: set[int] = set()
    val_files: set[int] = set()

    for cls in all_classes:
        candidates = [
            (i, d) for i, d in enumerate(file_data_shuffled)
            if cls in np.unique(d[1])
        ]
        candidates.sort(key=lambda x: len(x[1][0]), reverse=True)

        train_covered = any(
            cls in np.unique(file_data_shuffled[t][1]) for t in train_files
        )
        val_covered = any(
            cls in np.unique(file_data_shuffled[v][1]) for v in val_files
        )

        if train_covered and val_covered:
            continue

        unassigned = [
            c for c in candidates
            if c[0] not in train_files and c[0] not in val_files
        ]

        if not train_covered and unassigned:
            train_files.add(unassigned[0][0])
            unassigned = [c for c in unassigned if c[0] != unassigned[0][0]]

        if not val_covered and unassigned:
            val_files.add(unassigned[-1][0])

    remaining = set(range(len(file_data_shuffled))) - train_files - val_files
    remaining_sorted = sorted(
        remaining, key=lambda i: len(file_data_shuffled[i][0]), reverse=True,
    )

    total_samples = sum(len(d[0]) for d in file_data_shuffled)
    val_target = int(total_samples * val_ratio)

    for idx in remaining_sorted:
        current_val = sum(len(file_data_shuffled[v][0]) for v in val_files)
        if current_val + len(file_data_shuffled[idx][0]) <= val_target * 1.3:
            val_files.add(idx)
        else:
            train_files.add(idx)

    train_data = [file_data_shuffled[i] for i in sorted(train_files)]
    val_data = [file_data_shuffled[i] for i in sorted(val_files)]

    n_train = sum(len(d[0]) for d in train_data)
    n_val = sum(len(d[0]) for d in val_data)
    print(f"\n按文件划分:")
    print(f"  Train ({n_train} 样本, {100*n_train/total_samples:.1f}%):")
    for _, _, fname in train_data:
        print(f"    {fname}")
    print(f"  Val   ({n_val} 样本, {100*n_val/total_samples:.1f}%):")
    for _, _, fname in val_data:
        print(f"    {fname}")

    return train_data, val_data


def classification_report_from_cm(
    cm: np.ndarray,
    class_names: list[str],
) -> str:
    """
    从混淆矩阵计算分类报告 — precision, recall, f1-score。

    不依赖 sklearn，直接基于混淆矩阵手算。

    Args:
        cm: 混淆矩阵, shape=(n_classes, n_classes)
        class_names: 类别名称列表

    Returns:
        格式化的分类报告字符串
    """
    n_classes = len(class_names)
    lines = []
    lines.append("")
    lines.append(f"{'':>12}  precision  recall    f1-score  support")
    lines.append("-" * 58)

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        lines.append(
            f"{class_names[i]:>12}  {precision:.4f}    {recall:.4f}    "
            f"{f1:.4f}    {support}"
        )

    # Weighted average
    total = cm.sum()
    if total > 0:
        w_prec = sum(
            (cm[i, i] / (cm[:, i].sum()) if cm[:, i].sum() > 0 else 0)
            * (cm[i, :].sum()) for i in range(n_classes)
        ) / total
        w_rec = sum(cm[i, i] for i in range(n_classes)) / total
        w_f1 = (2 * w_prec * w_rec / (w_prec + w_rec)
                if (w_prec + w_rec) > 0 else 0)
    else:
        w_prec = w_rec = w_f1 = 0.0

    lines.append("-" * 58)
    lines.append(
        f"{'weighted avg':>12}  {w_prec:.4f}    {w_rec:.4f}    "
        f"{w_f1:.4f}    {total}"
    )

    report = "\n".join(lines)
    print(report)
    return report


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
) -> dict[int, float]:
    """
    计算类别权重 — 解决类别不平衡问题。

    使用 "inverse frequency" 方法:
      weight_c = n_samples / (n_classes * n_samples_c)

    效果: 样本少的类别权重高，样本多的类别权重低。
    例如 idle 有 200 个样本，loose 只有 30 个:
      weight_idle = 332 / (4 * 200) = 0.415
      weight_loose = 332 / (4 * 30) = 2.767

    训练时 loss 会乘以类别权重，让模型更关注少数类。

    Args:
        labels: shape=(N,) — 训练集标签
        num_classes: 类别总数

    Returns:
        {class_idx: weight} 字典，可直接传给 model.fit(class_weight=...)
    """
    n_samples = len(labels)
    weights = {}
    for c in range(num_classes):
        count = (labels == c).sum()
        if count > 0:
            weights[c] = n_samples / (num_classes * count)
        else:
            weights[c] = 1.0

    print(f"类别权重: {', '.join(f'{k}={v:.2f}' for k, v in weights.items())}")
    return weights
