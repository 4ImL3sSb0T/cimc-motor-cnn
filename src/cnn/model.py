"""
CNN 模型定义 — 用于 IMU 振动频谱图分类

=== 什么是 CNN? ===
CNN (Convolutional Neural Network，卷积神经网络) 是一种专门用来识别"图像-like"数据的神经网络。
我们的频谱图 (16帧 × 512频率bin × 4通道) 就是一种"图像"，所以用 CNN 来分类。

=== 模型做了什么? ===
输入一张频谱图 → 经过层层卷积提取特征 → 最终输出每个类别的概率
例如: 输入一张振动频谱图，输出 [0.05, 0.90, 0.03, 0.02]
      表示: 5%概率是idle, 90%概率是normal, 3%是loose, 2%是imbalance
      → 所以判断为 "normal"

=== 输入输出 ===
输入: (batch, 16, 512, 4)  — batch张图片, 每张16帧高×512频率宽×4通道(X/Y/Z/magnitude)
输出: (batch, num_classes)  — 每张图片对应 num_classes 个类别的概率

=== 设计目标 ===
- 轻量级: 总参数 ~36,000 个 (~144KB float32, ~36KB int8)，可装入 ESP32-S3
- 使用深度可分离卷积: 比普通卷积省参数
- SE 注意力机制: 自适应突出重要频率通道
- 残差连接: 改善梯度流，防止 SeparableConv 表达力不足导致的退化
- 非对称卷积核: 首层 1×7 沿频率方向提取宽谱谐波结构
- 温和频率压缩: 512→256→128→64 (vs 旧版 512→128→32→8)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import CNN_SAMPLE_FRAMES, CNN_FREQ_BINS, NUM_CLASSES, NUM_CHANNELS


# ═══════════════════════════════════════════════════════════════════════════
# 损失函数: Focal Loss
# ═══════════════════════════════════════════════════════════════════════════

def focal_loss(gamma: float = 2.0, alpha: float = 1.0):
    """
    Focal Loss — 解决类别不平衡的损失函数。

    标准交叉熵对所有样本一视同仁，但数据集中某些类别样本多 (如 idle)，
    某些类别样本少 (如 imbalance)。模型容易偏向多数类。

    Focal Loss 给"容易分类"的样本降低权重，让模型更关注"难分类"的样本:
      FL = -alpha * (1 - p_t)^gamma * log(p_t)

    其中 p_t 是模型对真实类别的预测概率:
      - p_t 高 (模型很确定) → (1-p_t)^gamma 小 → loss 贡献小
      - p_t 低 (模型不确定) → (1-p_t)^gamma 大 → loss 贡献大

    Args:
        gamma: 聚焦参数 (默认 2.0)。越大，对难样本的关注越强
        alpha: 平衡因子 (默认 1.0)。可与 class_weight 配合使用

    Returns:
        损失函数，可直接传给 model.compile(loss=...)
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
        fl = -alpha * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(fl)
    return loss


# ═══════════════════════════════════════════════════════════════════════════
# 工具层: SE 注意力
# ═══════════════════════════════════════════════════════════════════════════

def _se_block(x, reduction: int = 4, name: str | None = None):
    """
    Squeeze-and-Excitation 通道注意力模块。

    振动频谱中不同频率段的重要性差异巨大:
      - 0-50 Hz: 直流漂移残余, 不重要
      - 50-300 Hz: 电机故障特征频段, 极重要
      - 1000+ Hz: 高频噪声, 不重要

    SE 让模型自动学会给重要频段对应的通道分配高权重。

    Args:
        x: 输入张量 [H, W, C]
        reduction: 压缩比 (C → C/r → C)
        name: 层名前缀
    """
    prefix = f"{name}_" if name else ""
    filters = x.shape[-1]
    squeeze_dim = max(1, filters // reduction)

    se = layers.GlobalAveragePooling2D(name=f"{prefix}se_gap")(x)
    se = layers.Dense(squeeze_dim, activation="relu",
                       name=f"{prefix}se_squeeze")(se)
    se = layers.Dense(filters, activation="sigmoid",
                       name=f"{prefix}se_excite")(se)
    se = layers.Reshape((1, 1, filters), name=f"{prefix}se_reshape")(se)
    return layers.Multiply(name=f"{prefix}se_scale")([x, se])


# ═══════════════════════════════════════════════════════════════════════════
# 工具层: 残差卷积块
# ═══════════════════════════════════════════════════════════════════════════

def _residual_block(
    x,
    filters: int,
    kernel_size: int | tuple = 3,
    pool_size: int | tuple | None = None,
    use_se: bool = True,
    se_reduction: int = 4,
    spatial_dropout: float = 0.0,
    name: str | None = None,
):
    """
    带残差连接的卷积块 (类似 MobileNetV2 风格)。

    主路径: SepConv2D → BN → ReLU → [SpatialDropout] → [SE] → [MaxPool]
    短路:   Conv2D(1×1) → BN → [MaxPool]     (通道数不匹配时做投影)

    Add + ReLU 结束。

    残差连接的意义: SeparableConv 参数少 (表达力弱于标准 Conv)，
    残差给梯度一条"高速公路"，即使某个 separable conv 学不到有用特征，
    shortcut 也能把信息直送深层，防止训练退化。

    SpatialDropout2D: 随机丢弃整个特征图 (而非单个像素)。
    对于 CNN 来说，普通 dropout 只丢弃单个像素效果有限，
    因为相邻像素高度相关。SpatialDropout 强迫模型不依赖某几个特定通道。

    Args:
        x: 输入张量 [H, W, Cin]
        filters: 输出通道数
        kernel_size: 卷积核大小 (默认 3)
        pool_size: 池化核大小 (None=不池化)
        use_se: 是否加 SE 注意力
        se_reduction: SE 压缩比
        spatial_dropout: SpatialDropout2D 比率 (0=不使用)
        name: 层名前缀
    """
    prefix = f"{name}_" if name else ""
    in_filters = x.shape[-1]

    # ── 主路径 ──
    # 深度可分离卷积: depthwise(3×3) + pointwise(1×1)
    main = layers.SeparableConv2D(
        filters, kernel_size,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
        name=f"{prefix}sepconv",
    )(x)
    main = layers.BatchNormalization(name=f"{prefix}bn")(main)
    main = layers.ReLU(name=f"{prefix}relu")(main)

    # SpatialDropout2D: 随机丢弃整个特征图 (Block 2, 3 使用)
    if spatial_dropout > 0:
        main = layers.SpatialDropout2D(
            spatial_dropout, name=f"{prefix}spatial_drop",
        )(main)

    # SE 通道注意力 (Block 2, 3 使用; Block 4 不用)
    if use_se:
        main = _se_block(main, reduction=se_reduction, name=f"{prefix}se")

    # 池化 (Block 1-3 使用; Block 4 不用)
    if pool_size is not None:
        main = layers.MaxPool2D(
            pool_size=pool_size, name=f"{prefix}pool",
        )(main)

    # ── 短路路径 ──
    shortcut = x
    # 通道数不匹配 → 1×1 卷积投影
    if in_filters != filters:
        shortcut = layers.Conv2D(
            filters, 1, padding="same", use_bias=False,
            kernel_initializer="he_normal",
            name=f"{prefix}skip_proj",
        )(shortcut)
    shortcut = layers.BatchNormalization(name=f"{prefix}skip_bn")(shortcut)

    # 短路也需要同样的空间降采样
    if pool_size is not None:
        shortcut = layers.MaxPool2D(
            pool_size=pool_size, name=f"{prefix}skip_pool",
        )(shortcut)

    # ── 融合 ──
    out = layers.Add(name=f"{prefix}add")([main, shortcut])
    out = layers.ReLU(name=f"{prefix}out_relu")(out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 模型构建
# ═══════════════════════════════════════════════════════════════════════════

def build_model(
    input_shape: tuple = (CNN_SAMPLE_FRAMES, CNN_FREQ_BINS, NUM_CHANNELS),
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """
    构建 CNN 模型 (v2: 残差 + SE 注意力 + 频率裁剪 + SpatialDropout)。

    架构概述:

        Input [16, 128, 4]
          │
          ▼
        ┌─────────────────────────────────────────────────────┐
        │ Block 1: Conv2D(20, 1×7) + MaxPool(3,2)             │
        │   频率方向大 kernel (1×7 ~46Hz) 提取宽谱谐波结构      │
        │   [16, 128, 4] → [6, 64, 20]                         │
        │   频率: 128→64 (÷2)  时间: 16→6 (÷3)                │
        ├─────────────────────────────────────────────────────┤
        │ Block 2: SepConv(40) + SpDrop(0.1) + SE + Pool(2,2)  │
        │   [6, 64, 20] → [3, 32, 40]                          │
        │   频率: 64→32 (÷2)  时间: 6→3 (÷2)                  │
        ├─────────────────────────────────────────────────────┤
        │ Block 3: SepConv(80) + SpDrop(0.1) + SE + Pool(2,2)  │
        │   [3, 32, 40] → [2, 16, 80]                          │
        │   频率: 32→16 (÷2)  时间: 3→2 (÷2)                  │
        ├─────────────────────────────────────────────────────┤
        │ Block 4: SepConv(96) [+残差]   (无池化, 无 SE)       │
        │   [2, 16, 80] → [2, 16, 96]                          │
        ├─────────────────────────────────────────────────────┤
        │ GAP → [96] → Drop(0.25) → Dense(48) → Drop(0.3)     │
        │ → Dense(num_classes, softmax)                        │
        └─────────────────────────────────────────────────────┘
          │
          ▼
        输出 [num_classes] — 每个类别的概率

    关键改进 (vs v1):
      - 频率裁剪: 128 bin (0-833 Hz) 代替 512 bin，减少 75% 计算量
      - 频率分辨率: 最终 16 bin (52 Hz/bin) vs 旧版 8 bin (417 Hz/bin)
      - SE 注意力: Block 2/3 后自适应强调重要频率通道
      - 残差连接: 每个 SeparableConv 块有 shortcut，改善梯度流
      - SpatialDropout2D: Block 2/3 后随机丢弃整个特征图，防止过拟合
      - 1×7 首层 kernel: 沿频率方向提取宽谱谐波特征
      - 通道数增长: 20→40→80→96，更丰富的特征表示

    Args:
        input_shape: (frames, freq_bins, channels)，默认 (16, 128, 4)
        num_classes: 分类类别数

    Returns:
        未编译的 Keras Model
    """
    inputs = keras.Input(shape=input_shape, name="spectrogram")

    # ═══════════════════════════════════════════════════════════════
    # Block 1: 首层 — 频率方向宽 kernel，提取谐波结构
    # ═══════════════════════════════════════════════════════════════
    # 为什么用 1×7 而不是 3×3?
    #   振动故障的频谱特征体现为多个谐波的组合 (如 50Hz 的 1×, 2×, 3×...)
    #   1×7 覆盖 ~46 Hz 的频率范围，刚好能看全一组相邻谐波
    #   不混合时间维度，时间信息留给后面的 3×3 separable conv
    #
    # 为什么用 MaxPool(3,2)?
    #   频率: ÷2 (温和压缩, 保留频率细节)
    #   时间: ÷3 (激进压缩, 16帧→6帧, 短时平稳所以信息冗余)
    x = layers.Conv2D(
        20, (1, 7),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name="b1_conv",
    )(inputs)
    x = layers.BatchNormalization(name="b1_bn")(x)
    x = layers.ReLU(name="b1_relu")(x)
    x = layers.MaxPool2D(pool_size=(3, 2), name="b1_pool")(x)
    # → [6, 64, 20]

    # ═══════════════════════════════════════════════════════════════
    # Block 2: 残差块 + SpatialDropout + SE 注意力
    # ═══════════════════════════════════════════════════════════════
    x = _residual_block(
        x, filters=40, kernel_size=3, pool_size=(2, 2),
        use_se=True, se_reduction=4, spatial_dropout=0.1, name="b2",
    )
    # → [3, 32, 40]

    # ═══════════════════════════════════════════════════════════════
    # Block 3: 残差块 + SpatialDropout + SE 注意力
    # ═══════════════════════════════════════════════════════════════
    x = _residual_block(
        x, filters=80, kernel_size=3, pool_size=(2, 2),
        use_se=True, se_reduction=4, spatial_dropout=0.1, name="b3",
    )
    # → [2, 16, 80]

    # ═══════════════════════════════════════════════════════════════
    # Block 4: 残差块 (无池化, 无 SE — 保持空间结构原样送 GAP)
    # ═══════════════════════════════════════════════════════════════
    x = _residual_block(
        x, filters=96, kernel_size=3, pool_size=None,
        use_se=False, name="b4",
    )
    # → [2, 16, 96]

    # ═══════════════════════════════════════════════════════════════
    # 分类头
    # ═══════════════════════════════════════════════════════════════
    # GlobalAveragePooling: 每个通道的 2×16 特征图 → 1 个数字
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    # → [96]

    x = layers.Dropout(0.25, name="drop1")(x)
    x = layers.Dense(
        48, activation="relu",
        kernel_initializer="he_normal",
        name="fc1",
    )(x)
    # → [48]

    x = layers.Dropout(0.3, name="drop2")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax",
        name="class_prob",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="imu_cnn_v2")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 旧版模型 (v1) — 保留以做对比
# ═══════════════════════════════════════════════════════════════════════════

def build_model_v1(
    input_shape: tuple = (CNN_SAMPLE_FRAMES, CNN_FREQ_BINS, NUM_CHANNELS),
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """
    旧版 CNN 模型 (v1): 纯顺序结构, 无残差, 无 SE, 频率被过度压缩。

    架构:
        Conv2D(16,3×3) → Pool(2,4) → SepConv(32) → Pool(2,4)
        → SepConv(64) → Pool(2,4) → SepConv(64) → GAP → Dense(32) → Dense(4)

    保留此函数用于与 v2 对比实验。
    """
    inputs = keras.Input(shape=input_shape, name="spectrogram")

    # Block 1
    x = layers.Conv2D(16, 3, padding="same", use_bias=False,
                       kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 4))(x)

    # Block 2
    x = layers.SeparableConv2D(32, 3, padding="same", use_bias=False,
                                depthwise_initializer="he_normal",
                                pointwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 4))(x)

    # Block 3
    x = layers.SeparableConv2D(64, 3, padding="same", use_bias=False,
                                depthwise_initializer="he_normal",
                                pointwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 4))(x)

    # Block 4
    x = layers.SeparableConv2D(64, 3, padding="same", use_bias=False,
                                depthwise_initializer="he_normal",
                                pointwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_prob")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="imu_cnn_v1")


# ═══════════════════════════════════════════════════════════════════════════
# 模型编译
# ═══════════════════════════════════════════════════════════════════════════

def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    use_focal_loss: bool = False,
) -> keras.Model:
    """
    编译模型 — 配置优化器、损失函数、评估指标。

    Args:
        model: 未编译的 Keras 模型
        learning_rate: 学习率 (默认 0.001)
        use_focal_loss: 是否使用 Focal Loss (默认 False，使用标准交叉熵)
            Focal Loss 对类别不平衡更鲁棒，但训练初期可能不稳定

    Returns:
        编译好的模型
    """
    loss_fn = focal_loss(gamma=2.0) if use_focal_loss else "sparse_categorical_crossentropy"
    loss_name = "focal_loss" if use_focal_loss else "sparse_categorical_crossentropy"
    print(f"损失函数: {loss_name}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 直接运行查看模型结构
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("IMU CNN v2 (残差 + SE 注意力 + 温和频率压缩)")
    print("=" * 70)
    model_v2 = build_model()
    model_v2.summary()

    print("\n" + "=" * 70)
    print("IMU CNN v1 (旧版, 用于对比)")
    print("=" * 70)
    model_v1 = build_model_v1()
    model_v1.summary()
