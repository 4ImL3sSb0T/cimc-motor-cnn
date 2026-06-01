"""
CNN 模型定义 — 用于 IMU 振动频谱图分类

=== 什么是 CNN? ===
CNN (Convolutional Neural Network，卷积神经网络) 是一种专门用来识别"图像-like"数据的神经网络。
我们的频谱图 (16帧 × 512频率bin × 3通道) 就是一种"图像"，所以用 CNN 来分类。

=== 模型做了什么? ===
输入一张频谱图 → 经过层层卷积提取特征 → 最终输出每个类别的概率
例如: 输入一张振动频谱图，输出 [0.05, 0.90, 0.03, 0.02]
      表示: 5%概率是idle, 90%概率是normal, 3%是loose, 2%是imbalance
      → 所以判断为 "normal"

=== 输入输出 ===
输入: (batch, 16, 512, 3)  — batch张图片, 每张16帧高×512频率宽×3通道(X/Y/Z)
输出: (batch, num_classes)  — 每张图片对应 num_classes 个类别的概率

=== 设计目标 ===
- 轻量级: 总参数只有 11,012 个 (43 KB)，可以装进 ESP32-S3
- 使用深度可分离卷积: 比普通卷积省很多参数
- 全局平均池化: 替代大尺寸全连接层，进一步压缩模型
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import CNN_SAMPLE_FRAMES, SP_FREQ_BINS, NUM_CLASSES


def build_model(
    input_shape: tuple = (CNN_SAMPLE_FRAMES, SP_FREQ_BINS, 3),
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """
    构建 CNN 模型。

    Args:
        input_shape: (frames, freq_bins, channels)，即 (16, 512, 3)
            - frames=16: 时间维度，16个时间帧
            - freq_bins=512: 频率维度，512个频率bin
            - channels=3: X/Y/Z 三轴加速度
        num_classes: 分类类别数，比如4类(idle/normal/loose/imbalance)

    Returns:
        编译好的 Keras Model，可以直接用来训练或预测

    === 模型结构概览 ===

    输入 (16, 512, 3)
        │
        ▼
    ┌─────────────────────────────────────┐
    │ Block 1: Conv2D(16) + MaxPool       │  ← 提取基础特征，快速降维
    │ (16,512,3) → (8,128,16)            │
    ├─────────────────────────────────────┤
    │ Block 2: SeparableConv2D(32) + Pool │  ← 深层特征，省参数
    │ (8,128,16) → (4,32,32)             │
    ├─────────────────────────────────────┤
    │ Block 3: SeparableConv2D(64) + Pool │  ← 更深层特征
    │ (4,32,32) → (2,8,64)               │
    ├─────────────────────────────────────┤
    │ Block 4: SeparableConv2D(64)        │  ← 最深层特征
    │ (2,8,64) → (2,8,64)                │
    ├─────────────────────────────────────┤
    │ GlobalAveragePooling → (64,)        │  ← 把2D特征图压成1D向量
    │ Dense(32) → Dense(num_classes)      │  ← 分类决策
    └─────────────────────────────────────┘
        │
        ▼
    输出 (num_classes,) — 每个类别的概率
    """

    # ====================================================================
    # 输入层: 定义输入数据的形状
    # ====================================================================
    # shape=(16, 512, 3) 表示: 16帧(高) × 512频率bin(宽) × 3通道(X/Y/Z)
    # 不包含 batch 维度，Keras 会自动加
    inputs = keras.Input(shape=input_shape, name="spectrogram")

    # ====================================================================
    # Block 1: 第一个卷积块 — 提取基础特征
    # ====================================================================
    # 这一层用 16 个 3×3 的卷积核扫描输入图像，提取 16 种不同的特征
    #
    # Conv2D 工作原理 (简化理解):
    #   想象一个 3×3 的小窗口在图像上滑动，每个位置做一次计算
    #   窗口每滑到一个位置，就输出一个数字
    #   16 个不同的窗口 = 16 种不同的特征检测器
    #
    # 参数说明:
    #   16        → 输出 16 个通道 (16 种特征)
    #   3         → 卷积核大小 3×3
    #   padding="same" → 输出尺寸和输入一样 (边缘补零)
    #   use_bias=False → 不用偏置项 (因为后面有 BatchNorm，偏置会被抵消)
    #   kernel_initializer="he_normal" → 权重初始化方法，适合 ReLU 激活函数
    #
    # 形状变化: (batch, 16, 512, 3) → (batch, 16, 512, 16)
    x = layers.Conv2D(
        16, 3,
        padding="same",           # 保持空间尺寸不变
        use_bias=False,           # BatchNorm 会处理偏置
        kernel_initializer="he_normal",  # He 初始化，适合 ReLU
    )(inputs)

    # BatchNormalization: 批归一化
    # 作用: 让每一层的输出保持在合理的范围内，加速训练，提高稳定性
    # 原理: 减均值除标准差，然后学两个可训练参数 γ 和 β
    x = layers.BatchNormalization()(x)

    # ReLU 激活函数: max(0, x)
    # 作用: 引入非线性，让网络能学习复杂的模式
    # 直觉: 负值变成0，正值保持不变
    x = layers.ReLU()(x)

    # MaxPooling2D: 最大池化 — 降维
    # pool_size=(2, 4) 表示: 在高(时间)方向缩小2倍，宽(频率)方向缩小4倍
    # 作用: 减小数据尺寸，减少计算量，同时保留最重要的特征
    #
    # 形状变化: (batch, 16, 512, 16) → (batch, 8, 128, 16)
    #           高: 16→8 (÷2)    宽: 512→128 (÷4)    通道: 16 不变
    x = layers.MaxPool2D(pool_size=(2, 4))(x)

    # ====================================================================
    # Block 2: 深度可分离卷积 — 更高效的特征提取
    # ====================================================================
    # SeparableConv2D vs 普通 Conv2D:
    #
    # 普通 Conv2D:
    #   16通道输入, 32通道输出 → 需要 16×32×3×3 = 4608 个参数
    #
    # SeparableConv2D (深度可分离卷积):
    #   分两步:
    #   Step 1 - Depthwise: 每个通道单独做 3×3 卷积 → 16×3×3 = 144 参数
    #   Step 2 - Pointwise: 用 1×1 卷积混合通道 → 16×32×1×1 = 512 参数
    #   总共: 144 + 512 = 656 参数 (比 4608 少了 85%!)
    #
    # 为什么用它: 参数少 → 模型小 → 适合部署到 ESP32
    #
    # 形状变化: (batch, 8, 128, 16) → (batch, 8, 128, 32)
    x = layers.SeparableConv2D(
        32, 3,                    # 32 个输出通道, 3×3 卷积核
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",   # depthwise 部分的初始化
        pointwise_initializer="he_normal",   # pointwise 部分的初始化
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 再次降维: (batch, 8, 128, 32) → (batch, 4, 32, 32)
    # 高: 8→4 (÷2)    宽: 128→32 (÷4)
    x = layers.MaxPool2D(pool_size=(2, 4))(x)

    # ====================================================================
    # Block 3: 更深的特征提取
    # ====================================================================
    # 和 Block 2 结构一样，但通道数从 32 增加到 64
    # 通道数越多，能检测的特征种类越多
    #
    # 形状变化: (batch, 4, 32, 32) → (batch, 4, 32, 64)
    x = layers.SeparableConv2D(
        64, 3,                    # 64 个输出通道
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 降维: (batch, 4, 32, 64) → (batch, 2, 8, 64)
    # 高: 4→2 (÷2)    宽: 32→8 (÷4)
    x = layers.MaxPool2D(pool_size=(2, 4))(x)

    # ====================================================================
    # Block 4: 最后一个卷积块 (不再降维)
    # ====================================================================
    # 注意: 这里没有 MaxPool 了，尺寸保持 (2, 8) 不变
    # 因为已经够小了，再缩小会丢失太多信息
    #
    # 形状变化: (batch, 2, 8, 64) → (batch, 2, 8, 64)
    x = layers.SeparableConv2D(
        64, 3,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ====================================================================
    # 分类头: 把 2D 特征图变成类别概率
    # ====================================================================

    # GlobalAveragePooling2D: 全局平均池化
    # 作用: 把每个通道的 2D 特征图 (2×8) 求平均，变成一个数字
    # 输入: (batch, 2, 8, 64) — 64 个通道，每个是 2×8 的特征图
    # 输出: (batch, 64) — 每个通道变成一个数字 (64个数字)
    #
    # 为什么用它: 比 Flatten (展平) 参数少得多
    #   Flatten: 2×8×64 = 1024 个数字 → 需要很大的全连接层
    #   GlobalAveragePooling: 64 个数字 → 全连接层只需要 64×32 = 2048 参数
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout: 随机丢弃 30% 的神经元
    # 作用: 防止过拟合 (模型死记硬背训练数据，对新数据不灵)
    # 原理: 训练时随机把一些神经元的输出设为 0，强迫网络学习更鲁棒的特征
    # 注意: 只在训练时生效，推理(预测)时不会丢弃
    x = layers.Dropout(0.3)(x)

    # Dense: 全连接层 — 32 个神经元
    # 作用: 综合前面提取的所有特征，做最终决策
    # 每个神经元都和输入的 64 个数字相连，所以有 64×32 = 2048 个权重
    #
    # 形状变化: (batch, 64) → (batch, 32)
    x = layers.Dense(
        32,                       # 32 个神经元
        activation="relu",        # ReLU 激活
        kernel_initializer="he_normal",
    )(x)

    # 再加一层 Dropout (20%)，进一步防过拟合
    x = layers.Dropout(0.2)(x)

    # 输出层: num_classes 个神经元，对应 num_classes 个类别
    # softmax 激活函数: 把原始输出转换成概率分布
    #   例如原始输出 [2.1, 5.3, 0.5, -1.2]
    #   经过 softmax 变成 [0.05, 0.90, 0.03, 0.02]
    #   所有值加起来 = 1，每个值表示属于该类的概率
    #
    # 形状变化: (batch, 32) → (batch, num_classes)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",     # 输出概率分布
        name="class_prob",        # 给输出层起个名字
    )(x)

    # ====================================================================
    # 创建模型对象
    # ====================================================================
    # 把输入和输出连起来，构成完整的模型
    model = keras.Model(inputs=inputs, outputs=outputs, name="imu_cnn")
    return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """
    编译模型 — 配置训练参数

    编译就是告诉 Keras 三件事:
    1. 用什么优化器 (optimizer): Adam — 自适应学习率，训练快且稳定
    2. 用什么损失函数 (loss): sparse_categorical_crossentropy — 多分类问题的标准损失
    3. 用什么指标 (metrics): accuracy — 准确率，方便人看

    Args:
        model: 未编译的 Keras 模型
        learning_rate: 学习率，控制每次更新权重的步幅大小
            - 太大: 训练不稳定，loss 震荡
            - 太小: 训练太慢，容易卡在局部最优
            - 0.001 (1e-3) 是一个常用的默认值

    Returns:
        编译好的模型，可以调用 model.fit() 开始训练
    """
    model.compile(
        # Adam 优化器: 结合了 SGD 和 RMSProp 的优点
        # 它会自动调整每个参数的学习率，训练效果好且稳定
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),

        # 损失函数: sparse_categorical_crossentropy (稀疏分类交叉熵)
        # "sparse" 意味着标签是整数 (如 0, 1, 2, 3)，而不是 one-hot 编码
        # 它衡量模型预测和真实标签之间的差距，差距越小 loss 越低
        loss="sparse_categorical_crossentropy",

        # 评估指标: 准确率 (预测对的比例)
        # 不影响训练过程，只是方便我们观察训练效果
        metrics=["accuracy"],
    )
    return model


# ====================================================================
# 直接运行此文件时，打印模型结构
# ====================================================================
if __name__ == "__main__":
    # 构建模型并打印结构摘要
    # 运行命令: python -m src.cnn.model
    model = build_model()
    model.summary()
    # 输出会显示每一层的名称、输出形状、参数数量
    # 帮助你理解数据在模型中是怎么流动的
