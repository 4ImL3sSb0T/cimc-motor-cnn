"""
CNN 模型定义 — 用于 IMU 振动频谱图分类

输入: (batch, 16, 512, 3)  — 16帧 × 512频率bin × 3通道(X/Y/Z), channels_last
输出: (batch, num_classes)  — 类别概率

设计目标:
  - 轻量级，适合 ESP32-S3 部署
  - 使用深度可分离卷积减少参数量
  - 全局平均池化替代大尺寸全连接层
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
        num_classes: 分类类别数

    Returns:
        编译好的 Keras Model
    """
    inputs = keras.Input(shape=input_shape, name="spectrogram")

    # ── Block 1: 快速降维 ────────────────────────────────────────────────
    # Conv2D: (16, 512, 3) → (16, 512, 16)
    x = layers.Conv2D(16, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # MaxPool: (16, 512) → (8, 128)
    x = layers.MaxPool2D(pool_size=(2, 4))(x)

    # ── Block 2: 深度可分离卷积 ──────────────────────────────────────────
    # DepthwiseSeparable: (8, 128, 16) → (8, 128, 32)
    x = layers.SeparableConv2D(32, 3, padding="same", use_bias=False,
                               depthwise_initializer="he_normal",
                               pointwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # MaxPool: (8, 128) → (4, 32)
    x = layers.MaxPool2D(pool_size=(2, 4))(x)

    # ── Block 3: 深度可分离卷积 ──────────────────────────────────────────
    # DepthwiseSeparable: (4, 32, 32) → (4, 32, 64)
    x = layers.SeparableConv2D(64, 3, padding="same", use_bias=False,
                               depthwise_initializer="he_normal",
                               pointwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # MaxPool: (4, 32) → (2, 8)
    x = layers.MaxPool2D(pool_size=(2, 4))(x)

    # ── Block 4: 深度可分离卷积 ──────────────────────────────────────────
    # DepthwiseSeparable: (2, 8, 64) → (2, 8, 64)
    x = layers.SeparableConv2D(64, 3, padding="same", use_bias=False,
                               depthwise_initializer="he_normal",
                               pointwise_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ── 分类头 ──────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)    # (64,)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu",
                     kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="class_prob")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="imu_cnn")
    return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """编译模型"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
