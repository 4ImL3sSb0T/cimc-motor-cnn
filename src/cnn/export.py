"""
模型导出 — 把训练好的模型转成能部署到 ESP32 的格式

=== 为什么要导出? ===
训练好的模型是 .keras 格式 (Keras 专用)，ESP32 用不了。
需要转成 TFLite 格式 (TensorFlow Lite)，这是专为嵌入式设备设计的轻量格式。

=== 导出的三种格式 ===

1. TFLite float32 (.tflite):
   - 不做量化，精度最高
   - 文件较大，适合在 PC/手机上测试

2. TFLite int8 (.tflite):
   - 全量化: 所有权重和计算都用 int8 (8位整数)
   - 文件最小 (约是 float32 的 1/4)
   - 推理速度最快
   - ESP32-S3 部署用的就是这个

3. ONNX (.onnx):
   - 开放的模型格式，其他推理框架也能用
   - 如果不用 TFLite，可以用 ONNX Runtime 来跑

=== 什么是量化? ===
原始模型用 float32 (32位浮点数) 存储每个参数，一个参数占 4 字节。
量化后用 int8 (8位整数)，一个参数只占 1 字节。

好处:
  - 模型体积: 43 KB → 约 11 KB (缩小 4 倍)
  - 推理速度: int8 计算比 float32 快很多
  - 内存占用: 更少，ESP32 才放得下

代价:
  - 精度会略微下降 (通常 < 2%)

=== 校准数据 ===
int8 量化需要"校准数据"来确定每个层的数值范围。
校准数据就是一批真实的训练样本 (通常 100~200 个就够了)。
脚本会自动用训练数据来校准。

=== 用法 ===
  python -m src.cnn.export
  python -m src.cnn.export --model models/best.keras
  python -m src.cnn.export --model models/best.keras --data output/xxx.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import MODEL_DIR, OUTPUT_DIR


def find_npz(data_dir: Path) -> Path | None:
    """在目录中找到第一个 .npz 样本文件 (用作校准数据)"""
    files = sorted(data_dir.glob("*_samples.npz"))
    return files[0] if files else None


def export_tflite_int8(
    model: keras.Model,
    calib_samples: np.ndarray,
    out_path: str | Path,
) -> Path:
    """
    导出 int8 全量化 TFLite 模型 — ESP32 部署用。

    === int8 量化原理 ===
    float32 范围: 约 ±3.4×10^38，精度 7 位小数
    int8 范围: -128 ~ 127，只有 256 个值

    量化过程:
    1. 用校准数据跑一遍模型，记录每层的数值范围 (如 -5.3 ~ 3.7)
    2. 把 [-5.3, 3.7] 映射到 [-128, 127]
    3. 推理时所有计算都用 int8，最后再反量化回 float

    Args:
        model: 训练好的 Keras 模型
        calib_samples: 校准数据, shape=(N, 16, 512, 3)
        out_path: 输出文件路径

    Returns:
        输出文件路径
    """
    # 从 Keras 模型创建 TFLite 转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # ── 设置量化选项 ──────────────────────────────────────────────

    # 启用默认优化 (包括量化)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 提供校准数据: 转换器会用这些数据来确定每层的数值范围
    # 每次取 1 个样本，最多取 200 个 (够用了)
    converter.representative_dataset = lambda: [
        calib_samples[i:i+1] for i in range(min(len(calib_samples), 200))
    ]

    # 指定只用 int8 算子 (不用 float 算子)
    # 这样整个模型都是 int8 计算，最大化性能
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # 输入输出也用 int8 (而不是只量化内部层)
    # 好处: ESP32 采集到传感器数据后直接量化为 int8 就能送入模型
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # ── 执行转换 ──────────────────────────────────────────────────
    tflite_model = converter.convert()

    # 保存到文件
    out_path = Path(out_path)
    out_path.write_bytes(tflite_model)
    size_kb = out_path.stat().st_size / 1024
    print(f"TFLite int8: {out_path} ({size_kb:.1f} KB)")
    return out_path


def export_tflite_float32(
    model: keras.Model,
    out_path: str | Path,
) -> Path:
    """
    导出 float32 TFLite 模型 — 不做量化，保持原始精度。

    用途:
    - 在 PC/手机上测试推理效果
    - 和 int8 模型对比精度差异
    - 如果设备性能足够，也可以直接用 float32 版本

    Args:
        model: 训练好的 Keras 模型
        out_path: 输出文件路径

    Returns:
        输出文件路径
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    out_path = Path(out_path)
    out_path.write_bytes(tflite_model)
    size_kb = out_path.stat().st_size / 1024
    print(f"TFLite f32:  {out_path} ({size_kb:.1f} KB)")
    return out_path


def export_onnx(model: keras.Model, out_path: str | Path) -> Path:
    """
    导出 ONNX 格式模型。

    ONNX (Open Neural Network Exchange) 是一种通用的模型格式。
    如果你不用 TFLite，可以用 ONNX Runtime 来跑推理。
    ONNX Runtime 支持更多平台和硬件加速。

    Args:
        model: 训练好的 Keras 模型
        out_path: 输出文件路径

    Returns:
        输出文件路径，或 None (如果 tf2onnx 未安装)
    """
    try:
        import tf2onnx
    except ImportError:
        print("跳过 ONNX 导出 (需要 pip install tf2onnx)")
        return None

    out_path = Path(out_path)

    # 定义输入签名: 告诉转换器输入的形状和类型
    spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)

    # 转换
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

    # 保存
    with open(str(out_path), "wb") as f:
        f.write(model_proto.SerializeToString())
    size_kb = out_path.stat().st_size / 1024
    print(f"ONNX:        {out_path} ({size_kb:.1f} KB)")
    return out_path


def main():
    """
    命令行入口 — 导出模型为 TFLite 和 ONNX 格式。

    流程:
    1. 加载训练好的 .keras 模型
    2. 加载校准数据 (用于 int8 量化)
    3. 用训练时保存的归一化参数对校准数据做标准化
    4. 导出 float32 TFLite、int8 TFLite、ONNX 三种格式

    示例:
      python -m src.cnn.export
      python -m src.cnn.export --model models/best.keras --data output/xxx.npz
    """
    # ── 解析命令行参数 ────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="导出 CNN 模型")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR / "best.keras"),
                        help="Keras 模型路径 (默认: models/best.keras)")
    parser.add_argument("--data", type=str, default=None,
                        help="校准数据 npz 路径 (默认自动查找)")
    args = parser.parse_args()

    # ── 检查模型文件是否存在 ──────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"模型不存在: {model_path}")
        print("请先运行 python -m src.cnn.train 训练模型")
        sys.exit(1)

    # ── 加载模型 ──────────────────────────────────────────────────
    # keras.models.load_model 会恢复模型的全部结构和权重
    model = keras.models.load_model(str(model_path))
    model.summary()  # 打印模型结构

    # ── 加载校准数据 ──────────────────────────────────────────────
    # int8 量化需要真实数据来校准数值范围
    npz_path = args.data
    if npz_path is None:
        # 没指定则自动查找
        npz_path = find_npz(OUTPUT_DIR)

    if npz_path is None:
        # 没有校准数据，只能导出 float32 版本
        print("警告: 无校准数据，仅导出 float32 模型")
        calib = None
    else:
        # 加载校准数据
        data = np.load(str(npz_path))
        # 转置: (N, 3, 16, 512) → (N, 16, 512, 3) — channels_last
        calib = np.transpose(data["samples"], (0, 2, 3, 1)).astype(np.float32)

        # 用训练时的归一化参数做标准化
        # 这很重要! 推理时也必须用同样的 mean/std
        meta_path = model_path.parent / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            ns = meta["norm_stats"]
            # reshape 成 (1,1,1,3) 方便广播
            mean = np.array(ns["mean"]).reshape(1, 1, 1, 3)
            std = np.array(ns["std"]).reshape(1, 1, 1, 3)
            # 标准化: (x - mean) / std
            calib = (calib - mean) / std

    # ── 导出所有格式 ──────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. float32 TFLite (不量化，用于测试)
    export_tflite_float32(model, MODEL_DIR / "model_float32.tflite")

    # 2. int8 TFLite (全量化，用于 ESP32 部署)
    if calib is not None:
        export_tflite_int8(model, calib, MODEL_DIR / "model_int8.tflite")

    # 3. ONNX (通用格式)
    export_onnx(model, MODEL_DIR / "model.onnx")

    print(f"\n所有模型已导出到: {MODEL_DIR}")
    print("ESP32 部署请使用: model_int8.tflite + meta.json")


if __name__ == "__main__":
    main()
