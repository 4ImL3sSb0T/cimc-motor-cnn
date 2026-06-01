"""
模型导出 — 把训练好的模型转成 ONNX 格式

=== 为什么导出 ONNX? ===
训练好的模型是 .keras 格式 (Keras 专用)，部署时需要通用格式。
ONNX (Open Neural Network Exchange) 是开放的模型格式，支持多种推理框架:
  - ONNX Runtime (PC/服务器)
  - TensorRT (NVIDIA GPU)
  - OpenVINO (Intel CPU/GPU)
  - ESP-DL (ESP32-S3，需额外转换)

注: TFLite 导出因 TF 2.16.x 的 MLIR bug 无法使用 (详见 tensorflow#63987)。

=== 用法 ===
  python -m src.cnn.export
  python -m src.cnn.export --model models/best.keras
"""

import argparse
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.config import MODEL_DIR


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
    命令行入口 — 导出模型为 ONNX 格式。

    示例:
      python -m src.cnn.export
      python -m src.cnn.export --model models/best.keras
    """
    parser = argparse.ArgumentParser(description="导出 CNN 模型为 ONNX")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR / "best.keras"),
                        help="Keras 模型路径 (默认: models/best.keras)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"模型不存在: {model_path}")
        print("请先运行 python -m src.cnn.train 训练模型")
        sys.exit(1)

    model = keras.models.load_model(str(model_path))
    model.summary()

    MODEL_DIR.mkdir(exist_ok=True)
    export_onnx(model, MODEL_DIR / "model.onnx")

    print(f"\n模型已导出到: {MODEL_DIR / 'model.onnx'}")


if __name__ == "__main__":
    main()
