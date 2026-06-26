"""
模型导出 — 把训练好的模型转成 ONNX 格式 (支持 FP32 和 INT8 量化)

=== 为什么导出 ONNX? ===
训练好的模型是 .keras 格式 (Keras 专用)，部署时需要通用格式。
ONNX (Open Neural Network Exchange) 是开放的模型格式，支持多种推理框架:
  - ONNX Runtime (PC/服务器)
  - TensorRT (NVIDIA GPU)
  - OpenVINO (Intel CPU/GPU)
  - ESP-DL (ESP32-S3，需额外转换)

=== INT8 静态量化 ===
INT8 量化把 FP32 权重和激活值压缩到 INT8 (8位整数):
  - 模型大小: 约 4x 压缩 (140KB → ~35KB)
  - 推理速度: 快 2-4x (CPU/嵌入式设备)
  - 精度损失: 通常 <1% (需要校准数据验证)

注: TFLite 导出因 TF 2.16.x 的 MLIR bug 无法使用 (详见 tensorflow#63987)。

=== 用法 ===
  python -m src.cnn.export                      # 导出 FP32 ONNX
  python -m src.cnn.export --int8               # 导出 INT8 量化 ONNX
  python -m src.cnn.export --int8 --calib 200   # 用 200 个样本校准
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import MODEL_DIR, OUTPUT_DIR


class CalibrationDataReader:
    """
    ONNX Runtime 量化校准数据读取器。

    把 numpy 校准数据包装成 onnxruntime.quantization 要求的迭代器格式。
    每次 next() 返回一个 batch 的输入数据。
    """

    def __init__(self, data: np.ndarray, batch_size: int = 16):
        """
        Args:
            data: 校准数据, shape=(N, 16, 512, 3), float32
            batch_size: 每次返回的样本数
        """
        self.data = data
        self.batch_size = batch_size
        self.index = 0
        self.n_samples = len(data)

    def get_next(self) -> dict | None:
        """返回下一个 batch, 或 None 表示结束"""
        if self.index >= self.n_samples:
            return None

        end = min(self.index + self.batch_size, self.n_samples)
        batch = self.data[self.index:end]
        self.index = end

        # ONNX Runtime 期望 dict: {input_name: numpy_array}
        return {"input": batch}


def load_calibration_data(max_samples: int = 200) -> np.ndarray:
    """
    从 output/ 目录加载校准数据。

    自动查找 *_samples.npz 文件, 合并后随机采样 max_samples 个样本。
    校准数据用于 INT8 量化时统计激活值分布。

    Args:
        max_samples: 最多使用多少个样本校准 (越多越准, 但更慢)

    Returns:
        校准数据, shape=(N, 16, 512, 3), float32
    """
    npz_files = sorted(OUTPUT_DIR.glob("*_samples.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"找不到校准数据: {OUTPUT_DIR}/*_samples.npz\n"
            "请先运行 python -m src.data.process 生成样本"
        )

    # 加载所有样本
    all_samples = []
    for f in npz_files:
        data = np.load(f)
        samples = data["samples"]  # shape=(N, 3, 16, 512)
        # 转成 channels_last: (N, 3, 16, 512) → (N, 16, 512, 3)
        samples = np.transpose(samples, (0, 2, 3, 1))
        all_samples.append(samples)

    all_samples = np.concatenate(all_samples, axis=0)
    print(f"校准数据: {len(all_samples)} 个样本 (来自 {len(npz_files)} 个文件)")

    # 随机采样
    if len(all_samples) > max_samples:
        indices = np.random.choice(len(all_samples), max_samples, replace=False)
        all_samples = all_samples[indices]
        print(f"随机采样: {max_samples} 个样本用于校准")

    return all_samples.astype(np.float32)


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
    print(f"ONNX (FP32): {out_path} ({size_kb:.1f} KB)")
    return out_path


def export_onnx_int8(
    model: keras.Model,
    out_path: str | Path,
    calib_data: np.ndarray,
    batch_size: int = 16,
) -> Path:
    """
    导出 INT8 静态量化 ONNX 模型。

    静态量化流程:
    1. 先导出 FP32 ONNX (中间文件)
    2. 用校准数据跑一遍推理, 统计每层激活值的分布
    3. 根据统计结果确定 INT8 的缩放因子和零点
    4. 把权重和激活值都量化到 INT8

    Args:
        model: 训练好的 Keras 模型
        out_path: 输出 INT8 ONNX 文件路径
        calib_data: 校准数据, shape=(N, 16, 512, 3)
        batch_size: 校准时的 batch 大小

    Returns:
        输出文件路径
    """
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

    out_path = Path(out_path)

    # Step 1: 先导出 FP32 ONNX (量化需要输入模型)
    fp32_path = out_path.parent / f"{out_path.stem}_fp32.onnx"
    export_onnx(model, fp32_path)

    # Step 2: 创建校准数据读取器
    reader = CalibrationDataReader(calib_data, batch_size=batch_size)

    # Step 3: 静态量化
    print(f"正在进行 INT8 静态量化 (校准 {len(calib_data)} 个样本)...")
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        per_channel=True,           # 逐通道量化 (精度更高)
        weight_type=QuantType.QInt8,  # 权重用 INT8
        activation_type=QuantType.QInt8,  # 激活值用 INT8
        quant_format=QuantFormat.QOperator,  # QOperator 格式 (更简洁, 兼容性更好)
        op_types_to_quantize=["Conv", "MatMul"],  # 只量化 Conv 和 MatMul
    )

    # 清理中间文件
    if fp32_path.exists():
        fp32_path.unlink()

    size_kb = out_path.stat().st_size / 1024
    print(f"ONNX (INT8): {out_path} ({size_kb:.1f} KB)")
    return out_path


def main():
    """
    命令行入口 — 导出模型为 ONNX 格式 (支持 FP32 和 INT8)。

    示例:
      python -m src.cnn.export                      # 导出 FP32
      python -m src.cnn.export --int8               # 导出 INT8 量化
      python -m src.cnn.export --int8 --calib 500   # 用 500 个样本校准
    """
    parser = argparse.ArgumentParser(description="导出 CNN 模型为 ONNX (支持 INT8 量化)")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR / "best.keras"),
                        help="Keras 模型路径 (默认: models/best.keras)")
    parser.add_argument("--int8", action="store_true",
                        help="导出 INT8 量化模型 (需要校准数据)")
    parser.add_argument("--calib", type=int, default=200,
                        help="INT8 校准样本数 (默认: 200, 越多越准但更慢)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"模型不存在: {model_path}")
        print("请先运行 python -m src.cnn.train 训练模型")
        sys.exit(1)

    model = keras.models.load_model(str(model_path))
    model.summary()

    MODEL_DIR.mkdir(exist_ok=True)

    if args.int8:
        # INT8 静态量化
        print("\n" + "="*60)
        print("导出 INT8 量化模型")
        print("="*60)

        calib_data = load_calibration_data(max_samples=args.calib)
        export_onnx_int8(
            model,
            MODEL_DIR / "model_int8.onnx",
            calib_data=calib_data,
        )
        print(f"\nINT8 模型已导出到: {MODEL_DIR / 'model_int8.onnx'}")
    else:
        # 普通 FP32 导出
        export_onnx(model, MODEL_DIR / "model.onnx")
        print(f"\nFP32 模型已导出到: {MODEL_DIR / 'model.onnx'}")


if __name__ == "__main__":
    main()
