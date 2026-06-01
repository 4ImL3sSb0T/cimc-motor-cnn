"""
模型导出 — TFLite (int8 量化) + ONNX
用于 ESP32-S3 部署

用法:
  python -m src.cnn.export
  python -m src.cnn.export --model models/best.keras --data output/xxx_samples.npz
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
    files = sorted(data_dir.glob("*_samples.npz"))
    return files[0] if files else None


def export_tflite_int8(
    model: keras.Model,
    calib_samples: np.ndarray,
    out_path: str | Path,
) -> Path:
    """
    导出 int8 量化 TFLite 模型。
    calib_samples: 用于校准的样本, shape=(N, 16, 512, 3)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # int8 全量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: [
        calib_samples[i:i+1] for i in range(min(len(calib_samples), 200))
    ]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path = Path(out_path)
    out_path.write_bytes(tflite_model)
    size_kb = out_path.stat().st_size / 1024
    print(f"TFLite int8: {out_path} ({size_kb:.1f} KB)")
    return out_path


def export_tflite_float32(
    model: keras.Model,
    out_path: str | Path,
) -> Path:
    """导出 float32 TFLite 模型 (无量化)"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out_path = Path(out_path)
    out_path.write_bytes(tflite_model)
    size_kb = out_path.stat().st_size / 1024
    print(f"TFLite f32:  {out_path} ({size_kb:.1f} KB)")
    return out_path


def export_onnx(model: keras.Model, out_path: str | Path) -> Path:
    """导出 ONNX 格式"""
    try:
        import tf2onnx
        out_path = Path(out_path)
        spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
        with open(str(out_path), "wb") as f:
            f.write(model_proto.SerializeToString())
        size_kb = out_path.stat().st_size / 1024
        print(f"ONNX:        {out_path} ({size_kb:.1f} KB)")
        return out_path
    except ImportError:
        print("跳过 ONNX 导出 (需要 pip install tf2onnx)")
        return None


def main():
    parser = argparse.ArgumentParser(description="导出 CNN 模型")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR / "best.keras"),
                        help="Keras 模型路径")
    parser.add_argument("--data", type=str, default=None,
                        help="校准数据 npz 路径")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"模型不存在: {model_path}")
        sys.exit(1)

    # 加载模型
    model = keras.models.load_model(str(model_path))
    model.summary()

    # 加载校准数据
    npz_path = args.data
    if npz_path is None:
        npz_path = find_npz(OUTPUT_DIR)
    if npz_path is None:
        print("警告: 无校准数据，仅导出 float32 模型")
        calib = None
    else:
        data = np.load(str(npz_path))
        calib = np.transpose(data["samples"], (0, 2, 3, 1)).astype(np.float32)
        # 标准化 (使用训练时的参数)
        meta_path = model_path.parent / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            ns = meta["norm_stats"]
            mean = np.array(ns["mean"]).reshape(1, 1, 1, 3)
            std = np.array(ns["std"]).reshape(1, 1, 1, 3)
            calib = (calib - mean) / std

    # 导出
    MODEL_DIR.mkdir(exist_ok=True)

    export_tflite_float32(model, MODEL_DIR / "model_float32.tflite")

    if calib is not None:
        export_tflite_int8(model, calib, MODEL_DIR / "model_int8.tflite")

    export_onnx(model, MODEL_DIR / "model.onnx")

    # 保存 meta 副本到模型目录
    print(f"\n所有模型已导出到: {MODEL_DIR}")


if __name__ == "__main__":
    main()
