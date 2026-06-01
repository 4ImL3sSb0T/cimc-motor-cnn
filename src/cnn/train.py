"""
CNN 训练脚本

用法:
  python -m src.cnn.train                              # 使用默认 npz
  python -m src.cnn.train --data output/xxx_samples.npz  # 指定数据
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE,
    MODEL_DIR, OUTPUT_DIR, CLASS_NAMES, NUM_CLASSES,
)
from src.cnn.model import build_model, compile_model
from src.cnn.dataset import (
    load_npz, normalize, train_val_split, make_tf_dataset,
)


def find_npz(data_dir: Path) -> Path | None:
    """找到第一个 .npz 文件"""
    files = sorted(data_dir.glob("*_samples.npz"))
    return files[0] if files else None


def train(npz_path: str, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """完整训练流程"""

    # 1. 加载数据
    samples, labels = load_npz(npz_path)
    print(f"样本形状: {samples.shape}  标签: {'有' if labels is not None else '无'}")

    if labels is None:
        print("警告: 没有标签数据，生成虚拟标签用于演示")
        labels = np.random.randint(0, NUM_CLASSES, size=len(samples))

    # 2. 标准化
    samples, norm_stats = normalize(samples)

    # 3. 划分训练/验证集
    x_train, x_val, y_train, y_val = train_val_split(samples, labels)

    # 4. 构建 Dataset
    train_ds = make_tf_dataset(x_train, y_train, batch_size, shuffle=True, augment=True)
    val_ds = make_tf_dataset(x_val, y_val, batch_size, shuffle=False, augment=False)

    # 5. 构建模型
    model = build_model()
    model = compile_model(model, learning_rate=LEARNING_RATE)
    model.summary()

    # 6. 回调
    MODEL_DIR.mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1,
        ),
    ]

    # 7. 训练
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # 8. 保存归一化参数和类别名
    meta = {
        "norm_stats": norm_stats,
        "class_names": CLASS_NAMES,
        "input_shape": list(model.input_shape[1:]),
        "num_classes": NUM_CLASSES,
    }
    meta_path = MODEL_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"元数据已保存: {meta_path}")

    # 9. 保存最终模型
    model.save(str(MODEL_DIR / "final.keras"))
    print(f"模型已保存: {MODEL_DIR}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="训练 IMU CNN 模型")
    parser.add_argument("--data", type=str, default=None,
                        help="npz 文件路径 (默认自动查找)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    npz_path = args.data
    if npz_path is None:
        npz_path = find_npz(OUTPUT_DIR)
        if npz_path is None:
            print(f"未找到 npz 文件，请先运行 python -m src.data.process 生成样本")
            sys.exit(1)
    print(f"数据: {npz_path}")

    train(str(npz_path), epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
