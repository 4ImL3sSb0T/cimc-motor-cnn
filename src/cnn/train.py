"""
CNN 训练脚本

=== 训练流程概览 ===

  .npz 文件          逐文件加载        按文件划分           标准化            创建Dataset      训练
  (带标签)      →   [(s,N),...]   →  train_list / val_list → (train stats) → (tf.data) → model.fit
  │                                                                       │
  │                                                    └─ 保存 meta.json (归一化参数) ←────┘
  │
  └─ 多文件 → 按采集文件整体划分 (消除滑动窗口数据泄漏)
      单文件 → 回退到逐样本分层划分

=== 用法 ===
  python -m src.cnn.train                                        # 自动找 output/*.npz + 自动划分
  python -m src.cnn.train --data output/a.npz                    # 指定单个文件
  python -m src.cnn.train --data output/*.npz                    # 通配符，自动划分
  python -m src.cnn.train --val-files output/a.npz output/b.npz  # 手动指定验证文件
  python -m src.cnn.train --epochs 200 --batch-size 32
  python -m src.cnn.train --focal-loss                           # 使用 Focal Loss
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from src.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE, VALIDATION_SPLIT,
    MODEL_DIR, OUTPUT_DIR, CLASS_NAMES, NUM_CLASSES, CNN_FREQ_BINS,
)
from src.cnn.model import build_model, compile_model, focal_loss
from src.cnn.dataset import (
    load_npz, normalize, train_val_split, train_val_split_by_file,
    make_tf_dataset, compute_class_weights, classification_report_from_cm,
)

# 中文字体 (同 visualizer.py)
_CN_FONT = None
for _name in ["Microsoft YaHei", "SimHei", "NSimSun"]:
    _matches = [f for f in fm.fontManager.ttflist if f.name == _name]
    if _matches:
        _CN_FONT = fm.FontProperties(fname=_matches[0].fname)
        break
if _CN_FONT is None:
    from pathlib import Path as _P
    for _fname in ["msyh.ttc", "simhei.ttf"]:
        _fpath = _P("/mnt/c/Windows/Fonts") / _fname
        if _fpath.exists():
            fm.fontManager.addfont(str(_fpath))
            _CN_FONT = fm.FontProperties(fname=str(_fpath))
            break
if _CN_FONT is None:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    _CN_FONT = fm.FontProperties()


def find_npz(data_dir: Path) -> list[Path]:
    """在目录中找到所有 *_samples.npz 文件"""
    return sorted(data_dir.glob("*_samples.npz"))


def train(
    npz_paths: list[str],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    use_focal_loss: bool = False,
    val_files: list[str] | None = None,
):
    """
    完整训练流程 — 从加载数据到保存模型。

    多文件: 按采集文件划分 train/val (消除滑动窗口的数据泄漏)
    单文件: 回退到逐样本分层划分

    Args:
        npz_paths: .npz 样本文件路径列表
        epochs: 训练轮数 (默认100)
        batch_size: 每批样本数 (默认16)
        use_focal_loss: 是否使用 Focal Loss
        val_files: 手动指定验证文件 (覆盖自动划分, 默认 None)

    Returns:
        (model, history):
          - model: 训练好的 Keras 模型
          - history: 训练历史
    """
    from collections import Counter

    # ==================================================================
    # 第 1 步: 逐文件加载 (保留文件边界, 不做跨文件合并)
    # ==================================================================
    file_data: list[tuple[np.ndarray, np.ndarray, str]] = []
    for p in npz_paths:
        p = Path(p)
        samples, labels = load_npz(str(p))
        if labels is None:
            print(f"  跳过 {p.name} (无标签)")
            continue
        file_data.append((samples, labels, p.name))

    if not file_data:
        print("错误: 没有可用的带标签 .npz 文件")
        sys.exit(1)

    total = sum(len(d[0]) for d in file_data)
    print(f"\n加载 {len(file_data)} 个文件, 总计 {total} 样本")
    for samples, labels, fname in file_data:
        counts = Counter(labels.tolist())
        cls_str = ", ".join(
            f"{CLASS_NAMES[c]}:{counts[c]}" for c in sorted(counts)
        )
        print(f"  {fname}: {len(samples)} 样本  [{cls_str}]")

    # ==================================================================
    # 第 2 步: 按文件划分 (核心修复!)
    # ==================================================================
    if len(file_data) > 1:
        if val_files is not None:
            # 手动模式: 用户指定验证文件
            val_names = {Path(f).name for f in val_files}
            train_files = [d for d in file_data if d[2] not in val_names]
            val_files_list = [d for d in file_data if d[2] in val_names]
            n_tr = sum(len(d[0]) for d in train_files)
            n_vl = sum(len(d[0]) for d in val_files_list)
            print(f"\n手动划分:")
            print(f"  Train ({n_tr}): {[d[2] for d in train_files]}")
            print(f"  Val   ({n_vl}): {[d[2] for d in val_files_list]}")
        else:
            # 自动模式: 贪心算法确保每类都在 train/val 中出现
            train_files, val_files_list = train_val_split_by_file(
                file_data, val_ratio=VALIDATION_SPLIT,
            )

        x_train_raw = np.concatenate([d[0] for d in train_files])
        y_train = np.concatenate([d[1] for d in train_files])
        x_val_raw = np.concatenate([d[0] for d in val_files_list])
        y_val = np.concatenate([d[1] for d in val_files_list])
    else:
        # 单文件: 回退到逐样本分层划分
        print("\n仅 1 个文件, 回退到逐样本分层划分 (滑动窗口数据可能泄漏)")
        samples, labels, _ = file_data[0]
        x_train_raw, x_val_raw, y_train, y_val = train_val_split(
            samples, labels,
        )

    print(f"\n类别分布:")
    for name, arr in [("Train", y_train), ("Val", y_val)]:
        counts = Counter(arr.tolist())
        cls_str = ", ".join(
            f"{CLASS_NAMES[c]}:{counts.get(c, 0)}" for c in range(NUM_CLASSES)
        )
        print(f"  {name}: {len(arr)} 样本  [{cls_str}]")

    # ==================================================================
    # 第 3 步: 标准化 (stats 仅从训练集计算, 杜绝数据泄漏)
    # ==================================================================
    x_train, norm_stats = normalize(x_train_raw)
    x_val, _ = normalize(x_val_raw, stats=norm_stats)

    # ==================================================================
    # 第 4 步: 创建 tf.data.Dataset
    # ==================================================================
    train_ds = make_tf_dataset(x_train, y_train, batch_size,
                               shuffle=True, augment=True)
    val_ds = make_tf_dataset(x_val, y_val, batch_size,
                             shuffle=False, augment=False)

    # ==================================================================
    # 第 5 步: 构建并编译模型
    # ==================================================================
    model = build_model()
    model = compile_model(model, learning_rate=LEARNING_RATE, use_focal_loss=use_focal_loss)
    model.summary()

    # ==================================================================
    # 第 6 步: 计算类别权重
    # ==================================================================
    class_weight = compute_class_weights(y_train, NUM_CLASSES)

    # ==================================================================
    # 第 7 步: 设置回调函数
    # ==================================================================
    MODEL_DIR.mkdir(exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]

    # ==================================================================
    # 第 8 步: 开始训练
    # ==================================================================
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # ==================================================================
    # 第 9 步: 保存元数据
    # ==================================================================
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

    # ==================================================================
    # 第 10 步: 保存最终模型
    # ==================================================================
    model.save(str(MODEL_DIR / "final.keras"))
    print(f"模型已保存: {MODEL_DIR}")

    # ==================================================================
    # 第 11 步: 训练可视化 + 分类报告
    # ==================================================================
    plot_training_result(model, history, x_val, y_val)

    return model, history


def plot_training_result(
    model: keras.Model,
    history,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
    """绘制训练曲线 + 混淆矩阵"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("训练结果", fontproperties=_CN_FONT, fontsize=14)

    # ── 训练曲线 ──
    h = history.history
    epochs_range = range(1, len(h["loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs_range, h["loss"], "b-", label="训练 loss")
    ax.plot(epochs_range, h["val_loss"], "r-", label="验证 loss")
    ax.set_xlabel("Epoch", fontproperties=_CN_FONT)
    ax.set_ylabel("Loss", fontproperties=_CN_FONT)
    ax.set_title("损失曲线", fontproperties=_CN_FONT)
    ax.legend(prop=_CN_FONT)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_range, h["accuracy"], "b-", label="训练 accuracy")
    ax.plot(epochs_range, h["val_accuracy"], "r-", label="验证 accuracy")
    ax.set_xlabel("Epoch", fontproperties=_CN_FONT)
    ax.set_ylabel("Accuracy", fontproperties=_CN_FONT)
    ax.set_title("准确率曲线", fontproperties=_CN_FONT)
    ax.legend(prop=_CN_FONT)
    ax.grid(True, alpha=0.3)

    # ── 混淆矩阵 ──
    y_pred = model.predict(x_val, verbose=0)
    y_pred_cls = np.argmax(y_pred, axis=1)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_val, y_pred_cls):
        cm[t][p] += 1

    ax = axes[2]
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=12)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, fontproperties=_CN_FONT, fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontproperties=_CN_FONT, fontsize=9)
    ax.set_xlabel("预测", fontproperties=_CN_FONT)
    ax.set_ylabel("真实", fontproperties=_CN_FONT)
    ax.set_title("混淆矩阵", fontproperties=_CN_FONT)

    plt.tight_layout()
    save_path = MODEL_DIR / "training_result.png"
    fig.savefig(str(save_path), dpi=150)
    print(f"训练结果图已保存: {save_path}")
    plt.show()

    # ── 分类报告 ──
    classification_report_from_cm(cm, CLASS_NAMES)


def main():
    """
    命令行入口。

    支持的参数:
      --data: .npz 文件路径，支持多个文件或通配符 (不指定则自动查找 output/ 目录)
      --epochs: 训练轮数 (默认100)
      --batch-size: 每批样本数 (默认16)
      --focal-loss: 使用 Focal Loss 代替标准交叉熵 (对类别不平衡更鲁棒)
      --val-files: 手动指定验证文件 (覆盖自动按文件划分)

    示例:
      python -m src.cnn.train                                        # 自动找 output/*.npz + 自动划分
      python -m src.cnn.train --data output/a.npz                    # 单个文件
      python -m src.cnn.train --data output/*.npz                    # 通配符
      python -m src.cnn.train --epochs 200 --batch-size 32
      python -m src.cnn.train --focal-loss
      python -m src.cnn.train --val-files output/a.npz output/b.npz  # 手动指定验证文件
    """
    # 手动解析: --data / --val-files 后面所有非 -- 开头的参数都当作文件路径
    npz_paths: list[str] = []
    val_paths: list[str] = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("--data", "--val-files"):
            mode = arg
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                if mode == "--data":
                    npz_paths.append(sys.argv[i])
                else:
                    val_paths.append(sys.argv[i])
                i += 1
            continue
        i += 1

    # 解析其他参数
    parser = argparse.ArgumentParser(description="训练 IMU CNN 模型")
    parser.add_argument("--data", type=str, nargs="+", default=None)
    parser.add_argument("--val-files", type=str, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="训练轮数 (默认: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="每批样本数 (默认: %(default)s)")
    parser.add_argument("--focal-loss", action="store_true",
                        help="使用 Focal Loss (对类别不平衡更鲁棒)")
    args = parser.parse_args()

    # 确定数据文件路径
    if not npz_paths:
        found = find_npz(OUTPUT_DIR)
        if not found:
            print(f"未找到 npz 文件，请先运行 python -m src.data.process 生成样本")
            sys.exit(1)
        npz_paths = [str(p) for p in found]

    print(f"数据文件: {len(npz_paths)} 个")
    for p in npz_paths:
        print(f"  {p}")

    if val_paths:
        print(f"验证文件 (手动): {len(val_paths)} 个")
        for p in val_paths:
            print(f"  {p}")

    # 开始训练
    train(
        npz_paths,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_focal_loss=args.focal_loss,
        val_files=val_paths if val_paths else None,
    )


if __name__ == "__main__":
    main()
