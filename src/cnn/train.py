"""
CNN 训练脚本

=== 训练流程概览 ===

  .npz 文件          标准化           划分            构建           开始
  (带标签)      →   (减均值/除std)  → (80%训练/20%验证) → (tf.data)  → 训练
     │                                              │
     │                                              │
     └──────────── 保存 meta.json (归一化参数) ←─────┘

=== 用法 ===
  python -m src.cnn.train                                        # 自动找 output/*.npz
  python -m src.cnn.train --data output/a.npz                    # 指定单个文件
  python -m src.cnn.train --data output/a.npz output/b.npz       # 指定多个文件，合并训练
  python -m src.cnn.train --data output/*.npz                    # 通配符，合并所有
  python -m src.cnn.train --epochs 200 --batch-size 32           # 自定义参数
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
    BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE,
    MODEL_DIR, OUTPUT_DIR, CLASS_NAMES, NUM_CLASSES,
)
from src.cnn.model import build_model, compile_model
from src.cnn.dataset import (
    load_npz, normalize, train_val_split, make_tf_dataset,
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


def load_multiple_npz(npz_paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    加载并合并多个 .npz 文件。

    每个文件必须包含 "samples" 和 "labels" 键。
    只合并有标签的文件，跳过无标签的并打印警告。

    Returns:
        (samples, labels): 合并后的样本和标签
    """
    all_samples = []
    all_labels = []

    for p in npz_paths:
        p = Path(p)
        data = np.load(str(p))
        samples = data["samples"]
        labels = data.get("labels")

        if labels is None:
            print(f"  跳过 {p.name} (无标签)")
            continue

        # 转置: (N, 3, 16, 512) → (N, 16, 512, 3)
        samples = np.transpose(samples, (0, 2, 3, 1)).astype(np.float32)
        labels = labels.astype(np.int32)

        print(f"  {p.name}: {samples.shape[0]} 样本")
        all_samples.append(samples)
        all_labels.append(labels)

    if not all_samples:
        raise ValueError("没有可用的带标签 .npz 文件")

    samples = np.concatenate(all_samples, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"合并总计: {samples.shape[0]} 样本")
    return samples, labels


def train(npz_paths: list[str], epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """
    完整训练流程 — 从加载数据到保存模型。

    Args:
        npz_paths: .npz 样本文件路径列表 (支持多个文件合并训练)
        epochs: 训练轮数 (默认100)
        batch_size: 每批样本数 (默认16)

    Returns:
        (model, history):
          - model: 训练好的 Keras 模型
          - history: 训练历史 (包含每个 epoch 的 loss 和 accuracy)
    """

    # ==================================================================
    # 第 1 步: 加载数据
    # ==================================================================
    if len(npz_paths) == 1:
        # 单文件: 直接加载
        samples, labels = load_npz(npz_paths[0])
    else:
        # 多文件: 合并加载
        print(f"合并 {len(npz_paths)} 个文件:")
        samples, labels = load_multiple_npz(npz_paths)

    if labels is None:
        print("错误: 数据没有标签，无法训练。请先用 --label 生成带标签的样本")
        sys.exit(1)

    print(f"样本形状: {samples.shape}  标签: {len(labels)}")

    # ==================================================================
    # 第 2 步: 标准化
    # ==================================================================
    # 把数据缩放到均值≈0、标准差≈1 的范围
    # norm_stats 保存了 mean 和 std，推理时需要用同样的参数
    samples, norm_stats = normalize(samples)

    # ==================================================================
    # 第 3 步: 划分训练集和验证集
    # ==================================================================
    # 80% 训练，20% 验证
    # 分层抽样保证每个类别在两个集合中的比例一致
    x_train, x_val, y_train, y_val = train_val_split(samples, labels)

    # ==================================================================
    # 第 4 步: 创建 tf.data.Dataset
    # ==================================================================
    # 训练集: 打乱顺序 + 数据增强 (增加样本多样性)
    # 验证集: 不打乱、不增强 (保证评估结果可复现)
    train_ds = make_tf_dataset(x_train, y_train, batch_size,
                               shuffle=True, augment=True)
    val_ds = make_tf_dataset(x_val, y_val, batch_size,
                             shuffle=False, augment=False)

    # ==================================================================
    # 第 5 步: 构建并编译模型
    # ==================================================================
    # build_model() 创建 CNN 网络结构
    # compile_model() 配置优化器、损失函数、评估指标
    model = build_model()
    model = compile_model(model, learning_rate=LEARNING_RATE)

    # 打印模型结构摘要 (方便检查)
    # 会显示每一层的名称、输出形状、参数数量
    model.summary()

    # ==================================================================
    # 第 6 步: 设置回调函数 (Callbacks)
    # ==================================================================
    # 回调函数 = 训练过程中自动执行的操作
    # 它们不影响训练算法本身，但能帮我们更好地控制训练过程

    MODEL_DIR.mkdir(exist_ok=True)

    callbacks = [
        # ── 回调 1: EarlyStopping (早停) ──────────────────────────
        # 监控验证集的 val_loss
        # 如果连续 PATIENCE (15) 个 epoch val_loss 都没下降，就停止训练
        # restore_best_weights=True: 停止后恢复到 val_loss 最低的那次权重
        #
        # 为什么需要? 防止训练太久导致过拟合
        # 如果 val_loss 已经不再下降了，继续训练也没意义
        keras.callbacks.EarlyStopping(
            monitor="val_loss",         # 监控指标: 验证集损失
            patience=PATIENCE,          # 耐心值: 连续多少轮不改善就停止
            restore_best_weights=True,  # 恢复最佳权重
            verbose=1,                  # 打印停止信息
        ),

        # ── 回调 2: ModelCheckpoint (模型保存) ─────────────────────
        # 每当 val_accuracy 创新高，就把模型保存到 best.keras
        # save_best_only=True: 只保存最好的，不保存中间的
        #
        # 为什么需要? 训练过程中模型参数一直在变化，
        # 我们想要的是效果最好的那一次，不是最后一次
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best.keras"),  # 保存路径
            monitor="val_accuracy",                  # 监控指标
            save_best_only=True,                     # 只保存最好的
            verbose=1,
        ),

        # ── 回调 3: ReduceLROnPlateau (学习率衰减) ─────────────────
        # 如果 val_loss 连续 5 个 epoch 没下降，就把学习率减半
        # factor=0.5: 学习率乘以 0.5 (如 0.001 → 0.0005)
        #
        # 为什么需要? 训练后期，loss 可能卡住不再下降
        # 减小学习率可以让模型在最优解附近"精细搜索"
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",    # 监控指标
            factor=0.5,            # 衰减因子
            patience=5,            # 耐心值
            verbose=1,
        ),
    ]

    # ==================================================================
    # 第 7 步: 开始训练!
    # ==================================================================
    # model.fit() 是训练的核心函数
    # 它会:
    #   1. 从 train_ds 中取一个 batch (16 个样本)
    #   2. 用模型对这 16 个样本做预测
    #   3. 计算预测和真实标签之间的 loss
    #   4. 用梯度下降法调整模型参数，使 loss 降低
    #   5. 重复 1-4，直到所有 batch 都过一遍 = 1 个 epoch
    #   6. 在验证集上评估效果
    #   7. 重复 1-6，直到 epochs 轮结束 (或早停触发)
    #
    # 训练过程中你会看到:
    #   Epoch 1/100
    #   17/17 [====] - loss: 1.50 - accuracy: 0.26 - val_loss: 1.39 - val_accuracy: 0.26
    #
    #   loss 越小越好，accuracy 越大越好
    #   val_ 开头的是验证集指标，应该和训练集指标接近
    history = model.fit(
        train_ds,                  # 训练数据
        validation_data=val_ds,    # 验证数据
        epochs=epochs,             # 训练轮数
        callbacks=callbacks,       # 回调函数列表
    )

    # ==================================================================
    # 第 8 步: 保存元数据 (归一化参数 + 类别名)
    # ==================================================================
    # 推理时需要用到:
    #   - norm_stats: 对新数据做同样的标准化
    #   - class_names: 把数字标签 (0,1,2,3) 转成可读的名字
    #   - input_shape: 检查输入形状是否正确
    #   - num_classes: 检查类别数是否正确
    meta = {
        "norm_stats": norm_stats,          # {"mean": [...], "std": [...]}
        "class_names": CLASS_NAMES,        # ["idle", "normal", "loose", "imbalance"]
        "input_shape": list(model.input_shape[1:]),  # [16, 512, 3]
        "num_classes": NUM_CLASSES,        # 4
    }
    meta_path = MODEL_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"元数据已保存: {meta_path}")

    # ==================================================================
    # 第 9 步: 保存最终模型
    # ==================================================================
    # best.keras 已经在训练过程中保存了 (val_accuracy 最高的那次)
    # 这里再保存一份 final.keras (训练结束时的模型)
    # 一般用 best.keras 就好
    model.save(str(MODEL_DIR / "final.keras"))
    print(f"模型已保存: {MODEL_DIR}")

    # ==================================================================
    # 第 10 步: 训练可视化
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


def main():
    """
    命令行入口。

    支持的参数:
      --data: .npz 文件路径，支持多个文件或通配符 (不指定则自动查找 output/ 目录)
      --epochs: 训练轮数 (默认100)
      --batch-size: 每批样本数 (默认16)

    示例:
      python -m src.cnn.train                                        # 自动找 output/*.npz
      python -m src.cnn.train --data output/a.npz                    # 单个文件
      python -m src.cnn.train --data output/a.npz output/b.npz       # 多个文件合并
      python -m src.cnn.train --data output/*.npz                    # 通配符
      python -m src.cnn.train --epochs 200 --batch-size 32
    """
    # argparse 不直接支持 nargs='+' 和可选参数的组合
    # 手动解析: --data 后面所有非 -- 开头的参数都当作文件路径
    npz_paths = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--data":
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                npz_paths.append(sys.argv[i])
                i += 1
            continue
        i += 1

    # 解析其他参数
    parser = argparse.ArgumentParser(description="训练 IMU CNN 模型")
    parser.add_argument("--data", type=str, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="训练轮数 (默认: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="每批样本数 (默认: %(default)s)")
    args = parser.parse_args()

    # 确定数据文件路径
    if not npz_paths:
        # 没指定路径，自动在 output/ 目录下找
        found = find_npz(OUTPUT_DIR)
        if not found:
            print(f"未找到 npz 文件，请先运行 python -m src.data.process 生成样本")
            sys.exit(1)
        npz_paths = [str(p) for p in found]

    print(f"数据文件: {len(npz_paths)} 个")
    for p in npz_paths:
        print(f"  {p}")

    # 开始训练
    train(npz_paths, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
