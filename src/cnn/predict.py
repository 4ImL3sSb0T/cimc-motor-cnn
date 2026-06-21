"""
CNN 推理脚本 — 用训练好的模型对新数据做分类预测

=== 用法 ===
  python -m src.cnn.predict                                          # 自动找数据和模型
  python -m src.cnn.predict --data data/imu_new.csv                  # 指定数据文件
  python -m src.cnn.predict --model models/best.keras                # 指定模型
  python -m src.cnn.predict --data data/a.csv --model models/best.keras
  python -m src.cnn.predict --output result.csv                      # 保存结果到 CSV
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
    DC_OFFSET_SAMPLES, DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    CLASS_NAMES, CNN_SAMPLE_FRAMES, SP_HOP_SIZE, SP_FFT_SIZE, SP_SAMPLE_RATE,
)
from src.data.data_loader import load_data, find_data_file, remove_dc_offset
from src.data.fft_processor import process_3axis
from src.data.sample_generator import generate_samples

# 中文字体
_CN_FONT = None
for _name in ["Microsoft YaHei", "SimHei", "NSimSun"]:
    _matches = [f for f in fm.fontManager.ttflist if f.name == _name]
    if _matches:
        _CN_FONT = fm.FontProperties(fname=_matches[0].fname)
        break
if _CN_FONT is None:
    for _fname in ["msyh.ttc", "simhei.ttf"]:
        _fpath = Path("/mnt/c/Windows/Fonts") / _fname
        if _fpath.exists():
            fm.fontManager.addfont(str(_fpath))
            _CN_FONT = fm.FontProperties(fname=str(_fpath))
            break
if _CN_FONT is None:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    _CN_FONT = fm.FontProperties()


def predict(
    data_path: str,
    model_path: str = str(MODEL_DIR / "best.keras"),
    output_path: str | None = None,
):
    """
    对数据文件运行推理，输出每个样本的分类结果。

    Args:
        data_path: 输入数据文件 (xlsx/csv)
        model_path: 训练好的 .keras 模型路径
        output_path: 结果 CSV 保存路径 (可选)
    """
    data_path = Path(data_path)
    model_path = Path(model_path)

    # ── 1. 加载模型和元数据 ──
    if not model_path.exists():
        print(f"错误: 模型不存在: {model_path}")
        print("请先运行 python -m src.cnn.train 训练模型")
        sys.exit(1)

    meta_path = model_path.parent / "meta.json"
    if not meta_path.exists():
        print(f"错误: 元数据不存在: {meta_path}")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    model = keras.models.load_model(str(model_path))
    class_names = meta["class_names"]
    norm_stats = meta["norm_stats"]
    n_classes = meta["num_classes"]

    print(f"模型: {model_path.name}")
    print(f"类别: {class_names}")
    print(f"输入: {meta['input_shape']}")

    # ── 2. 加载数据 ──
    print(f"\n读取: {data_path.name}")
    ax_raw, ay_raw, az_raw = load_data(data_path)
    print(f"总采样数: {len(ax_raw)}")

    # ── 3. 去直流偏移 ──
    ax, ay, az = remove_dc_offset(ax_raw, ay_raw, az_raw, static_n=DC_OFFSET_SAMPLES)

    # ── 4. FFT 处理 ──
    result = process_3axis(ax, ay, az)

    # ── 5. 生成 CNN 样本 ──
    spectrograms = {
        "x": result["x"][1],
        "y": result["y"][1],
        "z": result["z"][1],
    }
    samples = generate_samples(spectrograms)  # (N, 3, 16, 512)

    # 转置: (N, 3, 16, 512) → (N, 16, 512, 3) channels_last
    samples = np.transpose(samples, (0, 2, 3, 1)).astype(np.float32)

    # ── 6. 归一化 (使用训练时的 mean/std) ──
    mean = np.array(norm_stats["mean"]).reshape(1, 1, 1, 3)
    std = np.array(norm_stats["std"]).reshape(1, 1, 1, 3)
    samples_norm = (samples - mean) / std

    # ── 7. 推理 ──
    print(f"\n推理中... ({len(samples_norm)} 个样本)")
    probs = model.predict(samples_norm, verbose=0)  # (N, n_classes)
    pred_cls = np.argmax(probs, axis=1)             # (N,)
    confidence = probs.max(axis=1)                   # (N,)

    # ── 8. 输出结果 ──
    print(f"\n{'='*60}")
    print(f"推理结果")
    print(f"{'='*60}")

    # 统计各类别
    for cls_idx, cls_name in enumerate(class_names):
        count = (pred_cls == cls_idx).sum()
        pct = count / len(pred_cls) * 100
        print(f"  {cls_name}: {count:>5d} ({pct:5.1f}%)")

    print(f"\n平均置信度: {confidence.mean():.4f}")
    print(f"最低置信度: {confidence.min():.4f}")

    # 低置信度样本警告
    low_conf = confidence < 0.9
    if low_conf.sum() > 0:
        print(f"\n⚠ 低置信度样本 (<90%): {low_conf.sum()} 个")

    # ── 9. 保存 CSV ──
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            # 写表头
            header = "sample_idx,predicted_class,confidence"
            for name in class_names:
                header += f",prob_{name}"
            f.write(header + "\n")

            # 写数据
            for i in range(len(pred_cls)):
                row = f"{i},{class_names[pred_cls[i]]},{confidence[i]:.6f}"
                for j in range(n_classes):
                    row += f",{probs[i][j]:.6f}"
                f.write(row + "\n")

        print(f"\n结果已保存: {output_path}")

    # ── 10. 可视化 ──
    plot_prediction(pred_cls, probs, confidence, class_names)

    return pred_cls, probs, confidence


def plot_prediction(
    pred_cls: np.ndarray,
    probs: np.ndarray,
    confidence: np.ndarray,
    class_names: list[str],
):
    """绘制推理结果图"""
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("推理结果", fontproperties=_CN_FONT, fontsize=14)

    # ── 类别分布饼图 ──
    counts = [(pred_cls == i).sum() for i in range(n_classes)]
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    axes[0].pie(counts, labels=class_names, autopct="%1.1f%%",
                colors=colors, textprops={"fontproperties": _CN_FONT})
    axes[0].set_title("类别分布", fontproperties=_CN_FONT)

    # ── 置信度分布直方图 ──
    axes[1].hist(confidence, bins=20, edgecolor="black", alpha=0.7)
    axes[1].axvline(0.9, color="r", linestyle="--", label="0.9 阈值")
    axes[1].set_xlabel("置信度", fontproperties=_CN_FONT)
    axes[1].set_ylabel("样本数", fontproperties=_CN_FONT)
    axes[1].set_title("置信度分布", fontproperties=_CN_FONT)
    axes[1].legend(prop=_CN_FONT)

    # ── 时间轴上的分类结果 ──
    sample_times = np.arange(len(pred_cls)) * (SP_HOP_SIZE / SP_SAMPLE_RATE) + \
                   (CNN_SAMPLE_FRAMES / 2 * SP_HOP_SIZE + SP_FFT_SIZE / 2) / SP_SAMPLE_RATE
    for cls_idx, cls_name in enumerate(class_names):
        mask = pred_cls == cls_idx
        axes[2].scatter(sample_times[mask], [cls_idx] * mask.sum(),
                       s=3, alpha=0.6, label=cls_name, color=colors[cls_idx])
    axes[2].set_xlabel("时间 (s)", fontproperties=_CN_FONT)
    axes[2].set_yticks(range(n_classes))
    axes[2].set_yticklabels(class_names, fontproperties=_CN_FONT, fontsize=9)
    axes[2].set_title("时间轴分类结果", fontproperties=_CN_FONT)

    plt.tight_layout()
    save_path = MODEL_DIR / "prediction_result.png"
    fig.savefig(str(save_path), dpi=150)
    print(f"结果图已保存: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="CNN 推理")
    parser.add_argument("--data", type=str, default=None,
                        help="数据文件路径 (xlsx/csv)")
    parser.add_argument("--model", type=str, default=str(MODEL_DIR / "best.keras"),
                        help="模型路径 (默认: models/best.keras)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="结果 CSV 保存路径")
    args = parser.parse_args()

    # 确定数据文件
    data_path = args.data
    if data_path is None:
        data_path = find_data_file(DATA_DIR)
        if data_path is None:
            print(f"未找到数据文件，请用 --data 指定")
            sys.exit(1)
        data_path = str(data_path)

    predict(data_path, model_path=args.model, output_path=args.output)


if __name__ == "__main__":
    main()
