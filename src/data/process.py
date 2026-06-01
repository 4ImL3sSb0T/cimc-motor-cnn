"""
IMU 加速度数据 FFT 频谱分析 + CNN 样本生成

用法:
  python -m src.data.process                              # 自动查找 data/ 下第一个文件
  python -m src.data.process --data data/imu_test.csv     # 指定数据文件
  python -m src.data.process --label data/labels.json     # 带标签生成
  python -m src.data.process --data data/a.csv --label data/labels.json
  python -m src.data.process --viewer                     # 交互式 CNN 样本查看器
  python -m src.data.process --static                     # 静态一次性 FFT 分析图
"""

import json
import sys
from pathlib import Path

from src.config import DC_OFFSET_SAMPLES, DATA_DIR, OUTPUT_DIR
from src.data.data_loader import load_data, find_data_file, remove_dc_offset
from src.data.fft_processor import process_3axis
from src.data.sample_generator import generate_samples, generate_labels, filter_labeled, save_samples
from src.visualizer import plot_fft_analysis, CNNSampleViewer


def resolve_data_path(data_path: str | None) -> Path | None:
    """解析数据文件路径: 指定路径直接用，否则自动查找"""
    if data_path:
        p = Path(data_path)
        if not p.exists():
            print(f"错误: 数据文件不存在: {p}")
            return None
        return p
    found = find_data_file(DATA_DIR)
    if not found:
        print(f"{DATA_DIR}/ 目录下没有找到 xlsx/csv 文件")
    return found


def run_pipeline(data_path: str | None = None, label_config: dict | None = None):
    """默认模式: 加载数据 → FFT → 生成 CNN 样本 → 保存 .npz"""
    data_path = resolve_data_path(data_path)
    if not data_path:
        return

    # 1. 加载原始数据
    print(f"读取: {data_path.name}")
    ax_raw, ay_raw, az_raw = load_data(data_path)
    print(f"总采样数: {len(ax_raw)}")

    # 2. 去直流偏移 (前 10000 行静态数据)
    ax, ay, az = remove_dc_offset(ax_raw, ay_raw, az_raw, static_n=DC_OFFSET_SAMPLES)

    # 3. FFT 处理
    result = process_3axis(ax, ay, az)

    # 4. 生成 CNN 样本
    spectrograms = {
        "x": result["x"][1],
        "y": result["y"][1],
        "z": result["z"][1],
    }
    samples = generate_samples(spectrograms)

    # 5. 生成标签 (如果提供了配置)
    labels = None
    if label_config is not None:
        labels = generate_labels(len(samples), label_config)
        samples, labels = filter_labeled(samples, labels)

    # 6. 保存
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{data_path.stem}_samples.npz"
    save_samples(samples, out_path, labels=labels)

    return samples


def run_viewer(data_path: str | None = None):
    """交互式 CNN 样本查看器"""
    data_path = resolve_data_path(data_path)
    if not data_path:
        return

    print(f"读取: {data_path.name}")
    ax_raw, ay_raw, az_raw = load_data(data_path)
    print(f"总采样数: {len(ax_raw)}")

    ax, ay, az = remove_dc_offset(ax_raw, ay_raw, az_raw, static_n=DC_OFFSET_SAMPLES)
    result = process_3axis(ax, ay, az)

    viewer = CNNSampleViewer()
    viewer.load_data(result)
    viewer.setup()
    viewer.show()

    sample = viewer.get_current_sample()
    print(f"\n当前样本 shape: {sample.shape}  dtype: {sample.dtype}")
    print(f"值范围: [{sample.min():.4f}, {sample.max():.4f}]")


def run_static(data_path: str | None = None):
    """静态 FFT 分析图"""
    data_path = resolve_data_path(data_path)
    if not data_path:
        return

    print(f"读取: {data_path.name}")
    ax_raw, ay_raw, az_raw = load_data(data_path)
    print(f"总采样数: {len(ax_raw)}")

    ax, ay, az = remove_dc_offset(ax_raw, ay_raw, az_raw, static_n=DC_OFFSET_SAMPLES)
    result = process_3axis(ax, ay, az)

    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_signals = {"x": ax, "y": ay, "z": az}
    plot_fft_analysis(
        result, raw_signals=raw_signals,
        save_path=OUTPUT_DIR / f"{data_path.stem}_fft_analysis.png",
    )


def main():
    # 解析公共参数
    data_path = None
    if "--data" in sys.argv:
        idx = sys.argv.index("--data")
        if idx + 1 >= len(sys.argv):
            print("错误: --data 需要指定数据文件路径")
            sys.exit(1)
        data_path = sys.argv[idx + 1]

    if "--viewer" in sys.argv:
        run_viewer(data_path=data_path)
    elif "--static" in sys.argv:
        run_static(data_path=data_path)
    else:
        # 解析 --label 参数
        label_config = None
        if "--label" in sys.argv:
            idx = sys.argv.index("--label")
            if idx + 1 >= len(sys.argv):
                print("错误: --label 需要指定 JSON 配置文件路径")
                sys.exit(1)
            label_path = Path(sys.argv[idx + 1])
            if not label_path.exists():
                print(f"错误: 标签配置文件不存在: {label_path}")
                sys.exit(1)
            with open(label_path, "r", encoding="utf-8") as f:
                label_config = json.load(f)
            print(f"标签配置: {label_path}")

        run_pipeline(data_path=data_path, label_config=label_config)


if __name__ == "__main__":
    main()
