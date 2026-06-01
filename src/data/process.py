"""
IMU 加速度数据 FFT 频谱分析 + CNN 样本生成

用法:
  python -m src.data.process              # 生成 CNN 样本并保存 .npz
  python -m src.data.process --viewer     # 交互式 CNN 样本查看器
  python -m src.data.process --static     # 静态一次性 FFT 分析图
"""

import sys
from pathlib import Path

from src.config import DC_OFFSET_SAMPLES, DATA_DIR, OUTPUT_DIR
from src.data.data_loader import load_xlsx, find_xlsx, remove_dc_offset
from src.data.fft_processor import process_3axis
from src.data.sample_generator import generate_samples, save_samples
from src.visualizer import plot_fft_analysis, CNNSampleViewer


def run_pipeline():
    """默认模式: 加载数据 → FFT → 生成 CNN 样本 → 保存 .npz"""
    xlsx_path = find_xlsx(DATA_DIR)
    if not xlsx_path:
        print(f"{DATA_DIR}/ 目录下没有找到 xlsx 文件")
        return

    # 1. 加载原始数据
    print(f"读取: {xlsx_path.name}")
    ax_raw, ay_raw, az_raw = load_xlsx(xlsx_path)
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

    # 5. 保存
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{xlsx_path.stem}_samples.npz"
    save_samples(samples, out_path)

    return samples


def run_viewer():
    """交互式 CNN 样本查看器"""
    xlsx_path = find_xlsx(DATA_DIR)
    if not xlsx_path:
        print(f"{DATA_DIR}/ 目录下没有找到 xlsx 文件")
        return

    print(f"读取: {xlsx_path.name}")
    ax_raw, ay_raw, az_raw = load_xlsx(xlsx_path)
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


def run_static():
    """静态 FFT 分析图"""
    xlsx_path = find_xlsx(DATA_DIR)
    if not xlsx_path:
        print(f"{DATA_DIR}/ 目录下没有找到 xlsx 文件")
        return

    print(f"读取: {xlsx_path.name}")
    ax_raw, ay_raw, az_raw = load_xlsx(xlsx_path)
    print(f"总采样数: {len(ax_raw)}")

    ax, ay, az = remove_dc_offset(ax_raw, ay_raw, az_raw, static_n=DC_OFFSET_SAMPLES)
    result = process_3axis(ax, ay, az)

    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_signals = {"x": ax, "y": ay, "z": az}
    plot_fft_analysis(
        result, raw_signals=raw_signals,
        save_path=OUTPUT_DIR / "fft_analysis.png",
    )


def main():
    if "--viewer" in sys.argv:
        run_viewer()
    elif "--static" in sys.argv:
        run_static()
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
