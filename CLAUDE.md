# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMU 加速度数据的 FFT 频谱分析 + CNN 神经网络分类项目，用于电机振动检测。
数据流程: xlsx/csv 原始数据 → 去直流偏移 → 滑动窗口 FFT (4通道: X/Y/Z/magnitude) → 频谱图 → CNN 分类 → ONNX 导出

## Project Structure

```
src/
├── __init__.py
├── config.py                  # 全局参数 (FFT + CNN)
├── data/
│   ├── __init__.py
│   ├── data_loader.py         # xlsx/csv 加载 + 去直流偏移
│   ├── fft_processor.py       # 滑动窗口 FFT
│   ├── sample_generator.py    # CNN 样本生成 + 标签生成 + 保存/加载
│   └── process.py             # 数据处理入口
├── cnn/
│   ├── __init__.py
│   ├── model.py               # CNN 模型定义 (35,850 参数, v2)
│   ├── dataset.py             # 数据加载 + 归一化 + 增强
│   ├── train.py               # 训练脚本
│   ├── predict.py             # 推理脚本
│   └── export.py              # ONNX 导出
└── visualizer.py              # FFT 分析图 + CNN 样本查看器
tcp_receiver.py                # ESP32 IMU 数据 TCP 接收器
verify_model.py                # 验证 v1/v2 模型结构与参数
data/                          # 原始数据 (csv) + 标签配置 (json)
output/                        # 生成的 .npz 样本 + FFT 分析图
models/                        # 训练好的模型 (.keras/.h5/.onnx) + 结果图
.serena/                       # Serena LSP 项目配置
tests/                         # 测试代码 (待补充)
```

## Environment

- **Platform**: WSL2 (Ubuntu)
- **Conda env**: `tf_gpu` at `/home/ws/miniconda3/envs/tf_gpu/`
- **Python**: 3.10.20
- 激活环境: `conda activate tf_gpu`

Key packages:
- `tensorflow` 2.16.2
- `keras` 3.12.1
- `scipy`, `numpy`, `openpyxl`, `matplotlib`

## Common Commands

```bash
conda activate tf_gpu

# 数据处理 (自动查找 data/ 下第一个文件)
python -m src.data.process

# 指定数据文件
python -m src.data.process --data data/imu_test.csv

# 自动匹配同名 json 标签 (如 imu_test.csv → imu_test.json)
python -m src.data.process --data data/imu_test.csv

# 显式指定标签文件 (覆盖自动匹配)
python -m src.data.process --data data/imu_test.csv --label data/other.json

# 批量处理所有 csv 文件
for f in data/*.csv; do python -m src.data.process --data "$f"; done

# CNN 样本查看器 (查看频谱图 + 时间分布)
python -m src.data.process --viewer --data data/imu_test.csv

# 静态 FFT 分析图
python -m src.data.process --static --data data/imu_test.csv

# 训练 CNN 模型 (按文件划分, 消除滑动窗口数据泄漏)
python -m src.cnn.train

# 使用 Focal Loss (对类别不平衡更鲁棒)
python -m src.cnn.train --focal-loss

# 手动指定验证文件 (覆盖自动划分)
python -m src.cnn.train --val-files output/file_a.npz output/file_b.npz

# 自定义参数
python -m src.cnn.train --epochs 200 --batch-size 32

# 推理 (用训练好的模型预测新数据)
python -m src.cnn.predict --data data/imu_new.csv
python -m src.cnn.predict --data data/imu_new.csv --output result.csv

# 导出 ONNX 模型 (TFLite 因 TF 2.16 bug 无法使用)
python -m src.cnn.export

# ESP32 数据采集
python tcp_receiver.py                          # 默认 192.168.4.1:8080
python tcp_receiver.py --duration 10 -o data/imu_test.csv
```

## Data Flow

```
采集: ESP32 IMU → tcp_receiver.py → data/*.csv
                                     data/*.xlsx (PC端采集)

处理: data/*.xlsx/csv
        → load_data()           自动识别格式, 提取 3 轴加速度
        → remove_dc_offset()    前 10000 行静态数据算均值, 减去传感器偏置
        → process_4axis()       滑动窗口 FFT (1024点, hop=256, Hann窗)
                                4 通道: X/Y/Z + magnitude(sqrt(x²+y²+z²))
        → generate_samples()    窗口 16 帧提取 CNN 样本
        → generate_labels()     按 JSON 配置映射时间段→类别 (可选)
        → save_samples()        保存到 output/*.npz

训练: output/*_samples.npz
        → load_npz()            逐文件加载 (保留文件边界)
        → train_val_split_by_file()  按采集文件划分 train/val (消除滑动窗口数据泄漏)
        → normalize()           stats 仅从训练集计算, 验证集复用 (杜绝数据泄漏)
        → model.fit()           CNN 训练, class_weight + EarlyStopping + ModelCheckpoint
        → classification_report 输出 precision/recall/f1 (不依赖 sklearn)
        → 保存 models/best.keras + meta.json

        单文件时自动回退: train_val_split() 逐样本分层划分

导出: models/best.keras
        → model.onnx            ONNX 通用格式 (45.9KB)
        注: TFLite 导出因 TF 2.16.x MLIR bug 不可用 (tensorflow#63987)
```

## Label Config Format

标注配置文件放在 `data/` 目录下，格式:

```json
{
  "default_class": "other",
  "labels": [
    {"start": 0.0, "end": 5.0, "class": "idle"},
    {"start": 5.0, "end": 10.0, "class": "normal"},
    {"start": 10.0, "end": 13.5, "class": "loose"}
  ]
}
```

- `default_class`: 未覆盖时间段的默认类别 (可选, 默认 "other")
- `labels[].class`: 必须是 `CLASS_NAMES` 中定义的: `idle`, `normal`, `loose`, `imbalance`
- 时间单位: 秒, 基于 FFT 帧中心时间

## Key Parameters

| 参数 | 值 | 含义 |
|---|---|---|
| 采样率 | 6667 Hz | IMU 每秒采样数 |
| FFT 窗口 | 1024 点 (0.15s) | 单次 FFT 的采样长度 |
| FFT 跳步 | 256 点 (0.038s) | 相邻帧间距, 75% 重叠 |
| 频率分辨率 | 6.51 Hz/bin | 每个频率 bin 的跨度 |
| CNN 窗口 | 16 帧 (0.61s) | 每个样本覆盖的时间 |
| CNN stride | 1 帧 (0.038s) | 相邻样本中心间距 |
| 输入通道 | 4 (X/Y/Z/magnitude) | 三轴加速度 + 合加速度 sqrt(x²+y²+z²) |
| 类别 | idle/normal/loose/imbalance | 4 分类 |
| 频率 bin | 128 (0-833 Hz) | CNN 实际使用的频段，裁剪掉 >833Hz 噪声 |
| 模型参数 | 35,990 (140KB) | 残差+SE注意力+SpatialDropout, v2 默认 |
