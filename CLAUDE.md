# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMU 加速度数据的 FFT 频谱分析 + CNN 神经网络分类项目，用于电机振动检测。
数据流程: xlsx/csv 原始数据 → 去直流偏移 → 滑动窗口 FFT → 频谱图 → CNN 分类 → TFLite 导出 (ESP32 部署)

## Project Structure

```
src/
├── config.py                  # 全局参数 (FFT + CNN)
├── data/
│   ├── data_loader.py         # xlsx/csv 加载 + 去直流偏移
│   ├── fft_processor.py       # 滑动窗口 FFT
│   ├── sample_generator.py    # CNN 样本生成 + 标签生成 + 保存/加载
│   └── process.py             # 数据处理入口
├── cnn/
│   ├── model.py               # CNN 模型定义 (11,012 参数)
│   ├── dataset.py             # 数据加载 + 归一化 + 增强
│   ├── train.py               # 训练脚本
│   └── export.py              # TFLite/ONNX 导出
└── visualizer.py              # FFT 分析图 + CNN 样本查看器
tcp_receiver.py                # ESP32 IMU 数据 TCP 接收器
data/                          # 原始数据 (xlsx/csv) + 标签配置 (json)
output/                        # 生成的 .npz 样本
models/                        # 训练好的模型
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

# 数据处理 (xlsx/csv → FFT → .npz 样本)
python -m src.data.process

# 带标签的数据处理 (需要先创建 data/labels.json)
python -m src.data.process --label data/labels.json

# CNN 样本查看器 (查看频谱图 + 时间分布)
python -m src.data.process --viewer

# 静态 FFT 分析图
python -m src.data.process --static

# 训练 CNN 模型
python -m src.cnn.train

# 导出 TFLite / ONNX 模型
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
        → process_3axis()       滑动窗口 FFT (1024点, hop=256, Hann窗)
        → generate_samples()    窗口 16 帧提取 CNN 样本
        → generate_labels()     按 JSON 配置映射时间段→类别 (可选)
        → save_samples()        保存到 output/*.npz

训练: output/*_samples.npz
        → load_npz()            加载 + 转置 (N,3,16,512)→(N,16,512,3)
        → normalize()           逐通道标准化 (减均值/除std)
        → train_val_split()     分层抽样 80%/20%
        → model.fit()           CNN 训练, EarlyStopping + ModelCheckpoint
        → 保存 models/best.keras + meta.json

导出: models/best.keras
        → model_float32.tflite  不量化, PC 测试
        → model_int8.tflite     全量化, ESP32 部署 (~11KB)
        → model.onnx            通用格式
```

## Label Config Format

标注配置文件放在 `data/` 目录下，格式:

```json
{
  "default_class": "other",
  "labels": [
    {"start": 0.0, "end": 5.0, "class": "idle"},
    {"start": 5.0, "end": 10.0, "class": "vibration"},
    {"start": 10.0, "end": 13.5, "class": "impact"}
  ]
}
```

- `default_class`: 未覆盖时间段的默认类别 (可选, 默认 "other")
- `labels[].class`: 必须是 `CLASS_NAMES` 中定义的: `idle`, `vibration`, `impact`, `other`
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
| 类别 | idle/vibration/impact/other | 4 分类 |
| 模型参数 | 11,012 (43KB) | 轻量级, 可装入 ESP32-S3 |
