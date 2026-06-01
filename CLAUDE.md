# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IMU 加速度数据的 FFT 频谱分析 + CNN 神经网络分类项目。
数据流程: xlsx 原始数据 → 去直流偏移 → 滑动窗口 FFT → 频谱图 → CNN 分类

## Project Structure

```
src/
├── config.py                  # 全局参数 (FFT + CNN)
├── data/
│   ├── data_loader.py         # xlsx 加载 + 去直流偏移
│   ├── fft_processor.py       # 滑动窗口 FFT
│   ├── sample_generator.py    # CNN 样本生成/保存/加载
│   └── process.py             # 数据处理入口
├── cnn/
│   ├── model.py               # CNN 模型定义
│   ├── dataset.py             # 数据加载 + 归一化 + 增强
│   ├── train.py               # 训练脚本
│   └── export.py              # TFLite/ONNX 导出
└── visualizer.py              # 可视化
data/                          # 原始 xlsx 数据
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

# 数据处理 (xlsx → FFT → .npz 样本)
python -m src.data.process

# CNN 样本查看器
python -m src.data.process --viewer

# 静态 FFT 分析图
python -m src.data.process --static

# 训练 CNN 模型
python -m src.cnn.train

# 导出 TFLite / ONNX 模型
python -m src.cnn.export

# 运行测试
python -m pytest tests/
```
