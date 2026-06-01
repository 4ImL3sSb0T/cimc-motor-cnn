# IMU 振动频谱分类 — CNN 神经网络项目

> 采集 IMU 加速度数据 → FFT 频谱分析 → CNN 分类 → ESP32-S3 部署

---

## 目录

- [项目概述](#项目概述)
- [数据流程](#数据流程)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [模块详解](#模块详解)
- [CNN 模型架构](#cnn-模型架构)
- [训练指南](#训练指南)
- [模型导出与部署](#模型导出与部署)
- [关键参数速查](#关键参数速查)
- [常见问题](#常见问题)

---

## 项目概述

本项目通过 IMU（惯性测量单元）采集电机/设备的三轴加速度信号，经 FFT 频谱变换后生成频谱图，再用轻量级 CNN 神经网络对运动模式进行分类。

**应用场景**: 电机状态监测、振动故障检测、运动模式识别

**最终目标**: 在 PC 上训练模型，导出量化后的 TFLite int8 模型，部署到 ESP32-S3 嵌入式设备上做实时推理。

---

## 数据流程

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐
│  xlsx/csv   │───▶│  去直流偏移   │───▶│  滑动窗口 FFT  │───▶│  CNN 样本    │───▶│  训练 / 导出  │
│  加速度数据  │    │  (前1万行)    │    │  (1024点/256跳)│    │  (3,16,512)  │    │  TFLite int8 │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘    └──────────────┘
```

### 各阶段说明

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| **数据采集** | ESP32 IMU | `data/*.csv` | `tcp_receiver.py` 通过 TCP 实时录制 |
| **数据加载** | `data/*.xlsx` 或 `*.csv` | 3 个 float32 数组 | 自动识别格式，读取 X/Y/Z 三轴加速度 |
| **去直流偏移** | 原始信号 | 交流分量 | 用前 10000 行静置数据计算 DC 偏移并减去 |
| **FFT 处理** | 交流分量 | 频谱图 `(n_frames, 512)` | Hann 窗 → FFT → 取幅度 → 转 dB |
| **样本生成** | 3 轴频谱图 | `(N, 3, 16, 512)` | 滑动窗口取 16 帧，3 通道拼接 |
| **标签生成** | JSON 配置 | `(N,)` int32 | 按时间段映射类别（可选） |
| **训练** | `.npz` 样本 | `.keras` 模型 | CNN 分类训练 |
| **导出** | `.keras` 模型 | `.tflite` / `.onnx` | int8 量化，部署到 ESP32 |

---

## 项目结构

```
tensorflow/
├── README.md                          # 本文件
├── CLAUDE.md                          # AI 辅助开发指引
├── tcp_receiver.py                    # ESP32 IMU 数据 TCP 接收器
├── data/                              # 原始数据 + 标签配置
│   ├── plotter-20260531-205012.xlsx   # PC 端采集的数据
│   ├── imu_test.csv                   # ESP32 TCP 采集的数据
│   ├── labels.json                    # 标签配置 (时间段→类别)
│   └── labels_example.json            # 标签配置示例
├── output/                            # 生成的 .npz 样本文件
│   └── plotter-20260531-205012_samples.npz
├── models/                            # 训练好的模型
│   ├── best.keras                     # 最佳验证精度模型
│   ├── final.keras                    # 最终模型
│   ├── meta.json                      # 归一化参数 + 类别名
│   ├── model_int8.tflite              # int8 量化 TFLite (ESP32 部署)
│   ├── model_float32.tflite           # float32 TFLite
│   └── model.onnx                     # ONNX 格式
├── src/
│   ├── config.py                      # 全局配置 (FFT参数/CNN参数/路径)
│   ├── visualizer.py                  # FFT 分析图 + CNN 样本查看器
│   ├── data/                          # ── 数据处理模块 ──
│   │   ├── data_loader.py             # xlsx/csv 加载 + 去直流偏移
│   │   ├── fft_processor.py           # 滑动窗口 FFT 处理
│   │   ├── sample_generator.py        # CNN 样本生成 + 标签生成 + 保存
│   │   └── process.py                 # 数据处理入口脚本
│   └── cnn/                           # ── CNN 模块 ──
│       ├── model.py                   # CNN 模型定义 (11,012 参数)
│       ├── dataset.py                 # 数据加载 + 归一化 + 增强
│       ├── train.py                   # 训练脚本
│       └── export.py                  # TFLite/ONNX 导出
└── tests/                             # 测试代码
```

---

## 环境配置

```bash
# 平台: WSL2 (Ubuntu)
# Conda 环境
conda activate tf_gpu

# 核心依赖
# tensorflow 2.16.2  (含 GPU 支持)
# keras 3.12.1
# scipy, numpy, openpyxl, matplotlib
```

---

## 快速开始

### 1. 数据采集（从 ESP32）

```bash
# 连接 ESP32 WiFi 热点后，录制 10 秒数据
python tcp_receiver.py --duration 10 -o data/imu_test.csv

# 默认连接 192.168.4.1:8080，可自定义
python tcp_receiver.py --ip 192.168.1.100 --port 9090 --duration 30
```

CSV 格式：
```csv
timestamp_us,datetime,acc_x,acc_y,acc_z
1717200000000000,2026-06-01 12:00:00.000000,1.006974,-0.033726,0.017533
...
```

### 2. 数据处理 — 生成 CNN 样本

```bash
# 无标签（演示用）
python -m src.data.process

# 带标签（训练用，需要先创建 data/labels.json）
python -m src.data.process --label data/labels.json
```

输出:
```
读取: imu_test.csv
总采样数: 89604
直流偏移 (前10000行): X=1.006974, Y=-0.033726, Z=0.017533
FFT 参数: size=1024, hop=256, bins=512, df=6.51Hz
  处理 X 轴 (89604 采样)...
  处理 Y 轴 (89604 采样)...
  处理 Z 轴 (89604 采样)...
生成样本: 332 个, shape=(332, 3, 16, 512), dtype=float32, range=[-78.1, 65.6]
标签生成: 332 个样本, 匹配 332, 默认 0
  idle: 121
  vibration: 130
  impact: 81
已保存: output/imu_test_samples.npz (27.9 MB)
```

### 3. 训练 CNN 模型

```bash
python -m src.cnn.train

# 自定义参数
python -m src.cnn.train --data output/xxx_samples.npz --epochs 200 --batch-size 32
```

### 4. 导出部署模型

```bash
python -m src.cnn.export
```

### 其他命令

```bash
# 交互式 CNN 样本查看器 (需要图形界面)
python -m src.data.process --viewer

# 静态 FFT 分析图 (保存为 png)
python -m src.data.process --static

# 查看模型结构
python -m src.cnn.model
```

---

## 模块详解

### `tcp_receiver.py` — ESP32 数据采集

通过 TCP 连接 ESP32，实时接收 IMU 加速度数据并保存为 CSV。

**数据协议**:
- 每个采样点: 12 字节 (3 × float32, 小端序): acc_x, acc_y, acc_z
- 帧分隔符: 4 字节 0xFFFFFFFF (NaN 作为帧尾标记)
- 采样率: 6667 Hz

```bash
python tcp_receiver.py                          # 默认 192.168.4.1:8080
python tcp_receiver.py --ip 192.168.1.100       # 自定义 IP
python tcp_receiver.py --duration 10            # 录制 10 秒后自动停止
python tcp_receiver.py --output data/test.csv   # 指定输出文件
```

### `src/config.py` — 全局配置

所有参数集中管理，修改一处即可影响整个流水线。

```python
# FFT 参数 (与 ESP32 固件对齐)
SP_FFT_SIZE = 1024          # FFT 点数
SP_HOP_SIZE = 256           # 滑动窗口跳步
SP_FREQ_BINS = 512          # 频率 bin 数 (FFT_SIZE / 2)
SP_SAMPLE_RATE = 6667.0     # 采样率 Hz
SP_FREQ_RES = 6.51          # 频率分辨率 Hz

# CNN 输入
CNN_SAMPLE_FRAMES = 16      # 每个样本的时间帧数
CNN_INPUT_SHAPE = (3, 16, 512)  # (通道, 帧, 频率bin)

# 训练
CLASS_NAMES = ["idle", "vibration", "impact", "other"]  # ← 修改为实际类别
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
```

### `src/data/data_loader.py` — 数据加载

- `find_data_file(data_dir)`: 在目录中查找第一个 `.xlsx` 或 `.csv` 文件（优先 xlsx）
- `load_xlsx(path)`: 读取 xlsx，返回 `(ax, ay, az)` 三个 float32 数组
- `load_csv(path)`: 读取 TCP receiver 生成的 CSV，返回 `(ax, ay, az)`
- `load_data(path)`: 自动识别格式并加载
- `remove_dc_offset(ax, ay, az, static_n=10000)`: 用前 N 行静态数据计算直流偏移并减去

支持的数据格式:

**xlsx 格式** (PC 端采集):
```
| PC Date | PC Time | Line 1 (X) | Line 2 (Y) | Line 3 (Z) |
|---------|---------|-------------|-------------|-------------|
| 日期    | 时间    | X轴加速度   | Y轴加速度   | Z轴加速度   |
```

**csv 格式** (ESP32 TCP 采集):
```
timestamp_us,datetime,acc_x,acc_y,acc_z
```

### `src/data/fft_processor.py` — FFT 处理

对应 ESP32 固件 `imu_data_process.c` 的处理逻辑，确保 PC 端和嵌入式端的频谱结果一致。

**单帧处理流程** (`process_frame`):
```
原始帧 (1024点)
  │
  ├─ 1. 去直流偏移 (减均值)
  ├─ 2. 去线性趋势 (减去首尾连线)
  ├─ 3. 乘 Hann 窗
  ├─ 4. FFT (1024点)
  ├─ 5. 取前 512 个频率 bin 的幅度
  └─ 6. 转 dB: 20 * log10(mag + 1e-10)
       │
       ▼
  频谱帧 (512个频率bin)
```

**滑动窗口** (`sliding_window_fft`):
- 窗口大小: 1024 采样点 (约 0.154 秒)
- 跳步大小: 256 采样点 (约 0.038 秒)
- 89604 个采样 → 346 个时间帧

### `src/data/sample_generator.py` — 样本生成 + 标签生成

将 3 轴频谱图组合成 CNN 训练样本:

```
X 轴频谱: (346, 512)  ─┐
Y 轴频谱: (346, 512)  ─┼─ stack ─▶ (332, 3, 16, 512) ─▶ .npz
Z 轴频谱: (346, 512)  ─┘
```

每个样本取连续 16 帧，stride=1 滑动，共生成 332 个样本。

**标签生成** (`generate_labels`):
- 根据 JSON 配置文件，将时间段映射到类别
- CNN 样本 `i` 的中心时间: `((i + 7.5) * 256 + 512) / 6667` 秒

**npz 文件格式**:
- `samples`: shape `(N, 3, 16, 512)`, float32 — CNN 输入
- `labels` (可选): shape `(N,)`, int32 — 分类标签 (0=idle, 1=vibration, 2=impact, 3=other)

### `src/data/process.py` — 数据处理入口

```bash
python -m src.data.process                              # 默认: 生成 .npz 样本
python -m src.data.process --label data/labels.json     # 带标签生成
python -m src.data.process --viewer                     # 交互式查看器 (需 GUI)
python -m src.data.process --static                     # 静态 FFT 分析图
```

---

## CNN 模型架构

### 设计原则

- **轻量级**: 总参数 11,012 (43 KB)，适合 ESP32-S3 部署
- **深度可分离卷积**: 减少计算量和参数量
- **全局平均池化**: 替代大尺寸全连接层
- **BatchNorm + ReLU**: 标准训练范式

### 网络结构

```
输入: (16, 512, 3)  — 16帧 × 512频率bin × 3通道(X/Y/Z)
  │
  ├─ Conv2D(16, 3×3) ─ BN ─ ReLU           # 提取低层特征
  ├─ MaxPool(2, 4)  → (8, 128, 16)         # 降维
  │
  ├─ SeparableConv2D(32, 3×3) ─ BN ─ ReLU  # 深度可分离
  ├─ MaxPool(2, 4)  → (4, 32, 32)
  │
  ├─ SeparableConv2D(64, 3×3) ─ BN ─ ReLU
  ├─ MaxPool(2, 4)  → (2, 8, 64)
  │
  ├─ SeparableConv2D(64, 3×3) ─ BN ─ ReLU
  │
  ├─ GlobalAveragePooling  → (64,)          # 全局特征
  ├─ Dropout(0.3)
  ├─ Dense(32, ReLU)
  ├─ Dropout(0.2)
  └─ Dense(num_classes, Softmax)            # 分类输出
       │
       ▼
输出: (num_classes,)  — 类别概率
```

### 参数量统计

| 层 | 参数量 |
|----|--------|
| Conv2D(16) | 432 |
| SeparableConv2D(32) | 656 |
| SeparableConv2D(64) | 2,336 |
| SeparableConv2D(64) | 4,672 |
| BatchNorm × 4 | 704 |
| Dense(32) | 2,080 |
| Dense(4) | 132 |
| **总计** | **11,012 (43 KB)** |

---

## 训练指南

### 标注数据 — 配置文件方式

在 `data/` 目录下创建 JSON 标签配置文件:

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

- `default_class`: 未覆盖时间段的默认类别（可选，默认 "other"）
- `labels[].class`: 必须是 `CLASS_NAMES` 中定义的类别名
- 时间单位: 秒，基于 FFT 帧中心时间

先用 `--viewer` 查看频谱图确定各状态的时间段，再编写配置文件。

### 修改类别名

编辑 `src/config.py`:

```python
CLASS_NAMES = ["idle", "vibration", "impact", "other"]
NUM_CLASSES = len(CLASS_NAMES)
```

### 训练命令

```bash
# 使用默认参数
python -m src.cnn.train

# 自定义参数
python -m src.cnn.train --data output/xxx_samples.npz --epochs 200 --batch-size 32
```

### 训练回调

- **EarlyStopping**: val_loss 15 轮不下降则停止
- **ModelCheckpoint**: 保存 val_accuracy 最高的模型到 `models/best.keras`
- **ReduceLROnPlateau**: val_loss 5 轮不下降则学习率减半

### 数据增强 (训练时自动启用)

| 增强方式 | 说明 |
|----------|------|
| 时间偏移 | ±2 帧随机偏移 |
| 频率偏移 | ±16 bin 随机偏移 |
| 高斯噪声 | stddev=0.1 |
| SpecAugment | 随机遮蔽 2~4 个连续频率 bin |

---

## 模型导出与部署

```bash
python -m src.cnn.export
```

导出产物:

| 格式 | 文件 | 用途 |
|------|------|------|
| Keras | `models/best.keras` | PC 端继续训练/微调 |
| TFLite f32 | `models/model_float32.tflite` | 测试/验证 |
| TFLite int8 | `models/model_int8.tflite` | **ESP32-S3 部署** |
| ONNX | `models/model.onnx` | 其他推理框架 |

### int8 量化细节

- 使用训练数据的前 200 个样本作为校准集
- 输入/输出均为 int8 类型
- 推理时需要对输入做归一化 (使用 `meta.json` 中的 mean/std)

### `meta.json` 内容

```json
{
  "norm_stats": {
    "mean": [8.60, 4.06, 2.20],
    "std": [18.95, 17.38, 17.35]
  },
  "class_names": ["idle", "vibration", "impact", "other"],
  "input_shape": [16, 512, 3],
  "num_classes": 4
}
```

ESP32 推理时需要:
1. 采集 16 帧 × 1024 点 FFT → 得到 `(3, 16, 512)` 频谱数据
2. 用 `meta.json` 中的 mean/std 做归一化
3. 量化为 int8 送入 TFLite 模型
4. 输出 4 个类别的概率

---

## 关键参数速查

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 6667 Hz | IMU 采样频率 |
| FFT 大小 | 1024 | 约 0.154 秒 |
| 跳步大小 | 256 | 约 0.038 秒 |
| 频率 bin 数 | 512 | 0 ~ 3333 Hz |
| 频率分辨率 | 6.51 Hz | 6667 / 1024 |
| 样本帧数 | 16 | 每个 CNN 输入的时间帧 |
| CNN 输入 | (16, 512, 3) | frames × freq_bins × channels |
| 模型参数 | 11,012 | 43 KB |
| 分类类别 | 4 | idle/vibration/impact/other |

---

## 常见问题

### Q: 如何标注数据?

在 `data/` 目录下创建 JSON 标签配置文件，指定每个时间段对应的类别。先用 `--viewer` 查看频谱图确定时间段，然后运行 `python -m src.data.process --label data/labels.json` 生成带标签的 `.npz`。

### Q: 数据没有标签能训练吗?

可以训练，但会使用随机标签（仅用于演示流程验证）。实际训练必须有真实标签。

### Q: 如何添加新的数据?

- **xlsx**: 把新的 `.xlsx` 文件放入 `data/` 目录
- **csv**: 运行 `python tcp_receiver.py -o data/xxx.csv` 录制

然后重新运行 `python -m src.data.process`。

### Q: 如何修改 FFT 参数?

编辑 `src/config.py` 中的 `SP_FFT_SIZE`、`SP_HOP_SIZE`、`SP_SAMPLE_RATE`。注意这些参数需要与 ESP32 固件保持一致。

### Q: 模型太大/太小怎么办?

编辑 `src/cnn/model.py`，调整卷积层的通道数或增减层数。

### Q: 如何在 ESP32 上使用导出的模型?

1. 将 `model_int8.tflite` 和 `meta.json` 部署到 ESP32
2. 使用 TensorFlow Lite Micro 解释器加载模型
3. 推理前用 `meta.json` 中的参数做归一化和量化
