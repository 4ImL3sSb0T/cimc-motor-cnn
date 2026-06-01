"""
全局配置 — FFT 参数与 CNN 参数
与固件 src/service/signal_process/imu_data_process.h 对齐
"""

from pathlib import Path

# ── FFT 参数 ─────────────────────────────────────────────────────────────────
SP_FFT_SIZE = 1024
SP_HOP_SIZE = 256
SP_FREQ_BINS = SP_FFT_SIZE // 2          # 512
SP_SAMPLE_RATE = 6667.0                   # Hz
SP_FREQ_RES = SP_SAMPLE_RATE / SP_FFT_SIZE  # 6.51 Hz

# ── CNN 输入 ─────────────────────────────────────────────────────────────────
CNN_SAMPLE_FRAMES = 16                    # 每个样本的时间帧数
CNN_INPUT_SHAPE = (3, CNN_SAMPLE_FRAMES, SP_FREQ_BINS)  # (channels, frames, freq_bins)

# ── 数据处理 ─────────────────────────────────────────────────────────────────
DC_OFFSET_SAMPLES = 10000                 # 用于计算直流偏移的前 N 行静态数据

# ── CNN 训练 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

# 分类类别 — 根据实际数据修改
CLASS_NAMES = ["idle", "vibration", "impact", "other"]
NUM_CLASSES = len(CLASS_NAMES)

# 训练超参数
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2
PATIENCE = 15                          # 早停耐心值
