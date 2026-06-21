"""验证 v1 / v2 模型结构和参数"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.cnn.model import build_model, build_model_v1

print("=" * 70)
print("IMU CNN v2 (Tier 1 + 2: 残差 + SE + 温和频率压缩)")
print("=" * 70)
m2 = build_model()
m2.summary()

print()
print("=" * 70)
print("IMU CNN v1 (旧版)")
print("=" * 70)
m1 = build_model_v1()
m1.summary()

print()
print(f"v1 参数: {m1.count_params():,}")
print(f"v2 参数: {m2.count_params():,}")
print(f"增长率: {(m2.count_params() / m1.count_params() - 1) * 100:.1f}%")
