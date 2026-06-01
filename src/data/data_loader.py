"""
从 xlsx 文件加载 IMU 加速度数据，去除直流偏移
"""

import numpy as np
import openpyxl
from pathlib import Path


def find_xlsx(data_dir: str | Path) -> Path | None:
    """在目录中找到第一个 xlsx 文件"""
    p = Path(data_dir)
    files = sorted(p.glob("*.xlsx"))
    return files[0] if files else None


def load_xlsx(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 xlsx 中的 3 轴加速度数据。
    列布局: PC Date, PC Time, Line 1(X), Line 2(Y), Line 3(Z)
    返回 (ax, ay, az), dtype=float32
    """
    wb = openpyxl.load_workbook(str(path), read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    data = rows[1:]  # 跳过表头
    n = len(data)
    ax = np.empty(n, dtype=np.float32)
    ay = np.empty(n, dtype=np.float32)
    az = np.empty(n, dtype=np.float32)
    for i, row in enumerate(data):
        ax[i] = row[2]
        ay[i] = row[3]
        az[i] = row[4]
    wb.close()
    return ax, ay, az


def remove_dc_offset(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    static_n: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用前 static_n 行静态数据计算直流偏移，减去后返回交流分量。
    """
    dc_x = ax[:static_n].mean()
    dc_y = ay[:static_n].mean()
    dc_z = az[:static_n].mean()
    print(f"直流偏移 (前{static_n}行): X={dc_x:.6f}, Y={dc_y:.6f}, Z={dc_z:.6f}")
    return ax - dc_x, ay - dc_y, az - dc_z
