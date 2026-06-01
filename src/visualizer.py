"""IMU FFT 结果可视化"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

from src.config import (
    SP_FFT_SIZE, SP_HOP_SIZE, SP_SAMPLE_RATE,
    SP_FREQ_RES, SP_FREQ_BINS, CNN_SAMPLE_FRAMES,
)

# 中文字体
_CN_FONT = None
for _name in ["Microsoft YaHei", "SimHei", "NSimSun"]:
    _matches = [f for f in fm.fontManager.ttflist if f.name == _name]
    if _matches:
        _CN_FONT = fm.FontProperties(fname=_matches[0].fname)
        break

if _CN_FONT is None:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    _CN_FONT = fm.FontProperties()


class CNNSampleViewer:
    """CNN 样本可视化器 — 滑动窗口，3 通道分开显示"""

    def __init__(self, sample_len: int = CNN_SAMPLE_FRAMES):
        self.sample_len = sample_len
        self.fig = None
        self.specs = {}
        self.n_frames = 0
        self.current_start = 0
        self._updating = False

    def load_data(self, result: dict):
        for key in ["x", "y", "z"]:
            _, spec = result[key]
            self.specs[key] = spec
        self.n_frames = self.specs["x"].shape[0]

    def setup(self):
        self.fig = plt.figure(figsize=(16, 12))
        gs = self.fig.add_gridspec(
            4, 4, height_ratios=[1, 2, 2, 2],
            hspace=0.45, left=0.06, right=0.93, top=0.93, wspace=0.4,
        )

        self.ax_overview = [self.fig.add_subplot(gs[0, i]) for i in range(3)]

        self.ax_ch = []
        self.ax_cb = []
        for row, (key, label) in enumerate(zip(["x", "y", "z"], ["X", "Y", "Z"])):
            ax = self.fig.add_subplot(gs[row + 1, 0:3])
            ax_cb = self.fig.add_subplot(gs[row + 1, 3])
            self.ax_ch.append(ax)
            self.ax_cb.append(ax_cb)
            ax.set_title(f"Ch{row}: {label} axis", fontproperties=_CN_FONT, fontsize=11)

        self.fig.suptitle(
            f"CNN Input Sample  |  shape=({3}, {self.sample_len}, {SP_FREQ_BINS})  "
            f"FFT={SP_FFT_SIZE}  HOP={SP_HOP_SIZE}  fs={SP_SAMPLE_RATE}Hz  "
            f"frames={self.n_frames}  dB",
            fontsize=11, fontproperties=_CN_FONT,
        )

        from matplotlib.widgets import Slider
        max_start = max(self.n_frames - self.sample_len, 0)
        ax_slider = self.fig.add_axes([0.12, 0.02, 0.76, 0.02])
        self.slider = Slider(
            ax_slider, "Start Frame", 0, max_start,
            valinit=0, valstep=1, valfmt="%d",
        )
        self.slider.on_changed(self._on_slider)
        self._draw_sample(0)

    def _draw_sample(self, start: int):
        if self._updating:
            return
        self._updating = True
        try:
            max_start = self.n_frames - self.sample_len
            start = max(0, min(start, max_start))
            self.current_start = start
            end = start + self.sample_len

            freqs = np.arange(SP_FREQ_BINS) * SP_FREQ_RES

            for ax, key in zip(self.ax_overview, ["x", "y", "z"]):
                ax.clear()
                spec = self.specs[key]
                times = np.arange(self.n_frames) * (SP_HOP_SIZE / SP_SAMPLE_RATE)
                ax.pcolormesh(times, freqs, spec.T, shading="auto", cmap="inferno")
                t0 = start * SP_HOP_SIZE / SP_SAMPLE_RATE
                t1 = end * SP_HOP_SIZE / SP_SAMPLE_RATE
                ax.axvspan(t0, t1, alpha=0.3, color="lime")
                ax.axvline(t0, color="lime", linewidth=0.8)
                ax.axvline(t1, color="lime", linewidth=0.8)
                ax.set_ylim(0, SP_SAMPLE_RATE / 2)
                ax.set_ylabel("Hz", fontproperties=_CN_FONT, fontsize=8)

            self.ax_overview[0].set_title(
                f"Overview  |  frames [{start}:{end}] / {self.n_frames}",
                fontproperties=_CN_FONT, fontsize=10,
            )

            for ax, cb_ax, key in zip(self.ax_ch, self.ax_cb, ["x", "y", "z"]):
                ax.clear()
                cb_ax.clear()
                ch_data = self.specs[key][start:end]
                im = ax.imshow(
                    ch_data.T, aspect="auto", cmap="inferno",
                    origin="lower", interpolation="nearest",
                    extent=[0, self.sample_len, 0, SP_SAMPLE_RATE / 2],
                )
                ax.set_ylabel("Hz", fontproperties=_CN_FONT, fontsize=9)
                ax.set_xlabel("Frame", fontproperties=_CN_FONT, fontsize=9)
                plt.colorbar(im, cax=cb_ax, label="dB")

            self.slider.set_val(start)
            self.fig.canvas.draw_idle()
        finally:
            self._updating = False

    def _on_slider(self, val):
        self._draw_sample(int(val))

    def get_current_sample(self) -> np.ndarray:
        """返回当前窗口的 CNN 输入: shape=(3, sample_len, FREQ_BINS), float32"""
        start = self.current_start
        end = start + self.sample_len
        return np.stack([
            self.specs["x"][start:end],
            self.specs["y"][start:end],
            self.specs["z"][start:end],
        ], axis=0).astype(np.float32)

    def show(self):
        plt.show()


def plot_fft_analysis(
    result: dict,
    raw_signals: dict | None = None,
    save_path: str | Path | None = None,
):
    """
    绘制完整的 FFT 分析图:
      行 0: 3 轴时域波形
      行 1: 3 轴频谱图 (时间-频率 热力图)
      行 2: 最后一帧频谱 (全频段)
      行 3: 最后一帧频谱 (0-500Hz 放大)
    """
    freqs = result["freqs"]
    sr = result["sample_rate"]
    n_samples = result["n_samples"]
    time_axis = np.arange(n_samples) / sr

    axis_labels = ["X", "Y", "Z"]
    keys = ["x", "y", "z"]

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle(
        f"IMU 加速度 FFT 分析  |  FFT={SP_FFT_SIZE}  HOP={SP_HOP_SIZE}  "
        f"fs={sr}Hz  df={SP_FREQ_RES:.2f}Hz  samples={n_samples}",
        fontsize=13, fontproperties=_CN_FONT,
    )

    for col, (key, label) in enumerate(zip(keys, axis_labels)):
        times, spec = result[key]

        if raw_signals and key in raw_signals:
            axes[0, col].plot(time_axis, raw_signals[key], linewidth=0.3, color="C0")
        axes[0, col].set_title(f"{label} 轴 时域波形", fontproperties=_CN_FONT)
        axes[0, col].set_xlabel("时间 (s)", fontproperties=_CN_FONT)
        axes[0, col].set_ylabel("加速度 (g)", fontproperties=_CN_FONT)
        axes[0, col].grid(True, alpha=0.3)

        im = axes[1, col].pcolormesh(
            times, freqs, spec.T, shading="auto", cmap="inferno",
        )
        axes[1, col].set_title(f"{label} 轴 频谱图", fontproperties=_CN_FONT)
        axes[1, col].set_xlabel("时间 (s)", fontproperties=_CN_FONT)
        axes[1, col].set_ylabel("频率 (Hz)", fontproperties=_CN_FONT)
        axes[1, col].set_ylim(0, sr / 2)
        plt.colorbar(im, ax=axes[1, col], label="dB")

        last_spec = spec[-1]
        axes[2, col].plot(freqs, last_spec, linewidth=0.8, color="C1")
        axes[2, col].set_title(f"{label} 轴 最后一帧频谱 (全频段)", fontproperties=_CN_FONT)
        axes[2, col].set_xlabel("频率 (Hz)", fontproperties=_CN_FONT)
        axes[2, col].set_ylabel("幅度", fontproperties=_CN_FONT)
        axes[2, col].grid(True, alpha=0.3)

        mask = freqs <= 500
        axes[3, col].plot(freqs[mask], last_spec[mask], linewidth=1.0, color="C2")
        axes[3, col].set_title(f"{label} 轴 最后一帧频谱 (0-500Hz)", fontproperties=_CN_FONT)
        axes[3, col].set_xlabel("频率 (Hz)", fontproperties=_CN_FONT)
        axes[3, col].set_ylabel("幅度", fontproperties=_CN_FONT)
        axes[3, col].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
        print(f"图表已保存: {save_path}")
    plt.show()
