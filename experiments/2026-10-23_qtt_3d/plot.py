"""3 次元の QTT の結果を図にする（この実験専用）。

使い方: python plot.py <data> <figures>
- compression.png: 左 = 圧縮率、中 = 最大ボンド次元、右 = 全格子での最大誤差 / ε の時間変化（ε ごと）
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data, figures = Path(sys.argv[1]), Path(sys.argv[2])
figures.mkdir(parents=True, exist_ok=True)
rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(data / "qtt" / "summary.csv"))]
tols = sorted({r["tolerance"] for r in rows}, reverse=True)
colors = plt.cm.plasma(np.linspace(0, 0.85, len(tols)))
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for color, tol in zip(colors, tols):
    sel = sorted((r for r in rows if r["tolerance"] == tol), key=lambda r: r["tau"])
    tau = [r["tau"] for r in sel]
    axes[0].semilogy(tau, [r["compression"] for r in sel], "o-", color=color, label=f"ε = {tol:.0e}")
    axes[1].plot(tau, [r["max_rank"] for r in sel], "o-", color=color, label=f"ε = {tol:.0e}")
    axes[2].semilogy(tau, [r["max_abs_error"] / tol for r in sel], "o-", color=color, label=f"ε = {tol:.0e}")
for ax in axes:
    ax.axvspan(25.75, 26.0, color="gray", alpha=0.3)
    ax.set_xlabel("conformal time τ")
    ax.legend(fontsize=8)
axes[0].set_ylabel("compression ratio (256³ / QTT parameters)")
axes[1].set_ylabel("max bond dimension")
axes[2].set_ylabel("max |φ_QTT − φ| / ε")
axes[2].axhline(1, color="gray", ls="--", lw=0.8)
axes[0].set_title("gray band: reconnection", fontsize=9)
fig.tight_layout()
fig.savefig(figures / "compression.png", dpi=110)
print(f"saved {figures / 'compression.png'}")
