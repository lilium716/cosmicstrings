"""QTT の圧縮率と誤差の図を描く（この実験専用）。

使い方: python plot.py <data> <figures>
- compression.png: 左 = 許容誤差ごとの圧縮率の時間変化、右 = 全格子での最大誤差と許容誤差の関係
- bonddims.png: ボンドごとの次元（許容誤差 1e-4）
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data, figures = Path(sys.argv[1]), Path(sys.argv[2])
figures.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(data / "summary.csv")))
tols = sorted({float(r["tolerance"]) for r in rows}, reverse=True)
taus = sorted({float(r["tau"]) for r in rows})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
colors = plt.cm.plasma(np.linspace(0, 0.85, len(tols)))
for color, t in zip(colors, tols):
    sel = sorted((r for r in rows if float(r["tolerance"]) == t), key=lambda r: float(r["tau"]))
    axes[0].semilogy([float(r["tau"]) for r in sel], [float(r["compression"]) for r in sel], "o-", color=color, label=f"ε = {t:.0e}")
    axes[1].loglog([t] * len(sel), [float(r["max_abs_error"]) for r in sel], "o", color=color)
axes[0].axvspan(38, 40, color="gray", alpha=0.2, label="annihilation")
axes[0].axhline(1, color="gray", lw=0.8, ls="--")
axes[0].set_xlabel("conformal time τ")
axes[0].set_ylabel("compression ratio (grid points / QTT parameters)")
axes[0].legend(fontsize=8)
lim = [min(tols) / 3, max(tols) * 3]
axes[1].loglog(lim, lim, color="gray", ls="--", lw=0.8, label="error = ε")
axes[1].set_xlabel("tolerance ε")
axes[1].set_ylabel("max |φ_QTT − φ| over all grid points")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(figures / "compression.png", dpi=120)

bonds = defaultdict(list)
for r in csv.DictReader(open(data / "bonddims.csv")):
    if abs(float(r["tolerance"]) - 1e-4) < 1e-12:
        bonds[int(r["snapshot"])].append(int(r["dim"]))
tau_of = {int(r["snapshot"]): float(r["tau"]) for r in rows}
fig, ax = plt.subplots(figsize=(6.5, 4.2))
bc = plt.cm.viridis(np.linspace(0, 0.9, len(bonds)))
for color, (k, dims) in zip(bc, sorted(bonds.items())):
    ax.plot(dims, "o-", color=color, label=f"τ = {tau_of[k]:.0f}")
nb = len(next(iter(bonds.values())))
ax.plot([min(2 ** (b + 1), 2 ** (nb - b)) for b in range(nb)], color="gray", ls="--", lw=0.8, label="upper bound")
ax.set_ylim(0, None)
ax.set_xlabel("bond (left = coarse scale)")
ax.set_ylabel("bond dimension (ε = 1e-4)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(figures / "bonddims.png", dpi=120)
print(f"saved {figures / 'compression.png'}, {figures / 'bonddims.png'}")
