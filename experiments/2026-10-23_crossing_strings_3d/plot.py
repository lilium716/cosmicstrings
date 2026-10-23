"""3 次元の交差するひもの衝突を図にする（この実験専用）。

使い方: python plot.py <data> <figures>
- cores.png: いくつかの時刻のひもの位置（巻き数をもつプラケットの中心）の 3 次元散布図
- slices.png: z = 0 と y = 0 の断面での |φ|
- timeseries.png: ひもの長さと |φ| の最小値の時間変化
あわせて、(1) 平行なひもでの 2 次元との比較の結果を表示する
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data, figures = Path(sys.argv[1]), Path(sys.argv[2])
figures.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- (1) 2 次元との比較
check, ref = data / "parallel_check", data.parent.parent / "2026-10-21_global_string_flrw" / "data"
if check.exists() and ref.exists():
    diffs = [
        np.abs(np.load(check / f"slice_z_{k:04d}.npy") - np.load(ref / f"snapshot_{k:04d}.npy")).max()
        for k in range(21)
    ]
    print(f"平行なひも（z 周期、512×512×4）と 2 次元の差の最大値: {max(diffs):.2e}")

# ---------------------------------------------------------------- (2) 交差するひも
run = data / "crossing"
params = json.loads((run / "params.json").read_text())
snaps = list(csv.DictReader(open(run / "snapshots.csv")))
ts = list(csv.DictReader(open(run / "timeseries.csv")))
half = params["length"] / 2
show = [k for k in (0, 6, 8, 10, 12, 16) if k < len(snaps)]

fig = plt.figure(figsize=(4.2 * len(show), 4.4))
for col, k in enumerate(show):
    ax = fig.add_subplot(1, len(show), col + 1, projection="3d")
    cores = list(csv.DictReader(open(run / f"cores_{k:04d}.csv")))
    if cores:
        xyz = np.array([[float(c["x"]), float(c["y"]), float(c["z"])] for c in cores])
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.5, c="C0")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"τ = {float(snaps[k]['tau']):.0f}")
    ax.view_init(elev=20, azim=-60)
fig.tight_layout()
fig.savefig(figures / "cores.png", dpi=110)

nx, nz = params["nx"], params["nz"]
fig, axes = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6.4), squeeze=False)
for col, k in enumerate(show):
    for row, (name, ylabel) in enumerate([("slice_z", "y"), ("slice_y", "z")]):
        phi = np.load(run / f"{name}_{k:04d}.npy")
        axes[row, col].imshow(np.abs(phi), origin="lower", extent=[-half, half, -half, half], vmin=0, vmax=1.1)
        axes[row, col].set_xlabel("x")
        axes[row, col].set_ylabel(ylabel)
    axes[0, col].set_title(f"τ = {float(snaps[k]['tau']):.0f}")
axes[0, 0].text(-0.45, 0.5, "|φ| at z = 0", transform=axes[0, 0].transAxes, rotation=90, va="center")
axes[1, 0].text(-0.45, 0.5, "|φ| at y = 0", transform=axes[1, 0].transAxes, rotation=90, va="center")
fig.tight_layout()
fig.savefig(figures / "slices.png", dpi=110)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
tau = [float(r["tau"]) for r in ts]
axes[0].plot(tau, [float(r["string_length"]) for r in ts], "o-")
axes[0].set_xlabel("conformal time τ")
axes[0].set_ylabel("string length (plaquettes × Δx)")
axes[1].plot(tau, [float(r["min_abs_phi"]) for r in ts], "o-")
axes[1].set_xlabel("conformal time τ")
axes[1].set_ylabel("min |φ| / η")
fig.tight_layout()
fig.savefig(figures / "timeseries.png", dpi=110)
print(f"saved {figures}/cores.png, slices.png, timeseries.png")
