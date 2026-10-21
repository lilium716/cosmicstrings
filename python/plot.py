"""結果の可視化。

使い方: python python/plot.py <データのディレクトリ> <図の出力先> [スナップショット番号 ...]

- snapshots.png: いくつかの時刻の |φ| と arg φ（番号を省略すると 6 枚を等間隔に選ぶ）
- min_abs_phi.png: |φ| の最小値の時間変化（ひもがある間は 0 に近く、対消滅すると 1 に近づく）
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    data = Path(sys.argv[1])
    figures = Path(sys.argv[2])
    figures.mkdir(parents=True, exist_ok=True)
    params = json.loads((data / "params.json").read_text())
    snapshots = read_csv(data / "snapshots.csv")
    if len(sys.argv) > 3:
        indices = [int(s) for s in sys.argv[3:]]
    else:
        indices = np.linspace(0, len(snapshots) - 1, 6).round().astype(int).tolist()

    half = params["length"] / 2
    extent = [-half, half, -half, half]
    fig, axes = plt.subplots(2, len(indices), figsize=(3 * len(indices), 6), squeeze=False)
    for col, k in enumerate(indices):
        row = snapshots[k]
        phi = np.load(data / row["file"])  # phi[j, i] = φ(x_i, y_j)
        im0 = axes[0, col].imshow(np.abs(phi), origin="lower", extent=extent, vmin=0, vmax=1.1, cmap="viridis")
        im1 = axes[1, col].imshow(np.angle(phi), origin="lower", extent=extent, vmin=-np.pi, vmax=np.pi, cmap="twilight")
        axes[0, col].set_title(f"τ = {float(row['tau']):.1f}, a = {float(row['a']):.2f}", fontsize=9)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("|φ| / η")
    axes[1, 0].set_ylabel("arg φ")
    fig.colorbar(im0, ax=axes[0, :], shrink=0.8)
    fig.colorbar(im1, ax=axes[1, :], shrink=0.8)
    fig.savefig(figures / "snapshots.png", dpi=120)

    ts = read_csv(data / "timeseries.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([float(r["tau"]) for r in ts], [float(r["min_abs_phi"]) for r in ts], "o-")
    ax.set_xlabel("conformal time τ")
    ax.set_ylabel("min |φ| / η")
    fig.tight_layout()
    fig.savefig(figures / "min_abs_phi.png", dpi=120)
    print(f"saved {figures / 'snapshots.png'}, {figures / 'min_abs_phi.png'}")


if __name__ == "__main__":
    main()
