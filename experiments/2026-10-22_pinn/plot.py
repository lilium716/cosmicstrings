"""PINN の学習結果の図を描く（この実験専用）。

使い方: python plot.py <data> <figures>
- training.png: 3 通りの学習での、RK4 との相対 L2 誤差と方程式の残差の推移（データを外した反復に縦線）
- error_vs_time.png: 学習後の誤差の時刻ごとの値（PINN の PyTorch 版・tenferro-rs 版、RBF を重ねる）。
  あわせて data/summary.csv に平均の誤差を書き出す
- fields.png: いくつかの時刻での |φ|（RK4、データを外した学習の PINN、その差）
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
RUNS = [
    ("curriculum", "data → physics only", "C0"),
    ("physics_only", "physics only", "C3"),
    ("data_physics", "data + physics", "C2"),
]


def load(name):
    rows = list(csv.DictReader(open(data / name / "metrics.csv")))
    col = lambda k: np.array([float(r[k]) for r in rows])
    config = json.loads((data / name / "config.json").read_text())
    return col, config


# ---------------------------------------------------------------- 学習の推移
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for name, label, color in RUNS:
    col, config = load(name)
    it = col("iteration")
    axes[0].semilogy(it, col("error_data_times"), color=color, label=f"{label} (data times)")
    axes[0].semilogy(it, col("error_heldout_times"), color=color, ls="--", label=f"{label} (held-out times)")
    axes[1].semilogy(it, col("loss_physics"), color=color, label=label)
    if config["switch_iteration"] is not None:
        for ax in axes:
            ax.axvline(config["switch_iteration"], color=color, ls=":", lw=1)
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("relative L2 error vs RK4")
axes[0].legend(fontsize=7)
axes[1].set_xlabel("iteration")
axes[1].set_ylabel("mean squared PDE residual")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(figures / "training.png", dpi=120)

# ---------------------------------------------------------------- 学習後の時刻ごとの誤差
rk4 = json.loads((data / "rk4" / "params.json").read_text())
grid = np.arange(rk4["nx"]) * rk4["dx"] - rk4["length"] / 2 - rk4["center_offset"]
taus = np.load(data / "curriculum" / "taus.npy")
ix = np.where((grid >= -8 - 1e-9) & (grid <= 8 + 1e-9))[0]
iy = np.where((grid >= -6 - 1e-9) & (grid <= 6 + 1e-9))[0]
snapshots = {round(float(r["tau"]), 6): r["file"] for r in csv.DictReader(open(data / "rk4" / "snapshots.csv"))}
exact = np.stack([np.load(data / "rk4" / snapshots[round(float(t), 6)])[np.ix_(iy, ix)] for t in taus])
is_data = np.isclose(taus, np.round(taus)) & (taus > taus[0] + 1e-9)

# 方法（線の種類）と学習のしかた（色）の組み合わせ
METHODS = [("", "PINN (PyTorch)", "-"), ("tenferro_", "PINN (tenferro-rs)", "-."), ("rbf_", "RBF", ":")]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
summary = []
for prefix, method, ls in METHODS:
    for name, label, color in RUNS:
        run = data / f"{prefix}{name}"
        if not (run / "prediction.npy").exists():
            continue
        pred = np.load(run / "prediction.npy")
        err = np.linalg.norm((pred - exact).reshape(len(taus), -1), axis=1) / np.linalg.norm(exact.reshape(len(taus), -1), axis=1)
        ax.semilogy(taus, err, color=color, ls=ls, label=f"{method}: {label}")
        summary.append((method, label, err[is_data].mean(), err[~is_data].mean(), err.max()))
ax.axvspan(38, 40, color="gray", alpha=0.15)
ax.set_xlabel("conformal time τ")
ax.set_ylabel("relative L2 error vs RK4 (after training)")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(figures / "error_vs_time.png", dpi=120)
with open(figures.parent / "data" / "summary.csv", "w") as f:
    f.write("method,training,error_data_times,error_heldout_times,error_max\n")
    for m, l, ed, eh, em in summary:
        f.write(f"{m},{l},{ed:.4e},{eh:.4e},{em:.4e}\n")
        print(f"{m:20s} {l:22s} データ時刻 {ed:.4f} 検証時刻 {eh:.4f} 最大 {em:.4f}")

# ---------------------------------------------------------------- 場の比較
pred = np.load(data / "curriculum" / "prediction.npy")
show = [np.argmin(np.abs(taus - t)) for t in (34.0, 37.0, 39.0, 42.0)]
extent = [grid[ix[0]], grid[ix[-1]], grid[iy[0]], grid[iy[-1]]]
fig, axes = plt.subplots(3, len(show), figsize=(3.4 * len(show), 7.2), squeeze=False)
for col_, k in enumerate(show):
    for row, (img, vmin, vmax, cmap) in enumerate(
        [
            (np.abs(exact[k]), 0, 1.1, "viridis"),
            (np.abs(pred[k]), 0, 1.1, "viridis"),
            (np.abs(pred[k] - exact[k]), 0, 0.3, "magma"),
        ]
    ):
        im = axes[row, col_].imshow(img, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
        axes[row, col_].set_xticks([])
        axes[row, col_].set_yticks([])
        if col_ == len(show) - 1:
            fig.colorbar(im, ax=axes[row, col_], shrink=0.8)
    axes[0, col_].set_title(f"τ = {taus[k]:.1f}")
axes[0, 0].set_ylabel("RK4 |φ|")
axes[1, 0].set_ylabel("PINN |φ|\n(data → physics only)")
axes[2, 0].set_ylabel("|φ_PINN − φ_RK4|")
fig.tight_layout()
fig.savefig(figures / "fields.png", dpi=120)
print(f"saved {figures}/training.png, error_vs_time.png, fields.png")
