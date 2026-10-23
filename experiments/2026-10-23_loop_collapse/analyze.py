"""円形ループの収縮を解析する（この実験専用）。

使い方: python analyze.py <data> <figures>

- ループの半径: ひもの位置（巻き数をもつプラケットの中心）の、z 軸からの距離 ρ の平均。ばらつき（標準偏差）も出す
- 太さのない弦（南部・後藤の弦）の予言: 円形ループの半径 R(τ) は
      R″ + 2ℋ (1 − R′²) R′ = −(1 − R′²) / R
  に従う（膨張なしでは ℋ = 0 で、解は R = R₀ cos((τ − τ_i)/R₀)、τ − τ_i = πR₀/2 で潰れる）。
  静止した状態（R′ = 0）から 4 次のルンゲ・クッタ法で解く
- radius.png: 半径の時間変化（シミュレーションと南部・後藤の弦）
- loop.png: いくつかの時刻のひもの位置（3 次元）と y = 0 断面の |φ|
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
RUNS = [("minkowski", "no expansion", "C0"), ("radiation", "radiation era", "C3")]


def radius_history(run):
    out = []
    for r in csv.DictReader(open(run / "snapshots.csv")):
        rows = list(csv.DictReader(open(run / f"cores_{int(r['index']):04d}.csv")))
        if rows:
            rho = np.array([np.hypot(float(c["x"]), float(c["y"])) for c in rows])
            out.append((float(r["tau"]), rho.mean(), rho.std(), len(rows)))
        else:
            out.append((float(r["tau"]), np.nan, np.nan, 0))
    return np.array(out)


def nambu_goto(r0, tau0, tau_end, expanding, dt=1e-4):
    """R″ + 2ℋ(1 − R′²)R′ = −(1 − R′²)/R を、R(τ₀) = r0、R′(τ₀) = 0 から解く（R が 0 になったら止める）"""
    def f(tau, y):
        r, v = y
        h = 1.0 / tau if expanding else 0.0
        return np.array([v, -2 * h * (1 - v * v) * v - (1 - v * v) / r])

    tau, y = tau0, np.array([r0, 0.0])
    out = [(tau, r0)]
    while tau < tau_end and y[0] > 1e-3:
        k1 = f(tau, y)
        k2 = f(tau + dt / 2, y + dt / 2 * k1)
        k3 = f(tau + dt / 2, y + dt / 2 * k2)
        k4 = f(tau + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        tau += dt
        out.append((tau, y[0]))
    return np.array(out)


fig, ax = plt.subplots(figsize=(7, 4.5))
for name, label, color in RUNS:
    run = data / name
    if not run.exists():
        continue
    p = json.loads((run / "params.json").read_text())
    hist = radius_history(run)
    alive = hist[hist[:, 3] > 0]
    gone = hist[hist[:, 3] == 0]
    t_collapse_sim = gone[0, 0] if len(gone) else np.nan
    ng = nambu_goto(p["radius"], p["tau_initial"], p["tau_end"], name == "radiation")
    t_collapse_ng = ng[-1, 0] if ng[-1, 1] < 1e-2 else np.nan
    ax.errorbar(alive[:, 0], alive[:, 1], yerr=alive[:, 2], fmt="o", ms=3, color=color, label=f"simulation ({label})")
    ax.plot(ng[:, 0], ng[:, 1], color=color, ls="--", label=f"Nambu-Goto ({label})")
    print(
        f"{label}: ループが消えた時刻（ひもが検出されなくなった最初の出力）τ = {t_collapse_sim:.2f}"
        f"（τ_i からの経過 {t_collapse_sim - p['tau_initial']:.2f}）、"
        f"南部・後藤の弦で潰れる時刻 τ = {t_collapse_ng:.2f}（経過 {t_collapse_ng - p['tau_initial']:.2f}）"
    )
    for t, r, sd, n in hist[::4]:
        ng_r = np.interp(t, ng[:, 0], ng[:, 1], right=0.0)
        print(f"   τ = {t:5.1f}: 半径 {r:6.2f} ± {sd:4.2f}（点 {int(n):4d}）   南部・後藤 {ng_r:6.2f}")
ax.set_xlabel("conformal time τ")
ax.set_ylabel("loop radius (comoving)")
ax.set_ylim(0, None)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(figures / "radius.png", dpi=110)

# いくつかの時刻の様子
for name, label, _ in RUNS:
    run = data / name
    if not run.exists():
        continue
    p = json.loads((run / "params.json").read_text())
    half = p["length"] / 2
    snaps = list(csv.DictReader(open(run / "snapshots.csv")))
    show = np.linspace(0, len(snaps) - 1, 6).round().astype(int)
    fig = plt.figure(figsize=(3.4 * len(show), 6.4))
    for col, k in enumerate(show):
        ax3 = fig.add_subplot(2, len(show), col + 1, projection="3d")
        rows = list(csv.DictReader(open(run / f"cores_{k:04d}.csv")))
        if rows:
            xyz = np.array([[float(c["x"]), float(c["y"]), float(c["z"])] for c in rows])
            ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1)
        ax3.set_xlim(-half, half)
        ax3.set_ylim(-half, half)
        ax3.set_zlim(-half, half)
        ax3.set_title(f"τ = {float(snaps[k]['tau']):.1f}")
        ax2 = fig.add_subplot(2, len(show), len(show) + col + 1)
        ax2.imshow(np.abs(np.load(run / f"slice_y_{k:04d}.npy")), origin="lower", extent=[-half, half, -half, half], vmin=0, vmax=1.1)
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
    fig.suptitle(f"{label}: string cores (top) and |φ| at y = 0 (bottom)")
    fig.tight_layout()
    fig.savefig(figures / f"loop_{name}.png", dpi=100)
print(f"saved {figures}/radius.png, loop_*.png")
