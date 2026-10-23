"""格子の細かさと QTT の圧縮率の関係を調べる（この実験専用）。

使い方: python analyze.py <data> <figures>

1. RK4 の収束: 格子 n と 2n の共通の格子点（細かい格子の偶数番目の点）での差の最大値と、
   最も細かい格子との差を、時刻ごとに表示する
2. QTT: 格子ごとの圧縮率・最大ボンド次元・パラメータ数・全格子での最大誤差を表と図にする
   - compression.png: 圧縮率と格子の 1 辺の点数の関係（時刻ごと、許容誤差ごと）
   - params.png: QTT のパラメータ数と格子の点数の関係
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
ns = sorted(int(p.name[1:]) for p in data.glob("n*") if p.is_dir())


def snapshots(n):
    return {int(r["index"]): (float(r["tau"]), r["file"]) for r in csv.DictReader(open(data / f"n{n}" / "snapshots.csv"))}


# ---------------------------------------------------------------- 1. RK4 の収束
print("RK4 の収束（共通の格子点での |φ_n − φ_2n| の最大値、最も細かい格子との差の最大値）")
finest = ns[-1]
for k in [4, 5, 10]:
    tau, _ = snapshots(ns[0])[k]
    row = [f"τ = {tau:4.0f}:"]
    fine_field = np.load(data / f"n{finest}" / snapshots(finest)[k][1])
    for n in ns[:-1]:
        coarse = np.load(data / f"n{n}" / snapshots(n)[k][1])
        finer = np.load(data / f"n{2 * n}" / snapshots(2 * n)[k][1])[::2, ::2]
        vs_finest = np.abs(coarse - fine_field[:: finest // n, :: finest // n]).max()
        row.append(f"n={n}: 隣と {np.abs(coarse - finer).max():.1e}, 最細と {vs_finest:.1e}")
    print("  " + " | ".join(row))

# ---------------------------------------------------------------- 2. QTT
rows = []
for n in ns:
    for r in csv.DictReader(open(data / f"qtt_n{n}" / "summary.csv")):
        rows.append({"n": n, **{k: float(v) for k, v in r.items()}})

taus = sorted({r["tau"] for r in rows})
tols = sorted({r["tolerance"] for r in rows}, reverse=True)
print("\nQTT の圧縮率（括弧内は最大ボンド次元）")
print("   τ      ε    " + "".join(f"{f'n = {n}':>18}" for n in ns))
for tau in taus:
    for tol in tols:
        cells = []
        for n in ns:
            r = next(r for r in rows if r["n"] == n and r["tau"] == tau and r["tolerance"] == tol)
            cells.append(f"{r['compression']:9.1f} ({int(r['max_rank']):3d})")
        print(f"{tau:5.0f}  {tol:.0e}  " + "".join(f"{c:>18}" for c in cells))
ratio = [r["max_abs_error"] / r["tolerance"] for r in rows]
print(f"\n全格子での最大誤差 / ε: {min(ratio):.2f} 〜 {max(ratio):.2f}")

fig, axes = plt.subplots(1, len(taus), figsize=(4 * len(taus), 3.8), sharey=True)
colors = plt.cm.plasma(np.linspace(0, 0.85, len(tols)))
for ax, tau in zip(axes, taus):
    for color, tol in zip(colors, tols):
        sel = sorted((r for r in rows if r["tau"] == tau and r["tolerance"] == tol), key=lambda r: r["n"])
        ax.loglog([r["n"] for r in sel], [r["compression"] for r in sel], "o-", color=color, label=f"ε = {tol:.0e}")
    ax.set_title(f"τ = {tau:.0f}")
    ax.set_xlabel("grid points per side n")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axhline(1, color="gray", ls="--", lw=0.8)
axes[0].set_ylabel("compression ratio (n² / QTT parameters)")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(figures / "compression.png", dpi=120)

fig, axes = plt.subplots(1, len(taus), figsize=(4 * len(taus), 3.8), sharey=True)
for ax, tau in zip(axes, taus):
    for color, tol in zip(colors, tols):
        sel = sorted((r for r in rows if r["tau"] == tau and r["tolerance"] == tol), key=lambda r: r["n"])
        ax.loglog([r["n"] for r in sel], [r["n_params"] for r in sel], "o-", color=color, label=f"ε = {tol:.0e}")
    ax.loglog(ns, [n * n for n in ns], color="gray", ls="--", lw=0.8, label="n² (full grid)")
    ax.set_title(f"τ = {tau:.0f}")
    ax.set_xlabel("grid points per side n")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[0].set_ylabel("number of QTT parameters")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(figures / "params.png", dpi=120)
print(f"saved {figures / 'compression.png'}, {figures / 'params.png'}")
