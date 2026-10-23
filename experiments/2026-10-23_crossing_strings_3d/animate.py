"""交差するひもの衝突を、組み換えが分かるように図とアニメーションにする（この実験専用）。

使い方: python animate.py <データ（crossing_fine）> <figures>

ひもの位置（巻き数をもつプラケットの中心）を、つながっている塊に分ける。
各塊が箱のどの壁で終わっているかで色を決める:
    ひも A（z = ±16 の両方で終わる）                 : 青
    ひも B（y = ±16 の両方で終わる）                 : 橙
    組み換え後「z = +16 と y = +16 をつなぐ」ひも   : 緑
    組み換え後「z = −16 と y = −16 をつなぐ」ひも   : 赤
組み換えが起きると、青と橙の 2 本が、緑と赤の 2 本に変わる。

出力:
- reconnection.gif: 3 次元の図と 3 方向からの投影図のアニメーション
- reconnection_summary.png: いくつかの時刻の図と、組み換えの模式図
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

run, figures = Path(sys.argv[1]), Path(sys.argv[2])
figures.mkdir(parents=True, exist_ok=True)
params = json.loads((run / "params.json").read_text())
half = params["length"] / 2
snaps = list(csv.DictReader(open(run / "snapshots.csv")))

COLORS = {"A": "tab:blue", "B": "tab:orange", "up": "tab:green", "down": "tab:red", "?": "gray"}
LABELS = {
    "A": "string A (z = -16 to +16)",
    "B": "string B (y = -16 to +16)",
    "up": "after reconnection: z = +16 to y = +16",
    "down": "after reconnection: z = -16 to y = -16",
}


def load(k):
    rows = list(csv.DictReader(open(run / f"cores_{k:04d}.csv")))
    return np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows]).reshape(-1, 3)


def pieces(points, threshold=0.2):
    """距離 threshold 以内の点をつないで塊に分ける"""
    label = -np.ones(len(points), dtype=int)
    current = 0
    for i in range(len(points)):
        if label[i] >= 0:
            continue
        stack, label[i] = [i], current
        while stack:
            j = stack.pop()
            near = np.where((np.linalg.norm(points - points[j], axis=1) < threshold) & (label < 0))[0]
            label[near] = current
            stack.extend(near)
        current += 1
    return [points[label == c] for c in range(current)]


def classify(piece):
    """塊が終わっている壁から種類を決める"""
    edge = half - 0.5
    ends = {
        "z+": piece[:, 2].max() > edge,
        "z-": piece[:, 2].min() < -edge,
        "y+": piece[:, 1].max() > edge,
        "y-": piece[:, 1].min() < -edge,
    }
    if ends["z+"] and ends["z-"]:
        return "A"
    if ends["y+"] and ends["y-"]:
        return "B"
    if ends["z+"] and ends["y+"]:
        return "up"
    if ends["z-"] and ends["y-"]:
        return "down"
    return "?"


frames = []
for k in range(len(snaps)):
    pts = load(k)
    frames.append((float(snaps[k]["tau"]), [(classify(p), p) for p in pieces(pts)] if len(pts) else []))


VIEWS = [  # (タイトル, 横軸, 縦軸)
    ("front (x-z, seen along y)", 0, 2),
    ("top (x-y, seen along z)", 0, 1),
    ("side (y-z, seen along x)", 1, 2),
]
AXIS = "xyz"


def draw(axes3d, axes2d, tau, items):
    axes3d.cla()
    for ax in axes2d:
        ax.cla()
    for kind, p in items:
        axes3d.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c=COLORS[kind])
        for ax, (_, h, v) in zip(axes2d, VIEWS):
            ax.scatter(p[:, h], p[:, v], s=1, c=COLORS[kind])
    axes3d.set_xlim(-half, half)
    axes3d.set_ylim(-half, half)
    axes3d.set_zlim(-half, half)
    axes3d.set_xlabel("x")
    axes3d.set_ylabel("y")
    axes3d.set_zlabel("z")
    axes3d.view_init(elev=20, azim=-60)
    kinds = {k for k, _ in items}
    state = "before reconnection" if "A" in kinds else "after reconnection" if kinds & {"up", "down"} else ""
    axes3d.set_title(f"τ = {tau:.2f}  {state}")
    for ax, (title, h, v) in zip(axes2d, VIEWS):
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_xlabel(AXIS[h])
        ax.set_ylabel(AXIS[v])
        ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------- アニメーション
fig = plt.figure(figsize=(14, 4))
ax3 = fig.add_subplot(1, 4, 1, projection="3d")
ax2 = [fig.add_subplot(1, 4, c) for c in (2, 3, 4)]
handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[k], label=LABELS[k]) for k in LABELS]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8)
fig.subplots_adjust(bottom=0.2, wspace=0.35)
anim = FuncAnimation(fig, lambda k: draw(ax3, ax2, *frames[k]), frames=len(frames))
anim.save(figures / "reconnection.gif", writer=PillowWriter(fps=8), dpi=80)

# ---------------------------------------------------------------- まとめの図
show_taus = [20.0, 23.0, 25.0, 26.0, 28.0, 34.0]
show = [min(range(len(frames)), key=lambda k: abs(frames[k][0] - t)) for t in show_taus]
fig = plt.figure(figsize=(15, 3.1 * (len(show) + 1)))
for row, k in enumerate(show):
    a3 = fig.add_subplot(len(show) + 1, 4, 4 * row + 1, projection="3d")
    a2 = [fig.add_subplot(len(show) + 1, 4, 4 * row + c) for c in (2, 3, 4)]
    draw(a3, a2, *frames[k])

# 模式図: 組み換えの前と後
for col, (title, lines) in enumerate(
    [
        (
            "schematic: before",
            [("A", [(-4, 0, -16), (-4, 0, 16)]), ("B", [(4, -16, 0), (4, 16, 0)])],
        ),
        (
            "schematic: after",
            [("up", [(-4, 0, 16), (0, 0.5, 0.5), (4, 16, 0)]), ("down", [(-4, 0, -16), (0, -0.5, -0.5), (4, -16, 0)])],
        ),
    ]
):
    ax = fig.add_subplot(len(show) + 1, 4, 4 * len(show) + 1 + col, projection="3d")
    for kind, pts in lines:
        p = np.array(pts, dtype=float)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=COLORS[kind], lw=3)
    if col == 0:
        ax.quiver(-4, 0, 0, 3, 0, 0, color="k")
        ax.quiver(4, 0, 0, -3, 0, 0, color="k")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=-60)
    ax.set_title(title)
fig.legend(handles=handles, loc="lower right", ncol=1, fontsize=9, bbox_to_anchor=(0.98, 0.03))
fig.tight_layout()
fig.savefig(figures / "reconnection_summary.png", dpi=90)

for tau, items in frames[:: max(1, len(frames) // 16)]:
    print(f"τ = {tau:5.2f}: " + ", ".join(f"{k}({len(p)})" for k, p in items))
print(f"saved {figures / 'reconnection.gif'}, {figures / 'reconnection_summary.png'}")
