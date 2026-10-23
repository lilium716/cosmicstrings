"""512³（箱 64）と 256³（箱 32）の交差するひもの衝突を比べる（この実験専用）。

使い方: python compare.py <512³ の data> <256³ の data> <figures>

Δx と座標の取り方が同じなので、箱 32 の領域では両者の格子点がぴったり重なる。
- 組み換えの時刻（ひもの位置の塊の種類が A, B から up, down に変わる時刻）
- z = 0 の断面での |φ_512 − φ_256| の最大値（中心付近 |x|, |y| < 8 と、箱 32 全体）
を表示し、差の時間変化を figures/box_size.png に描く。
箱 32 の壁の影響は光速で伝わるので、中心（壁から 16）には τ = 36 まで届かないはずである。
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

big, small, figures = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
figures.mkdir(parents=True, exist_ok=True)


def reconnection_tau(run):
    half = json.loads((run / "params.json").read_text())["length"] / 2
    edge = half - 0.5
    for r in csv.DictReader(open(run / "snapshots.csv")):
        rows = list(csv.DictReader(open(run / f"cores_{int(r['index']):04d}.csv")))
        pts = np.array([[float(c["y"]), float(c["z"])] for c in rows]).reshape(-1, 2)
        # 組み換え前は、z = ±half の両方に届く塊（ひも A）がある。組み換え後は上下で分かれる
        # 簡単のため: y > 0 かつ z > 0 の点と、y < 0 かつ z < 0 の点だけからなる（中心を通らない）なら組み換え後
        near_center = pts[np.linalg.norm(pts, axis=1) < 1.0]
        if len(near_center) == 0 and len(pts) > 0:
            return float(r["tau"])
    return float("nan")


pb = json.loads((big / "params.json").read_text())
ps = json.loads((small / "params.json").read_text())
assert abs(pb["dx"] - ps["dx"]) < 1e-12
shift = int(round((pb["length"] - ps["length"]) / 2 / pb["dx"]))  # 箱 32 の原点が箱 64 のどこに来るか
n_small = ps["nx"]
grid = np.arange(n_small) * ps["dx"] - ps["length"] / 2 - ps["center_offset"]
center = (np.abs(grid)[None, :] < 8) & (np.abs(grid)[:, None] < 8)

print(f"組み換えの時刻（中心付近 r < 1 からひもが消えた最初の出力時刻）: 512³ τ = {reconnection_tau(big)}, 256³ τ = {reconnection_tau(small)}")
taus, d_center, d_all = [], [], []
snaps_small = {round(float(r["tau"]), 6): int(r["index"]) for r in csv.DictReader(open(small / "snapshots.csv"))}
for r in csv.DictReader(open(big / "snapshots.csv")):
    tau = round(float(r["tau"]), 6)
    if tau not in snaps_small:
        continue
    a = np.load(big / f"slice_z_{int(r['index']):04d}.npy")[shift : shift + n_small, shift : shift + n_small]
    b = np.load(small / f"slice_z_{snaps_small[tau]:04d}.npy")
    diff = np.abs(a - b)
    taus.append(tau)
    d_center.append(diff[center].max())
    d_all.append(diff.max())
    print(f"τ = {tau:5.1f}: z = 0 断面の差の最大値  中心 |x|,|y| < 8: {d_center[-1]:.1e}   箱 32 全体: {d_all[-1]:.1e}")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogy(taus, d_center, "o-", label="center |x|, |y| < 8")
ax.semilogy(taus, d_all, "s-", label="whole box-32 region")
ax.set_xlabel("conformal time τ")
ax.set_ylabel("max |φ(512³, box 64) − φ(256³, box 32)| at z = 0")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(figures / "box_size.png", dpi=110)
print(f"saved {figures / 'box_size.png'}")
