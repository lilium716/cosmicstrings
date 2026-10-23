"""時間刻みを変えた実行のエネルギーの相対変化を比べる（この実験専用）。

使い方: python analyze.py RUN1 RUN2 ... --figure 図のパス
  RUN は時間刻みを半分ずつにした順に並べる。

各実行で (E(τ) − E(τ_i)) / E(τ_i) を求め、その最大の大きさと、
隣り合う実行の比（時間刻みを半分にしたときの減り方）と、その log₂（収束次数）を表示する。
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

args = sys.argv[1:]
figure = Path(args[args.index("--figure") + 1])
runs = [Path(a) for a in args[: args.index("--figure")]]

drifts = []
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for run in runs:
    params = json.loads((run / "params.json").read_text())
    rows = list(csv.DictReader(open(run / "timeseries.csv")))
    tau = np.array([float(r["tau"]) for r in rows])
    e = np.array([float(r["energy"]) for r in rows])
    rel = (e - e[0]) / e[0]
    drift = np.abs(rel).max()
    drifts.append(drift)
    print(f"Δτ/Δx = {params['courant']:5}: E(τ_i) = {e[0]:.10f}, 最大の相対変化 {drift:.2e}, 終了時 {rel[-1]:+.2e}")
    ax.semilogy(tau, np.abs(rel) + 1e-17, "o-", ms=3, label=f"Δτ/Δx = {params['courant']}")
for coarse, fine in zip(drifts, drifts[1:]):
    print(f"比 {coarse / fine:6.1f}（収束次数 {np.log2(coarse / fine):.2f}）")

ax.set_xlabel("conformal time τ")
ax.set_ylabel("|E − E_i| / E_i")
ax.set_title("energy without expansion (RK4)")
ax.legend(fontsize=8)
fig.tight_layout()
figure.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure, dpi=120)
print(f"saved {figure}")
