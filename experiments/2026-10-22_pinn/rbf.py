"""PINN と同じ問題を、時空のガウス型動径基底関数（RBF）とガウス・ニュートン法で解く（この実験専用）。

近似の形（中心は時空の規則格子、幅は中心の間隔の SIGMA 倍）:
    φ(x, y, τ) = Σ_{a,b,c} W[a, b, c] g_a(x) g_b(y) g_c(τ),   g_a(x) = exp(−((x − x_a) / (SIGMA·h))²)
基底が x・y・τ の積なので、格子点での値や微分は 1 次元の行列を順にかけるだけで求まる。
W（複素数）について φ は線形なので、方程式の非線形項（ポテンシャル）以外は線形最小二乗になる。
これをガウス・ニュートン法で解く。各ステップの線形最小二乗は、正規方程式を共役勾配法（CG）で解く。

損失（PINN と同じ重み）:
    物理       方程式の残差。窓の中の規則格子（Δx = 0.125、Δτ = 0.125）で評価する
    初期条件   τ = 32 の φ と φ′
    境界条件   窓の縁の φ（保存した全時刻）
    データ     窓の内部の φ（整数の時刻 τ = 33, …, 42）
各損失は「点の数で割った 2 乗和 × 重み」で、全体の和を最小にする。

--mode curriculum   : データありで解いたあと、データを外して解き直す
（データありの場合の出発点は、方程式を使わずにデータ・初期条件・境界条件を最小二乗で当てはめたもの）
--mode physics_only : データを使わない。初期値は τ = 32 の φ を時間方向にそのまま伸ばしたもの
--mode data_physics : 最後までデータを使う

出力（--out）: metrics.csv、prediction.npy（[時刻, ny, nx]）、taus.npy、config.json
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp, vjp

parser = argparse.ArgumentParser()
parser.add_argument("--rk4", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--mode", choices=["curriculum", "physics_only", "data_physics"], required=True)
parser.add_argument("--spacing", type=float, default=0.25, help="中心の間隔（空間）")
parser.add_argument("--dt-spacing", type=float, default=0.25, help="中心の間隔（時間）")
parser.add_argument("--sigma", type=float, default=1.0, help="ガウス関数の幅 / 中心の間隔")
parser.add_argument("--gn-steps", type=int, default=10, help="ガウス・ニュートン法の反復回数（1 段階あたり）")
parser.add_argument("--cg-iters", type=int, default=300, help="各ステップの CG の反復回数")
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(args.threads)
torch.set_default_dtype(torch.float64)

X_RANGE, Y_RANGE, T_RANGE = (-8.0, 8.0), (-6.0, 6.0), (32.0, 42.0)
WEIGHTS = {"physics": 1.0, "ic_phi": 10.0, "ic_pi": 10.0, "boundary": 10.0, "data": 10.0}

# ---------------------------------------------------------------- RK4 のデータ（PINN と同じ窓）
params = json.loads((args.rk4 / "params.json").read_text())
eta, lam, tau_i = params["eta"], params["lambda"], params["tau_initial"]
grid = np.arange(params["nx"]) * params["dx"] - params["length"] / 2 - params["center_offset"]
ix = np.where((grid >= X_RANGE[0] - 1e-9) & (grid <= X_RANGE[1] + 1e-9))[0]
iy = np.where((grid >= Y_RANGE[0] - 1e-9) & (grid <= Y_RANGE[1] + 1e-9))[0]
xs, ys = grid[ix], grid[iy]
frames = []
with open(args.rk4 / "snapshots.csv") as f:
    for row in csv.DictReader(f):
        tau = float(row["tau"])
        if T_RANGE[0] - 1e-9 <= tau <= T_RANGE[1] + 1e-9:
            phi = np.load(args.rk4 / row["file"])[np.ix_(iy, ix)]
            pi = np.load(args.rk4 / row["file"].replace("snapshot", "momentum"))[np.ix_(iy, ix)]
            frames.append((tau, phi, pi))
taus = np.array([f[0] for f in frames])
is_data_time = np.isclose(taus, np.round(taus)) & (taus > taus[0] + 1e-9)
exact = np.stack([f[1] for f in frames])  # [T, ny, nx]
phi0, pi0 = frames[0][1], frames[0][2]
t_col = np.arange(T_RANGE[0], T_RANGE[1] + 1e-9, 0.125)  # 物理の評価時刻

# ---------------------------------------------------------------- 基底
def centers(lo, hi, h):
    """[lo, hi] を覆い、両側に 1 つずつはみ出す等間隔の中心"""
    n = int(np.ceil((hi - lo) / h)) + 1
    return lo + h * (np.arange(n + 2) - 1)


cx = centers(xs[0], xs[-1], args.spacing)
cy = centers(ys[0], ys[-1], args.spacing)
ct = centers(T_RANGE[0], T_RANGE[1], args.dt_spacing)


def basis(points, c, h):
    """g(p) = exp(−((p − c)/(σh))²) とその 1 階・2 階微分の行列 [len(points), len(c)]"""
    s = args.sigma * h
    d = (np.asarray(points)[:, None] - c[None, :]) / s
    g = np.exp(-(d**2))
    return [torch.tensor(m) for m in (g, -2 * d / s * g, (4 * d**2 - 2) / s**2 * g)]


Gx, Gy, Gt_col = basis(xs, cx, args.spacing), basis(ys, cy, args.spacing), basis(t_col, ct, args.dt_spacing)
Gt_frames = basis(taus, ct, args.dt_spacing)
n_unknowns = 2 * len(cx) * len(cy) * len(ct)
print(f"中心 {len(cx)}×{len(cy)}×{len(ct)}（未知数 {n_unknowns}）、物理の評価点 {len(xs)}×{len(ys)}×{len(t_col)}")


def field(W, gx, gy, gt):
    """W[2, a, b, c] から格子上の値 [2, nt, ny, nx] を求める（gx: [nx, a] など）"""
    # 1 次元ずつ順にかけて、途中結果を小さく保つ
    w = torch.einsum("kabc,tc->kabt", W, gt)
    w = torch.einsum("kabt,yb->katy", w, gy)
    return torch.einsum("katy,xa->ktyx", w, gx)


# 方程式の係数（物理の評価時刻ごと）
a2 = torch.tensor((t_col / tau_i) ** 2)[None, :, None, None]
hubble = torch.tensor(1.0 / t_col)[None, :, None, None]
edge = np.zeros((len(ys), len(xs)), dtype=bool)
edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
edge_t = torch.tensor(edge)
interior_t = ~edge_t
phi0_t = torch.tensor(np.stack([phi0.real, phi0.imag]))  # [2, ny, nx]
pi0_t = torch.tensor(np.stack([pi0.real, pi0.imag]))
exact_t = torch.tensor(np.stack([exact.real, exact.imag]))  # [2, T, ny, nx]
data_idx = torch.tensor(np.where(is_data_time)[0])
n_phys = 2 * len(t_col) * len(ys) * len(xs)
n_ic = 2 * len(ys) * len(xs)
n_bc = 2 * len(taus) * int(edge.sum())
n_data = 2 * len(data_idx) * int((~edge).sum())


def residuals(W, use_data):
    """重み付きの残差ベクトル（2 乗和が損失になるように √(重み / 点の数) をかけてある）"""
    gx, gxp, gxpp = Gx
    gy, gyp, gypp = Gy
    gt, gtp, gtpp = Gt_col
    phi = field(W, gx, gy, gt)
    phi_t = field(W, gx, gy, gtp)
    phi_tt = field(W, gx, gy, gtpp)
    lap = field(W, gxpp, gy, gt) + field(W, gx, gypp, gt)
    abs2 = (phi**2).sum(dim=0, keepdim=True)
    physics = phi_tt + 2 * hubble * phi_t - lap + a2 * lam * (abs2 - eta**2) * phi
    gtf, gtfp, _ = Gt_frames
    frames_phi = field(W, gx, gy, gtf)  # [2, T, ny, nx]
    ic_phi = frames_phi[:, 0] - phi0_t
    ic_pi = field(W, gx, gy, gtfp)[:, 0] - pi0_t
    boundary = (frames_phi - exact_t)[:, :, edge_t]
    parts = [
        physics.flatten() * (WEIGHTS["physics"] / n_phys) ** 0.5,
        ic_phi.flatten() * (WEIGHTS["ic_phi"] / n_ic) ** 0.5,
        ic_pi.flatten() * (WEIGHTS["ic_pi"] / n_ic) ** 0.5,
        boundary.flatten() * (WEIGHTS["boundary"] / n_bc) ** 0.5,
    ]
    if use_data:
        data = (frames_phi - exact_t)[:, data_idx][:, :, interior_t]
        parts.append(data.flatten() * (WEIGHTS["data"] / n_data) ** 0.5)
    return torch.cat(parts)


def losses(W):
    """損失の内訳（PINN の metrics.csv と同じ定義：重みをかける前の平均 2 乗）"""
    with torch.no_grad():
        r = residuals(W, True)
        sizes = [n_phys, n_ic, n_ic, n_bc, n_data]
        names = ["physics", "ic_phi", "ic_pi", "boundary", "data"]
        out, k = {}, 0
        for name, n in zip(names, sizes):
            out[name] = float((r[k : k + n] ** 2).sum() / WEIGHTS[name])
            k += n
        return out


def errors(W):
    """保存した各時刻の窓内全格子点での相対 L2 誤差"""
    with torch.no_grad():
        pred = field(W, Gx[0], Gy[0], Gt_frames[0])
        diff = ((pred - exact_t) ** 2).sum(dim=(0, 2, 3)).sqrt()
        return (diff / (exact_t**2).sum(dim=(0, 2, 3)).sqrt()).numpy()


def least_squares_step(f, W):
    """min ||f(W + δ)||² を、f を W のまわりで線形化して CG（正規方程式）で解き、W + δ を返す"""
    r, pull = vjp(f, W)
    rhs = -pull(r)[0]
    delta = torch.zeros_like(W)
    res = rhs.clone()
    p = res.clone()
    rr = rr0 = (res * res).sum()
    for _ in range(args.cg_iters):
        Ap = pull(jvp(f, (W,), (p,))[1])[0]
        alpha = rr / (p * Ap).sum()
        delta += alpha * p
        res -= alpha * Ap
        rr_new = (res * res).sum()
        if rr_new < 1e-24 * rr0:
            break
        p = res + (rr_new / rr) * p
        rr = rr_new
    return W + delta


def gauss_newton(W, use_data, steps, log, label):
    """ガウス・ニュートン法: 残差を線形化した最小二乗を解いて W を更新することを繰り返す"""
    for step in range(steps):
        W = least_squares_step(lambda w: residuals(w, use_data), W)
        log(W, label, step + 1)
    return W


def fit_without_physics(W):
    """方程式を使わず、初期条件・境界条件・データだけを最小二乗で当てはめる（線形なので 1 回で解ける）"""
    def f(w):
        r = residuals(w, True)
        return r[n_phys:]  # 先頭の物理の残差を除く
    return least_squares_step(f, W)


# ---------------------------------------------------------------- 実行
metrics = open(args.out / "metrics.csv", "w", newline="")
writer = csv.writer(metrics)
writer.writerow(
    ["stage", "gn_step", "loss_physics", "loss_ic_phi", "loss_ic_pi", "loss_boundary", "loss_data",
     "error_data_times", "error_heldout_times", "error_max", "seconds"]
)
start = time.time()


def log(W, stage, step):
    l, e = losses(W), errors(W)
    ed, eh = e[is_data_time].mean(), e[~is_data_time].mean()
    writer.writerow(
        [stage, step] + [f"{l[k]:.4e}" for k in ["physics", "ic_phi", "ic_pi", "boundary", "data"]]
        + [f"{ed:.4e}", f"{eh:.4e}", f"{e.max():.4e}", f"{time.time() - start:.1f}"]
    )
    metrics.flush()
    print(
        f"{stage:12s} GN {step:2d}: 残差 {l['physics']:.2e} 誤差(データ時刻) {ed:.4f} "
        f"誤差(検証時刻) {eh:.4f} 最大 {e.max():.4f} {time.time() - start:6.0f} 秒",
        flush=True,
    )


W = torch.zeros(2, len(cx), len(cy), len(ct))
if args.mode == "physics_only":
    # データを使わない初期値: τ = 32 の φ を時間方向にそのまま伸ばしたものを最小二乗で当てはめる
    target = phi0_t[:, None].expand(2, len(t_col), len(ys), len(xs))
    W = least_squares_step(lambda w: (field(w, Gx[0], Gy[0], Gt_col[0]) - target).flatten(), W)
    log(W, "initial", 0)
    W = gauss_newton(W, False, args.gn_steps, log, "physics")
else:
    # データありの出発点: 方程式を使わずにデータ・初期条件・境界条件を当てはめたもの
    W = fit_without_physics(W)
    log(W, "initial", 0)
    W = gauss_newton(W, True, args.gn_steps, log, "with_data")
    if args.mode == "curriculum":
        W = gauss_newton(W, False, args.gn_steps, log, "without_data")

metrics.close()
with torch.no_grad():
    pred = field(W, Gx[0], Gy[0], Gt_frames[0]).numpy()
np.save(args.out / "prediction.npy", pred[0] + 1j * pred[1])
np.save(args.out / "taus.npy", taus)
(args.out / "config.json").write_text(
    json.dumps({"method": "rbf", **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "n_unknowns": n_unknowns, "weights": WEIGHTS}, indent=2)
)
