"""RK4 の計算結果を使って PINN（physics-informed neural network）を学習する（この実験専用）。

解く方程式（共形時間 τ、輻射優勢期 a = τ/τ_i、ℋ = 1/τ）:
    φ'' + 2ℋφ' − ∇²φ + a² λ(|φ|² − η²)φ = 0

学習する範囲（窓）: x ∈ [−8, 8]、y ∈ [−6, 6]、τ ∈ [32, 42]（衝突と対消滅を含む）
ネットワーク: (x, y, τ) → (Re φ, Im φ)。入力にランダムなフーリエ特徴量を付けた全結合ネットワーク

損失:
    物理       方程式の残差の 2 乗平均（窓の中のランダムな点、自動微分で計算）
    初期条件   τ = 32 での φ と φ′（RK4 の値）
    境界条件   窓の縁での φ（RK4 の値、保存した時刻のみ）
    データ     窓の内部での φ（RK4 の値、整数の時刻 τ = 33, 34, …, 42 のみ）

--mode curriculum   : 最初は 4 つすべてで学習し、データ時刻での相対 L2 誤差が --switch を下回ったら
                      データを外して、物理・初期条件・境界条件だけで続ける
--mode physics_only : 最初からデータを使わない（比較用）
--mode data_physics : 最後までデータを使う（比較用）

評価: 保存した全時刻（τ = 32, 32.5, …, 42）の窓内の全格子点で、RK4 との相対 L2 誤差を求める。
半整数の時刻（32.5, 33.5, …）は学習に一度も使っていない検証用の時刻である。

出力（--out）: metrics.csv（損失と誤差の履歴）、prediction.npy（学習後の予測、shape [時刻, ny, nx] の複素数）、
             model.pt、config.json
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- 設定
parser = argparse.ArgumentParser()
parser.add_argument("--rk4", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--mode", choices=["curriculum", "physics_only", "data_physics"], required=True)
parser.add_argument("--iterations", type=int, default=10000, help="全体の反復回数")
parser.add_argument("--switch", type=float, default=0.02, help="データを外す相対 L2 誤差のしきい値")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

X_RANGE, Y_RANGE, T_RANGE = (-8.0, 8.0), (-6.0, 6.0), (32.0, 42.0)
WEIGHTS = {"physics": 1.0, "ic_phi": 10.0, "ic_pi": 10.0, "boundary": 10.0, "data": 10.0}
BATCH = {"physics": 4096, "ic": 2048, "boundary": 2048, "data": 4096}
EVAL_EVERY = 250

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------------- RK4 のデータを読む
params = json.loads((args.rk4 / "params.json").read_text())
eta, lam, tau_i = params["eta"], params["lambda"], params["tau_initial"]
dx, length = params["dx"], params["length"]
grid = np.arange(params["nx"]) * dx - length / 2 - params["center_offset"]
ix = np.where((grid >= X_RANGE[0] - 1e-9) & (grid <= X_RANGE[1] + 1e-9))[0]
iy = np.where((grid >= Y_RANGE[0] - 1e-9) & (grid <= Y_RANGE[1] + 1e-9))[0]
xs, ys = grid[ix], grid[iy]
# 窓を格子点にそろえる（center_offset のぶんずれる）
X_RANGE, Y_RANGE = (xs[0], xs[-1]), (ys[0], ys[-1])

frames = []  # (τ, φ の窓 [ny, nx]、π の窓)
with open(args.rk4 / "snapshots.csv") as f:
    for row in csv.DictReader(f):
        tau = float(row["tau"])
        if T_RANGE[0] - 1e-9 <= tau <= T_RANGE[1] + 1e-9:
            phi = np.load(args.rk4 / row["file"])[np.ix_(iy, ix)]
            pi = np.load(args.rk4 / row["file"].replace("snapshot", "momentum"))[np.ix_(iy, ix)]
            frames.append((tau, phi, pi))
taus = np.array([f[0] for f in frames])
is_data_time = np.isclose(taus, np.round(taus)) & (taus > T_RANGE[0] + 1e-9)
print(f"窓: x {X_RANGE}, y {Y_RANGE}, τ {T_RANGE}, 格子 {len(xs)}×{len(ys)}, 時刻 {len(taus)} 個")
print(f"データに使う時刻: {taus[is_data_time]}")

XX, YY = np.meshgrid(xs, ys)  # [ny, nx]


def to_tensor(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)


# 初期条件（τ = 32 の全格子点）
_, phi0, pi0 = frames[0]
ic_xyt = to_tensor(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, taus[0])], axis=1))
ic_phi = to_tensor(np.stack([phi0.real.ravel(), phi0.imag.ravel()], axis=1))
ic_pi = to_tensor(np.stack([pi0.real.ravel(), pi0.imag.ravel()], axis=1))

# 境界条件（窓の縁の格子点、保存した全時刻）
edge = np.zeros_like(XX, dtype=bool)
edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
bc_xyt, bc_phi = [], []
for tau, phi, _ in frames:
    bc_xyt.append(np.stack([XX[edge], YY[edge], np.full(edge.sum(), tau)], axis=1))
    bc_phi.append(np.stack([phi[edge].real, phi[edge].imag], axis=1))
bc_xyt, bc_phi = to_tensor(np.concatenate(bc_xyt)), to_tensor(np.concatenate(bc_phi))

# データ（窓の内部、整数の時刻のみ）
interior = ~edge
d_xyt, d_phi = [], []
for (tau, phi, _), use in zip(frames, is_data_time):
    if use:
        d_xyt.append(np.stack([XX[interior], YY[interior], np.full(interior.sum(), tau)], axis=1))
        d_phi.append(np.stack([phi[interior].real, phi[interior].imag], axis=1))
d_xyt, d_phi = to_tensor(np.concatenate(d_xyt)), to_tensor(np.concatenate(d_phi))

# 評価用（全時刻の窓内の全格子点）
eval_xyt = [to_tensor(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, t)], axis=1)) for t in taus]
eval_phi = [phi for _, phi, _ in frames]


# ---------------------------------------------------------------- ネットワーク
class PINN(torch.nn.Module):
    def __init__(self, features=128, sigma=1.0, width=128, depth=4):
        super().__init__()
        lo = torch.tensor([X_RANGE[0], Y_RANGE[0], T_RANGE[0]], dtype=torch.float32)
        hi = torch.tensor([X_RANGE[1], Y_RANGE[1], T_RANGE[1]], dtype=torch.float32)
        self.register_buffer("center", (lo + hi) / 2)
        self.register_buffer("radius", (hi - lo) / 2)
        # ランダムなフーリエ特徴量 [sin(2π B z), cos(2π B z)]、z は [−1, 1] に正規化した入力
        self.register_buffer("B", torch.randn(3, features) * sigma)
        layers, d = [], 2 * features
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers.append(torch.nn.Linear(d, 2))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xyt):
        z = (xyt - self.center) / self.radius
        proj = 2 * torch.pi * z @ self.B
        return self.net(torch.cat([torch.sin(proj), torch.cos(proj)], dim=1))


model = PINN().to(device)


def grad(out, inp):
    return torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True)[0]


def residual(xyt):
    """方程式の残差 φ'' + 2ℋφ' − ∇²φ + a²λ(|φ|² − η²)φ（実部と虚部）"""
    xyt = xyt.requires_grad_(True)
    phi = model(xyt)
    res = []
    for c in range(2):
        g = grad(phi[:, c], xyt)
        gx, gy, gt = g[:, 0], g[:, 1], g[:, 2]
        gxx = grad(gx, xyt)[:, 0]
        gyy = grad(gy, xyt)[:, 1]
        gtt = grad(gt, xyt)[:, 2]
        res.append((gtt, gt, gxx + gyy))
    tau = xyt[:, 2]
    a2 = (tau / tau_i) ** 2
    hubble = 1.0 / tau
    potential = a2 * lam * (phi[:, 0] ** 2 + phi[:, 1] ** 2 - eta**2)
    out = [gtt + 2 * hubble * gt - lap + potential * phi[:, c] for c, (gtt, gt, lap) in enumerate(res)]
    return torch.stack(out, dim=1)


def time_derivative(xyt):
    xyt = xyt.requires_grad_(True)
    phi = model(xyt)
    return torch.stack([grad(phi[:, c], xyt)[:, 2] for c in range(2)], dim=1)


def sample(xyt, values, n):
    idx = torch.randint(0, xyt.shape[0], (n,), device=device)
    return xyt[idx], (values[idx] if values is not None else None)


def random_points(n):
    lo = torch.tensor([X_RANGE[0], Y_RANGE[0], T_RANGE[0]], dtype=torch.float32, device=device)
    hi = torch.tensor([X_RANGE[1], Y_RANGE[1], T_RANGE[1]], dtype=torch.float32, device=device)
    return lo + (hi - lo) * torch.rand(n, 3, device=device)


@torch.no_grad()
def evaluate():
    """各時刻の窓内全格子点での相対 L2 誤差"""
    errors = []
    for xyt, phi in zip(eval_xyt, eval_phi):
        pred = model(xyt).cpu().numpy()
        pred = (pred[:, 0] + 1j * pred[:, 1]).reshape(phi.shape)
        errors.append(np.linalg.norm(pred - phi) / np.linalg.norm(phi))
    return np.array(errors)


# ---------------------------------------------------------------- 学習
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / args.iterations))
use_data = args.mode in ("curriculum", "data_physics")
switch_iteration = None

metrics = open(args.out / "metrics.csv", "w", newline="")
writer = csv.writer(metrics)
writer.writerow(
    ["iteration", "use_data", "loss_physics", "loss_ic_phi", "loss_ic_pi", "loss_boundary", "loss_data",
     "error_data_times", "error_heldout_times", "error_max", "seconds"]
)
start = time.time()
for it in range(args.iterations + 1):
    optimizer.zero_grad()
    losses = {}
    losses["physics"] = residual(random_points(BATCH["physics"])).pow(2).mean()
    p, v = sample(ic_xyt, ic_phi, BATCH["ic"])
    losses["ic_phi"] = (model(p) - v).pow(2).mean()
    p, v = sample(ic_xyt, ic_pi, BATCH["ic"])
    losses["ic_pi"] = (time_derivative(p) - v).pow(2).mean()
    p, v = sample(bc_xyt, bc_phi, BATCH["boundary"])
    losses["boundary"] = (model(p) - v).pow(2).mean()
    p, v = sample(d_xyt, d_phi, BATCH["data"])
    losses["data"] = (model(p) - v).pow(2).mean()  # データを外した後も記録のために計算する

    total = sum(WEIGHTS[k] * l for k, l in losses.items() if k != "data" or use_data)
    if it < args.iterations:
        total.backward()
        optimizer.step()
        scheduler.step()

    if it % EVAL_EVERY == 0:
        err = evaluate()
        err_data, err_held = err[is_data_time].mean(), err[~is_data_time].mean()
        writer.writerow(
            [it, int(use_data)] + [f"{losses[k].item():.4e}" for k in ["physics", "ic_phi", "ic_pi", "boundary", "data"]]
            + [f"{err_data:.4e}", f"{err_held:.4e}", f"{err.max():.4e}", f"{time.time() - start:.1f}"]
        )
        metrics.flush()
        print(
            f"{it:6d} {'data+物理' if use_data else '物理のみ':6s} 残差 {losses['physics'].item():.2e} "
            f"誤差(データ時刻) {err_data:.3f} 誤差(検証時刻) {err_held:.3f} 最大 {err.max():.3f} "
            f"{time.time() - start:6.0f} 秒",
            flush=True,
        )
        if args.mode == "curriculum" and use_data and err_data < args.switch:
            use_data = False
            switch_iteration = it
            print(f"--- 反復 {it} でデータを外す（データ時刻の誤差 {err_data:.3f} < {args.switch}）", flush=True)

metrics.close()
torch.save(model.state_dict(), args.out / "model.pt")
with torch.no_grad():
    prediction = np.stack([(lambda q: (q[:, 0] + 1j * q[:, 1]).reshape(XX.shape))(model(xyt).cpu().numpy()) for xyt in eval_xyt])
np.save(args.out / "prediction.npy", prediction)
np.save(args.out / "taus.npy", taus)
(args.out / "config.json").write_text(
    json.dumps(
        {
            "mode": args.mode,
            "iterations": args.iterations,
            "switch_threshold": args.switch,
            "switch_iteration": switch_iteration,
            "seed": args.seed,
            "weights": WEIGHTS,
            "batch": BATCH,
            "window": {"x": X_RANGE, "y": Y_RANGE, "tau": T_RANGE},
            "device": str(device),
        },
        indent=2,
        default=float,
    )
)
