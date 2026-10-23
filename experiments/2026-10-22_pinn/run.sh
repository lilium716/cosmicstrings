#!/bin/sh
# RK4 の計算結果を学習データにして PINN を学習する。
# ある程度学習できたらデータを外し、物理（方程式の残差）・初期条件・境界条件だけで学習を続ける。
# 比較として、最初からデータを使わない学習と、最後までデータを使う学習も行う。
set -eu
cd "$(dirname "$0")/../.."
E=experiments/2026-10-22_pinn
PYTHON="${PYTHON:-.venv/bin/python}"

# 学習データ: 2026-10-21_global_string_flrw と同じ設定で、τ = 0.5 ごとに φ と π = φ′ を保存（τ = 42 まで）
cargo run --release --bin simulate -- $E/data/rk4 tau_end=42 output_interval=0.5 save_momentum=true

"$PYTHON" $E/pinn.py --rk4 $E/data/rk4 --out $E/data/curriculum --mode curriculum
"$PYTHON" $E/pinn.py --rk4 $E/data/rk4 --out $E/data/physics_only --mode physics_only
"$PYTHON" $E/pinn.py --rk4 $E/data/rk4 --out $E/data/data_physics --mode data_physics
# tenferro-rs 版（Rust、CPU）。PyTorch 版と同じ設定で「データ → 物理のみ」を学習する
cargo run --release --bin pinn -- --rk4 $E/data/rk4 --out $E/data/tenferro_curriculum --mode curriculum --threads 4

# ニューラルネットワークの代わりに、時空のガウス型動径基底関数（RBF）とガウス・ニュートン法で同じ問題を解く
for mode in curriculum physics_only data_physics; do
    "$PYTHON" $E/rbf.py --rk4 $E/data/rk4 --out $E/data/rbf_$mode --mode $mode
done

"$PYTHON" $E/plot.py $E/data $E/figures
