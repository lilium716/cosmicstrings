#!/bin/sh
# 3 次元の大域的ひも。
# (1) 確認: z 方向に周期的な薄い格子（512×512×4）で平行なひもを解き、2 次元（2026-10-21_global_string_flrw）と一致するか見る
# (2) 本番: 向きの違う 2 本（z 方向と y 方向）を x 方向から衝突させる（256³、箱 32、Δx = 0.125）
set -eu
cd "$(dirname "$0")/../.."
E=experiments/2026-10-23_crossing_strings_3d
PYTHON="${PYTHON:-.venv/bin/python}"

cargo run --release --bin simulate3d -- $E/data/parallel_check config=parallel nx=512 nz=4 length=64 periodic_z=true tau_end=60 separation=16
cargo run --release --bin simulate3d -- $E/data/crossing config=crossing nx=256 nz=256 length=32 tau_end=36 output_interval=1 separation=8 velocity=0.5
# アニメーション用（τ = 0.25 ごとに出力、それ以外は同じ）
cargo run --release --bin simulate3d -- $E/data/crossing_fine config=crossing nx=256 nz=256 length=32 tau_end=36 output_interval=0.25 separation=8 velocity=0.5
"$PYTHON" $E/plot.py $E/data $E/figures
"$PYTHON" $E/animate.py $E/data/crossing_fine $E/figures
