#!/bin/sh
# 3 次元の交差するひもの衝突を 512³ で計算する。
# 箱を 64 に広げ（Δx = 0.125 は 2026-10-23_crossing_strings_3d と同じ）、壁の影響が組み換えに出ていないかを
# 256³（箱 32）の結果と比べる。メモリを抑えるため 2N-storage の RK（1 点 64 バイト、約 8.6GB）を使う。
set -eu
cd "$(dirname "$0")/../.."
E=experiments/2026-10-23_crossing_512
PYTHON="${PYTHON:-.venv/bin/python}"

cargo run --release --bin simulate3d -- $E/data/crossing config=crossing nx=512 nz=512 length=64 tau_end=36 output_interval=1 separation=8 velocity=0.5 integrator=lsrk4
"$PYTHON" experiments/2026-10-23_crossing_strings_3d/animate.py $E/data/crossing $E/figures
"$PYTHON" $E/compare.py $E/data/crossing experiments/2026-10-23_crossing_strings_3d/data/crossing $E/figures
