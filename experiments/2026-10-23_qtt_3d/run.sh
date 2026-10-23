#!/bin/sh
# 3 次元の交差するひもの衝突（2026-10-23_crossing_strings_3d と同じ設定）を、場全体を保存して計算し直し、
# いくつかの時刻を 3 次元の QTT（x, y, z のビットを交互に並べた 24 サイト）にする。
set -eu
cd "$(dirname "$0")/../.."
E=experiments/2026-10-23_qtt_3d
PYTHON="${PYTHON:-.venv/bin/python}"

# τ = 2 ごとに場全体（256³、1 時刻 268MB）を保存する
cargo run --release --bin simulate3d -- $E/data/crossing config=crossing nx=256 nz=256 length=32 tau_end=36 output_interval=2 separation=8 velocity=0.5 save_full=true
# スナップショット 0, 2, 3, 4, 8 は τ = 20（はじめ）、24（接近中）、26（組み換え直後）、28（ほどけている途中）、36（最後）
cargo run --release --bin qtt -- $E/data/crossing $E/data/qtt 0,2,3,4,8 1e-2,1e-3,1e-4
"$PYTHON" $E/plot.py $E/data $E/figures
