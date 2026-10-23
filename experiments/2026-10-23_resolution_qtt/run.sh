#!/bin/sh
# 格子を細かくして RK4 を回し（箱の大きさは同じ）、いくつかの時刻を QTT にして圧縮率を比べる。
set -eu
cd "$(dirname "$0")/../.."
E=experiments/2026-10-23_resolution_qtt
PYTHON="${PYTHON:-.venv/bin/python}"

for n in 256 512 1024 2048; do
    # τ = 4 ごとに保存（τ = 20, 24, …, 60）。それ以外の設定は 2026-10-21_global_string_flrw と同じ
    cargo run --release --bin simulate -- $E/data/n$n nx=$n output_interval=4
    # スナップショット 0, 4, 5, 10 は τ = 20（初期）、36（接近中）、40（対消滅直後）、60（放射が広がった後）
    cargo run --release --bin qtt -- $E/data/n$n $E/data/qtt_n$n 0,4,5,10 1e-2,1e-4,1e-6
done
"$PYTHON" $E/analyze.py $E/data $E/figures
