#!/bin/sh
# 円形のひものループ（半径 8、xy 平面、静止）が収縮する様子を、膨張なしと輻射優勢期の FLRW で計算し、
# 太さのない弦（南部・後藤の弦）の予言と比べる。
set -eu
cd "$(dirname "$0")/../.."
E=experiments/2026-10-23_loop_collapse
PYTHON="${PYTHON:-.venv/bin/python}"
COMMON="config=loop radius=8 nx=256 nz=256 length=32 output_interval=0.5 integrator=lsrk4"

cargo run --release --bin simulate3d -- $E/data/minkowski $COMMON expansion=none tau_end=36
cargo run --release --bin simulate3d -- $E/data/radiation $COMMON expansion=radiation tau_end=40
"$PYTHON" $E/analyze.py $E/data $E/figures
