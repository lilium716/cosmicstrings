#!/bin/sh
# RK4 でエネルギーが保存するかを調べる。
# 膨張なし（expansion=none）で時間刻み Δτ = courant × Δx を変えて、エネルギーの相対変化を比べる。
set -eu
cd "$(dirname "$0")/../.."
E=experiments/2026-10-21_energy_conservation
PYTHON="${PYTHON:-.venv/bin/python}"

for c in 0.4 0.2 0.1 0.05; do
    cargo run --release --bin simulate -- $E/data/none_c$c expansion=none courant=$c
done
"$PYTHON" $E/analyze.py $E/data/none_c0.4 $E/data/none_c0.2 $E/data/none_c0.1 $E/data/none_c0.05 --figure $E/figures/energy.png
