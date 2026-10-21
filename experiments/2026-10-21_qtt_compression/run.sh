#!/bin/sh
# 大域的ひもの対消滅（2026-10-21_global_string_flrw）のいくつかの場面を QTT にして、圧縮率と誤差を調べる。
set -eu
cd "$(dirname "$0")/../.."
E=experiments/2026-10-21_qtt_compression
SRC=experiments/2026-10-21_global_string_flrw/data
PYTHON="${PYTHON:-.venv/bin/python}"

# スナップショット番号 k は τ = 20 + 2k（0: 初期、9: 衝突直前、10: 対消滅直後）
cargo run --release --bin qtt -- $SRC $E/data 0,5,8,9,10,15,20 1e-2,1e-3,1e-4,1e-5,1e-6
"$PYTHON" $E/plot.py $E/data $E/figures
