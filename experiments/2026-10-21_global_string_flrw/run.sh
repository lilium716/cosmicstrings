#!/bin/sh
# この実験で実行したコマンド。
set -eu
cd "$(dirname "$0")/../.."
E=experiments/2026-10-21_global_string_flrw
PYTHON="${PYTHON:-.venv/bin/python}"

cargo run --release --bin simulate -- $E/data
"$PYTHON" python/plot.py $E/data $E/figures
