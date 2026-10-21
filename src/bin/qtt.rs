//! シミュレーションのスナップショット φ(x_i, y_j) を QTT（quantics tensor train）に圧縮し、
//! 全格子点で元データと比べて誤差を測る。
//!
//! n×n = 2^R × 2^R 格子上の場を、格子点の値を返す関数とみなして
//! tensor4all-rs の Quantics TCI（テンソルクロス補間）で 2R サイトの QTT にする。
//! ビットは x と y を上位から交互に並べる（interleaved 方式）。
//! TCI は格子の一部の点しか見ないので、QTT を全格子に展開して元データと比較する。
//!
//! 使い方: cargo run --release --bin qtt -- <データのディレクトリ> <出力先> <スナップショット番号,...> <許容誤差,...>
//! 例:     cargo run --release --bin qtt -- data out 0,10,20 1e-2,1e-4
//!
//! 出力: summary.csv（ランク・パラメータ数・圧縮率・誤差）、bonddims.csv（ボンドごとの次元）

use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use num_complex::Complex64;
use tensor4all_quanticstci::prelude::*;
use tensor4all_simplett::{AbstractTensorTrain, Tensor3Ops};

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err("使い方: qtt <データ> <出力先> <番号,...> <許容誤差,...>".into());
    }
    let data = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    let snapshots: Vec<usize> = args[3]
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let tolerances: Vec<f64> = args[4]
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    fs::create_dir_all(out)?;

    let mut summary = fs::File::create(out.join("summary.csv"))?;
    writeln!(
        summary,
        "snapshot,tau,tolerance,max_rank,n_params,compression,eval_fraction,max_abs_error,rel_l2_error,seconds"
    )?;
    let mut bonddims = fs::File::create(out.join("bonddims.csv"))?;
    writeln!(bonddims, "snapshot,tolerance,bond,dim")?;

    for &index in &snapshots {
        let tau = snapshot_tau(data, index)?;
        let (rows, cols, field) = read_npy(&data.join(format!("snapshot_{index:04}.npy")))?;
        if rows != cols || !rows.is_power_of_two() {
            return Err(format!("2 のべきの正方格子が必要です: {rows}×{cols}").into());
        }
        let n = rows;

        for &tol in &tolerances {
            let options = QtciOptions::default().with_tolerance(tol).with_rng_seed(0);
            let start = Instant::now();
            // 格子の添字 (i, j) は 0 始まり。field は行優先で field[j * n + i] = φ(x_i, y_j)
            let (qtci, _ranks, _errors) = quanticscrossinterpolate_discrete_batch::<Complex64, _>(
                &[n, n],
                pointwise_index_batch(|idx: &[usize]| field[idx[1] * n + idx[0]]),
                None,
                options,
            )?;
            let seconds = start.elapsed().as_secs_f64();

            // 全格子点に展開して元データと比べる
            let tt = qtci.tensor_train();
            let approx = to_grid(&tt, n)?;
            check_layout(&qtci, &approx, n)?;
            let mut max_abs = 0.0_f64;
            let (mut diff2, mut norm2) = (0.0, 0.0);
            for (a, e) in approx.iter().zip(&field) {
                let d = (a - e).norm();
                max_abs = max_abs.max(d);
                diff2 += d * d;
                norm2 += e.norm_sqr();
            }
            let rel_l2 = (diff2 / norm2).sqrt();

            let n_params: usize = tt.site_tensors().iter().map(|t| t.len()).sum();
            let compression = (n * n) as f64 / n_params as f64;
            let eval_fraction = qtci.num_evals() as f64 / (n * n) as f64;
            writeln!(
                summary,
                "{index},{tau},{tol:e},{},{n_params},{compression},{eval_fraction},{max_abs:e},{rel_l2:e},{seconds:.3}",
                qtci.rank()
            )?;
            for (bond, dim) in qtci.link_dims().iter().enumerate() {
                writeln!(bonddims, "{index},{tol:e},{bond},{dim}")?;
            }
            println!(
                "τ = {tau:4.0}, tol = {tol:.0e}: 最大ランク {:3}, 圧縮率 {compression:6.1}, \
                 最大誤差 {max_abs:.1e}, 相対 L2 誤差 {rel_l2:.1e}, 評価点 {:4.1}%",
                qtci.rank(),
                100.0 * eval_fraction
            );
        }
    }
    Ok(())
}

/// QTT を左から順に縮約して n×n 格子上の値（行優先、index = j * n + i）に展開する。
/// interleaved 方式では、サイト 2b が x の第 b ビット、サイト 2b+1 が y の第 b ビット（上位から）。
fn to_grid(tt: &SimpleTensorTrain<Complex64>, n: usize) -> Result<Vec<Complex64>, Error> {
    let cores = tt.site_tensors();
    let bits = n.trailing_zeros() as usize;
    if cores.len() != 2 * bits || cores.iter().any(|c| c.site_dim() != 2) {
        return Err("想定している interleaved 方式の QTT ではありません".into());
    }
    // left[p * r + a]: 先頭のサイトのビット列 p（上位から）と右ボンド a に対する部分縮約
    let mut left = vec![Complex64::new(1.0, 0.0)];
    for core in cores {
        let (rl, rr) = (core.left_dim(), core.right_dim());
        let prefixes = left.len() / rl;
        let mut next = vec![Complex64::new(0.0, 0.0); prefixes * 2 * rr];
        for p in 0..prefixes {
            for s in 0..2 {
                let row = &mut next[(p * 2 + s) * rr..(p * 2 + s + 1) * rr];
                for a in 0..rl {
                    let m = left[p * rl + a];
                    for (b, out) in row.iter_mut().enumerate() {
                        *out += m * core.get3(a, s, b);
                    }
                }
            }
        }
        left = next;
    }
    // ビット列 (x_{R-1} y_{R-1} … x_0 y_0) を (i, j) に戻す
    let mut grid = vec![Complex64::new(0.0, 0.0); n * n];
    for (q, value) in left.into_iter().enumerate() {
        let (mut i, mut j) = (0, 0);
        for b in 0..bits {
            let pair = (q >> (2 * (bits - 1 - b))) & 0b11;
            i = (i << 1) | (pair >> 1);
            j = (j << 1) | (pair & 1);
        }
        grid[j * n + i] = value;
    }
    Ok(grid)
}

/// to_grid のビットの並びの想定が正しいことを、ライブラリの evaluate と比べて確かめる
fn check_layout(
    qtci: &QuanticsTensorCI2<Complex64>,
    grid: &[Complex64],
    n: usize,
) -> Result<(), Error> {
    for k in 0..1000u64 {
        let h = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (i, j) = ((h >> 20) as usize % n, (h >> 40) as usize % n);
        let expected = qtci.evaluate(&[i, j])?;
        if (expected - grid[j * n + i]).norm() > 1e-10 * (1.0 + expected.norm()) {
            return Err(format!("QTT の展開が evaluate と一致しません: ({i}, {j})").into());
        }
    }
    Ok(())
}

/// snapshots.csv から番号 index の時刻 τ を読む
fn snapshot_tau(data: &Path, index: usize) -> Result<f64, Error> {
    let text = fs::read_to_string(data.join("snapshots.csv"))?;
    for line in text.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols[0].parse::<usize>()? == index {
            return Ok(cols[1].parse()?);
        }
    }
    Err(format!("スナップショット {index} が snapshots.csv にありません").into())
}

/// simulate.rs が書いた .npy（complex128、C 順序の 2 次元）を読む。戻り値は (rows, cols, data)
fn read_npy(path: &Path) -> Result<(usize, usize, Vec<Complex64>), Error> {
    let bytes = fs::read(path)?;
    if bytes.len() < 10 || &bytes[..8] != b"\x93NUMPY\x01\x00" {
        return Err("npy バージョン 1.0 のファイルではありません".into());
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len])?;
    if !header.contains("'descr': '<c16'") || !header.contains("'fortran_order': False") {
        return Err("complex128・C 順序の配列のみ対応しています".into());
    }
    let shape = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or("shape が読めません")?;
    let dims: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let [rows, cols] = dims[..] else {
        return Err("2 次元配列ではありません".into());
    };
    let body = &bytes[10 + header_len..];
    if body.len() != rows * cols * 16 {
        return Err("データの長さが shape と一致しません".into());
    }
    let data = body
        .chunks_exact(16)
        .map(|c| {
            Complex64::new(
                f64::from_le_bytes(c[..8].try_into().unwrap()),
                f64::from_le_bytes(c[8..].try_into().unwrap()),
            )
        })
        .collect();
    Ok((rows, cols, data))
}
