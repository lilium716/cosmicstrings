//! シミュレーションのスナップショット φ を QTT（quantics tensor train）に圧縮し、
//! 全格子点で元データと比べて誤差を測る。2 次元・3 次元の両方に使える。
//!
//! 一辺 n = 2^R の d 次元格子（d = 2 または 3）上の場を、格子点の値を返す関数とみなして
//! tensor4all-rs の Quantics TCI（テンソルクロス補間）で dR サイトの QTT にする。
//! ビットは座標ごとに上位から交互に並べる（interleaved 方式: x の最上位, y の最上位, (z の最上位), x の次, …）。
//! TCI は格子の一部の点しか見ないので、QTT を全格子に展開して元データと比較する。
//!
//! 使い方: cargo run --release --bin qtt -- <データのディレクトリ> <出力先> <スナップショット番号,...> <許容誤差,...>
//! 例:     cargo run --release --bin qtt -- data out 0,10,20 1e-2,1e-4
//!
//! 入力: snapshots.csv（index, tau, …）と snapshot_XXXX.npy（complex128、shape (n, n) または (n, n, n)、
//!       配列[j, i] = φ(x_i, y_j)、3 次元なら配列[k, j, i] = φ(x_i, y_j, z_k)）
//! 出力: summary.csv（ランク・パラメータ数・圧縮率・誤差）、bonddims.csv（ボンドごとの次元）

use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use num_complex::Complex64;
use rayon::prelude::*;
use tensor4all_quanticstci::prelude::*;
use tensor4all_simplett::{AbstractTensorTrain, Tensor3Ops};

use cosmicstrings::io::npy::read_complex;

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
        let (shape, field) = read_complex(&data.join(format!("snapshot_{index:04}.npy")))?;
        let n = shape[0];
        let d = shape.len();
        if !(d == 2 || d == 3) || shape.iter().any(|&s| s != n) || !n.is_power_of_two() {
            return Err(format!(
                "一辺が 2 のべきの 2 次元または 3 次元の格子が必要です: {shape:?}"
            )
            .into());
        }
        let total = field.len();
        // 格子の添字 idx = (i, j[, k]) は 0 始まり。field は行優先で i が最も速く変わる
        let flat = move |idx: &[usize]| idx.iter().rev().fold(0, |acc, &v| acc * n + v);

        for &tol in &tolerances {
            let options = QtciOptions::default().with_tolerance(tol).with_rng_seed(0);
            let start = Instant::now();
            let (qtci, _ranks, _errors) = quanticscrossinterpolate_discrete_batch::<Complex64, _>(
                &vec![n; d],
                pointwise_index_batch(|idx: &[usize]| field[flat(idx)]),
                None,
                options,
            )?;
            let seconds = start.elapsed().as_secs_f64();

            // 全格子点に展開して元データと比べる
            let tt = qtci.tensor_train();
            let approx = to_grid(&tt, n, d)?;
            check_layout(&qtci, &approx, n, d)?;
            let (max_abs, diff2, norm2) = approx
                .par_iter()
                .zip(field.par_iter())
                .map(|(a, e)| {
                    let diff = (a - e).norm();
                    (diff, diff * diff, e.norm_sqr())
                })
                .reduce(
                    || (0.0, 0.0, 0.0),
                    |x, y| (x.0.max(y.0), x.1 + y.1, x.2 + y.2),
                );
            let rel_l2 = (diff2 / norm2).sqrt();

            let n_params: usize = tt.site_tensors().iter().map(|t| t.len()).sum();
            let compression = total as f64 / n_params as f64;
            let eval_fraction = qtci.num_evals() as f64 / total as f64;
            writeln!(
                summary,
                "{index},{tau},{tol:e},{},{n_params},{compression},{eval_fraction},{max_abs:e},{rel_l2:e},{seconds:.3}",
                qtci.rank()
            )?;
            for (bond, dim) in qtci.link_dims().iter().enumerate() {
                writeln!(bonddims, "{index},{tol:e},{bond},{dim}")?;
            }
            println!(
                "τ = {tau:4.0}, tol = {tol:.0e}: 最大ランク {:3}, 圧縮率 {compression:8.1}, \
                 最大誤差 {max_abs:.1e}, 相対 L2 誤差 {rel_l2:.1e}, 評価点 {:5.2}%, TCI {seconds:.1} 秒",
                qtci.rank(),
                100.0 * eval_fraction
            );
        }
    }
    Ok(())
}

/// QTT を d 次元の格子上の値（行優先、i が最も速く変わる）に展開する。
/// サイトを真ん中で左右に分け、左側のビット列と右側のビット列それぞれについて部分縮約を作り、
/// 最後に真ん中のボンドで掛け合わせる（途中の配列が大きくなりすぎないように）。
/// interleaved 方式では、サイト d·b + c が座標 c（0 = x, 1 = y, 2 = z）の第 b ビット（上位から）。
fn to_grid(tt: &SimpleTensorTrain<Complex64>, n: usize, d: usize) -> Result<Vec<Complex64>, Error> {
    let cores = tt.site_tensors();
    let bits = n.trailing_zeros() as usize;
    let sites = d * bits;
    if cores.len() != sites || cores.iter().any(|c| c.site_dim() != 2) {
        return Err("想定している interleaved 方式の QTT ではありません".into());
    }
    let zero = Complex64::new(0.0, 0.0);
    let mid = sites / 2;

    // left[p * r + a]: 先頭 mid サイトのビット列 p（上位から）と、右ボンド a に対する部分縮約
    let mut left = vec![Complex64::new(1.0, 0.0)];
    for core in &cores[..mid] {
        let (rl, rr) = (core.left_dim(), core.right_dim());
        let prefixes = left.len() / rl;
        let mut next = vec![zero; prefixes * 2 * rr];
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
    // right[a * m + q]: 左ボンド a と、残りのサイトのビット列 q（上位から、長さ m = 2^(sites − mid)）に対する部分縮約
    let mut right = vec![Complex64::new(1.0, 0.0)];
    let mut suffixes = 1usize;
    for core in cores[mid..].iter().rev() {
        let (rl, rr) = (core.left_dim(), core.right_dim());
        let mut next = vec![zero; rl * 2 * suffixes];
        for a in 0..rl {
            for s in 0..2 {
                let out = &mut next
                    [a * 2 * suffixes + s * suffixes..a * 2 * suffixes + (s + 1) * suffixes];
                for b in 0..rr {
                    let m = core.get3(a, s, b);
                    let src = &right[b * suffixes..(b + 1) * suffixes];
                    for (o, v) in out.iter_mut().zip(src) {
                        *o += m * v;
                    }
                }
            }
        }
        right = next;
        suffixes *= 2;
    }
    let r = cores[mid].left_dim();
    let prefixes = 1usize << mid;

    // grid[q]（q は全サイトのビット列）= Σ_a left[p, a] right[a, s]、q = p · 2^(sites − mid) + s
    let mut by_bits = vec![zero; prefixes * suffixes];
    by_bits
        .par_chunks_mut(suffixes)
        .enumerate()
        .for_each(|(p, row)| {
            for a in 0..r {
                let m = left[p * r + a];
                for (o, v) in row.iter_mut().zip(&right[a * suffixes..(a + 1) * suffixes]) {
                    *o += m * v;
                }
            }
        });
    // ビット列を格子の添字に戻す
    let mut grid = vec![zero; n.pow(d as u32)];
    for (q, value) in by_bits.into_iter().enumerate() {
        let mut idx = [0usize; 3];
        for b in 0..bits {
            let chunk = (q >> (d * (bits - 1 - b))) & ((1 << d) - 1);
            for (c, v) in idx.iter_mut().enumerate().take(d) {
                *v = (*v << 1) | ((chunk >> (d - 1 - c)) & 1);
            }
        }
        let flat = idx[..d].iter().rev().fold(0, |acc, &v| acc * n + v);
        grid[flat] = value;
    }
    Ok(grid)
}

/// to_grid のビットの並びの想定が正しいことを、ライブラリの evaluate と比べて確かめる
fn check_layout(
    qtci: &QuanticsTensorCI2<Complex64>,
    grid: &[Complex64],
    n: usize,
    d: usize,
) -> Result<(), Error> {
    for k in 0..1000u64 {
        let h = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let idx: Vec<usize> = (0..d).map(|c| (h >> (13 + 17 * c)) as usize % n).collect();
        let expected = qtci.evaluate(&idx)?;
        let flat = idx.iter().rev().fold(0, |acc, &v| acc * n + v);
        if (expected - grid[flat]).norm() > 1e-10 * (1.0 + expected.norm()) {
            return Err(format!("QTT の展開が evaluate と一致しません: {idx:?}").into());
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
