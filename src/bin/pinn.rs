//! RK4 の計算結果を使って PINN（physics-informed neural network）を学習する。tenferro-rs 版。
//! experiments/2026-10-22_pinn/pinn.py（PyTorch 版）と同じ問題設定・損失・切り替えの規則にしている。
//!
//! 解く方程式（共形時間 τ、輻射優勢期 a = τ/τ_i、ℋ = 1/τ）:
//!     φ'' + 2ℋφ' − ∇²φ + a² λ(|φ|² − η²)φ = 0
//! 学習する範囲（窓）: x ∈ [−8, 8]、y ∈ [−6, 6]、τ ∈ [32, 42]
//! ネットワーク: (x, y, τ) → (Re φ, Im φ)。ランダムなフーリエ特徴量 + tanh の全結合層
//!
//! 損失: 物理（方程式の残差）、初期条件（τ = 32 の φ と φ′）、境界条件（窓の縁の φ）、
//!       データ（窓の内部の φ、整数の時刻 τ = 33, …, 42 のみ）
//! 微分は tenferro-rs の traced 自動微分で計算する。入力についての微分は jvp（方向微分）、
//! パラメータについての勾配は grad。パラメータは 1 本のベクトル θ にまとめ、層ごとに切り出して使う。
//!
//! 使い方: cargo run --release --bin pinn -- --rk4 <RK4 の出力> --out <出力先> --mode <curriculum|physics_only|data_physics>
//!         [--iterations 10000] [--switch 0.02] [--seed 0] [--threads 4]
//! 出力: metrics.csv、prediction.npy（[時刻, ny, nx] の複素数）、taus.npy、config.json

use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use num_complex::Complex64;
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::{CpuBackend, runtime_engine_registration};
use tenferro_runtime::{
    CompiledGraph, DType, DotGeneralConfig, GraphCompiler, Runtime, TracedTensor,
};
use tenferro_tensor::{SliceConfig, Tensor};

use cosmicstrings::io::npy;

type Error = Box<dyn std::error::Error>;
type R<T> = tenferro_runtime::Result<T>;

const X_RANGE: (f64, f64) = (-8.0, 8.0);
const Y_RANGE: (f64, f64) = (-6.0, 6.0);
const T_RANGE: (f64, f64) = (32.0, 42.0);
// 損失の重み（PyTorch 版と同じ）
const W_PHYSICS: f64 = 1.0;
const W_IC: f64 = 10.0;
const W_BOUNDARY: f64 = 10.0;
const W_DATA: f64 = 10.0;
// 1 反復で使う点の数
const N_RES: usize = 4096;
const N_IC: usize = 2048;
const N_BC: usize = 2048;
const N_DATA: usize = 4096;
const EVAL_EVERY: usize = 250;
// ネットワークの大きさ（PyTorch 版と同じ）
const FEATURES: usize = 128;
const SIGMA: f64 = 1.0;
const WIDTH: usize = 128;
const DEPTH: usize = 4;

struct Args {
    rk4: PathBuf,
    out: PathBuf,
    mode: String,
    iterations: usize,
    switch: f64,
    seed: u64,
    threads: usize,
}

fn parse_args() -> Result<Args, Error> {
    let mut a = Args {
        rk4: PathBuf::new(),
        out: PathBuf::new(),
        mode: String::new(),
        iterations: 10000,
        switch: 0.02,
        seed: 0,
        threads: 4,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    for pair in argv.chunks(2) {
        let [key, value] = pair else {
            return Err("引数は --key value の組で指定してください".into());
        };
        match key.as_str() {
            "--rk4" => a.rk4 = value.into(),
            "--out" => a.out = value.into(),
            "--mode" => a.mode = value.clone(),
            "--iterations" => a.iterations = value.parse()?,
            "--switch" => a.switch = value.parse()?,
            "--seed" => a.seed = value.parse()?,
            "--threads" => a.threads = value.parse()?,
            _ => return Err(format!("未知の引数です: {key}").into()),
        }
    }
    if !["curriculum", "physics_only", "data_physics"].contains(&a.mode.as_str()) {
        return Err("--mode は curriculum, physics_only, data_physics のどれかです".into());
    }
    Ok(a)
}

/// 乱数（SplitMix64）
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// [0, 1) の一様乱数
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// 標準正規分布（ボックス＝ミュラー法）
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

// ---------------------------------------------------------------- ネットワーク

/// 全結合層 1 枚分のパラメータの、θ の中での位置
struct Layer {
    offset: usize,
    fan_in: usize,
    fan_out: usize,
}

/// フーリエ特徴量 + tanh の全結合ネットワーク。パラメータは 1 本のベクトル θ にまとめる
struct Net {
    /// フーリエ特徴量の行列 B（3 × FEATURES、学習しない）
    fourier: Vec<f64>,
    /// 1 層目は sin 部分と cos 部分の重みを分けて持つ（連結の代わり）
    first_sin: Layer,
    first_cos: Layer,
    first_bias: usize,
    hidden: Vec<(Layer, usize)>,
    output: (Layer, usize),
    n_params: usize,
}

impl Net {
    fn new(rng: &mut Rng) -> Self {
        let fourier = (0..3 * FEATURES).map(|_| SIGMA * rng.normal()).collect();
        let mut offset = 0;
        let mut layer = |fan_in, fan_out| {
            let l = Layer {
                offset,
                fan_in,
                fan_out,
            };
            offset += fan_in * fan_out;
            l
        };
        let first_sin = layer(FEATURES, WIDTH);
        let first_cos = layer(FEATURES, WIDTH);
        let mut biases = Vec::new();
        let hidden_layers: Vec<Layer> = (1..DEPTH).map(|_| layer(WIDTH, WIDTH)).collect();
        let out_layer = layer(WIDTH, 2);
        let mut b_offset = offset;
        for len in std::iter::once(WIDTH)
            .chain((1..DEPTH).map(|_| WIDTH))
            .chain([2])
        {
            biases.push(b_offset);
            b_offset += len;
        }
        let hidden = hidden_layers
            .into_iter()
            .zip(biases[1..DEPTH].iter().copied())
            .collect();
        Net {
            fourier,
            first_sin,
            first_cos,
            first_bias: biases[0],
            hidden,
            output: (out_layer, biases[DEPTH]),
            n_params: b_offset,
        }
    }

    /// Xavier の一様分布で重みを、0 でバイアスを初期化する
    fn init(&self, rng: &mut Rng) -> Vec<f64> {
        let mut theta = vec![0.0; self.n_params];
        let layers = [&self.first_sin, &self.first_cos]
            .into_iter()
            .chain(self.hidden.iter().map(|(l, _)| l))
            .chain([&self.output.0]);
        for l in layers {
            let fan_in = if l.offset == self.first_cos.offset || l.offset == self.first_sin.offset {
                2 * FEATURES
            } else {
                l.fan_in
            };
            let scale = (6.0 / (fan_in + l.fan_out) as f64).sqrt();
            for w in &mut theta[l.offset..l.offset + l.fan_in * l.fan_out] {
                *w = scale * (2.0 * rng.uniform() - 1.0);
            }
        }
        theta
    }

    fn weight(theta: &TracedTensor, l: &Layer) -> R<TracedTensor> {
        theta
            .slice(SliceConfig {
                starts: vec![l.offset],
                limits: vec![l.offset + l.fan_in * l.fan_out],
                strides: vec![1],
            })?
            .reshape([l.fan_in, l.fan_out])
    }

    fn bias(theta: &TracedTensor, offset: usize, len: usize) -> R<TracedTensor> {
        theta
            .slice(SliceConfig {
                starts: vec![offset],
                limits: vec![offset + len],
                strides: vec![1],
            })?
            .reshape([1, len])
    }

    /// (x, y, τ)（それぞれ [n, 1]）から φ の (実部, 虚部)（[n, 2]）を計算する
    fn forward(
        &self,
        theta: &TracedTensor,
        x: &TracedTensor,
        y: &TracedTensor,
        t: &TracedTensor,
        n: usize,
    ) -> R<TracedTensor> {
        // 入力を [−1, 1] に正規化する
        let normalize = |v: &TracedTensor, (lo, hi): (f64, f64)| -> R<TracedTensor> {
            let (c, r) = ((lo + hi) / 2.0, (hi - lo) / 2.0);
            v.scale_real(1.0 / r)?.add(&constant(n, -c / r)?)
        };
        let z = TracedTensor::stack(
            &[
                &normalize(x, X_RANGE)?,
                &normalize(y, Y_RANGE)?,
                &normalize(t, T_RANGE)?,
            ],
            1,
        )?
        .reshape([n, 3])?;
        let b = TracedTensor::from_vec_col_major(vec![3, FEATURES], self.fourier.clone())?;
        let proj = matmul(&z, &b)?.scale_real(2.0 * PI)?;
        let mut h = matmul(&proj.sin()?, &Self::weight(theta, &self.first_sin)?)?
            .add(&matmul(
                &proj.cos()?,
                &Self::weight(theta, &self.first_cos)?,
            )?)?
            .add(&Self::bias(theta, self.first_bias, WIDTH)?)?
            .tanh()?;
        for (l, b) in &self.hidden {
            h = matmul(&h, &Self::weight(theta, l)?)?
                .add(&Self::bias(theta, *b, WIDTH)?)?
                .tanh()?;
        }
        let (l, b) = &self.output;
        matmul(&h, &Self::weight(theta, l)?)?.add(&Self::bias(theta, *b, 2)?)
    }
}

fn matmul(a: &TracedTensor, b: &TracedTensor) -> R<TracedTensor> {
    a.dot_general(
        b,
        DotGeneralConfig {
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        },
    )
}

/// 値がすべて c の [n, 1] の定数
fn constant(n: usize, c: f64) -> R<TracedTensor> {
    TracedTensor::from_vec_col_major(vec![n, 1], vec![c; n])
}

/// Σ (a − b)² / count
fn mean_square(a: &TracedTensor, b: &TracedTensor, count: usize) -> R<TracedTensor> {
    let d = a.sub(b)?;
    d.mul(&d)?
        .reduce_sum(Some(&[0, 1]))?
        .scale_real(1.0 / count as f64)
}

fn input(shape: &[usize]) -> R<TracedTensor> {
    TracedTensor::input_concrete_shape(DType::F64, shape)
}

// ---------------------------------------------------------------- データ

/// 窓の中の格子点と、各時刻の φ・π
struct Window {
    xs: Vec<f64>,
    ys: Vec<f64>,
    taus: Vec<f64>,
    phi: Vec<Vec<Complex64>>,
    pi0: Vec<Complex64>,
}

fn load_window(rk4: &Path) -> Result<Window, Error> {
    let params = fs::read_to_string(rk4.join("params.json"))?;
    let get = |key: &str| -> Result<f64, Error> {
        let line = params
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("\"{key}\"")))
            .ok_or(format!("params.json に {key} がありません"))?;
        Ok(line
            .split(':')
            .nth(1)
            .unwrap()
            .trim()
            .trim_end_matches(',')
            .parse()?)
    };
    let (nx, dx, length, offset) = (
        get("nx")? as usize,
        get("dx")?,
        get("length")?,
        get("center_offset")?,
    );
    let grid: Vec<f64> = (0..nx)
        .map(|i| i as f64 * dx - length / 2.0 - offset)
        .collect();
    let pick = |(lo, hi): (f64, f64)| -> Vec<usize> {
        (0..nx)
            .filter(|&i| grid[i] >= lo - 1e-9 && grid[i] <= hi + 1e-9)
            .collect()
    };
    let (ix, iy) = (pick(X_RANGE), pick(Y_RANGE));
    let cut = |field: &[Complex64]| -> Vec<Complex64> {
        iy.iter()
            .flat_map(|&j| ix.iter().map(move |&i| field[j * nx + i]))
            .collect()
    };
    let mut taus = Vec::new();
    let mut phi = Vec::new();
    let mut pi0 = Vec::new();
    for line in fs::read_to_string(rk4.join("snapshots.csv"))?
        .lines()
        .skip(1)
    {
        let cols: Vec<&str> = line.split(',').collect();
        let tau: f64 = cols[1].parse()?;
        if tau < T_RANGE.0 - 1e-9 || tau > T_RANGE.1 + 1e-9 {
            continue;
        }
        let (_, _, field) = npy::read_complex_2d(&rk4.join(cols[3]))?;
        if taus.is_empty() {
            let (_, _, momentum) =
                npy::read_complex_2d(&rk4.join(cols[3].replace("snapshot", "momentum")))?;
            pi0 = cut(&momentum);
        }
        taus.push(tau);
        phi.push(cut(&field));
    }
    Ok(Window {
        xs: ix.iter().map(|&i| grid[i]).collect(),
        ys: iy.iter().map(|&j| grid[j]).collect(),
        taus,
        phi,
        pi0,
    })
}

/// 点の集合（x, y, τ と、必要なら φ の値）
#[derive(Default)]
struct Points {
    x: Vec<f64>,
    y: Vec<f64>,
    t: Vec<f64>,
    phi: Vec<Complex64>,
    pi: Vec<Complex64>,
}

impl Points {
    fn push(&mut self, x: f64, y: f64, t: f64, phi: Complex64) {
        self.x.push(x);
        self.y.push(y);
        self.t.push(t);
        self.phi.push(phi);
    }
}

fn column(v: Vec<f64>) -> Tensor {
    let n = v.len();
    Tensor::from_vec_col_major(vec![n, 1], v).expect("[n, 1] のテンソル")
}

/// 複素数の列を [n, 2]（列優先なので先に実部 n 個、次に虚部 n 個）のテンソルにする
fn complex_columns(v: &[Complex64]) -> Tensor {
    let n = v.len();
    let data = v
        .iter()
        .map(|z| z.re)
        .chain(v.iter().map(|z| z.im))
        .collect();
    Tensor::from_vec_col_major(vec![n, 2], data).expect("[n, 2] のテンソル")
}

/// 点の集合から n 個を無作為に選び、(x, y, τ, φ, π) のテンソルにする
fn sample(p: &Points, n: usize, rng: &mut Rng) -> [Tensor; 5] {
    let idx: Vec<usize> = (0..n).map(|_| rng.below(p.x.len())).collect();
    let pick = |v: &[f64]| column(idx.iter().map(|&k| v[k]).collect());
    let pick_c = |v: &[Complex64]| {
        if v.is_empty() {
            complex_columns(&vec![Complex64::new(0.0, 0.0); n])
        } else {
            complex_columns(&idx.iter().map(|&k| v[k]).collect::<Vec<_>>())
        }
    };
    [
        pick(&p.x),
        pick(&p.y),
        pick(&p.t),
        pick_c(&p.phi),
        pick_c(&p.pi),
    ]
}

// ---------------------------------------------------------------- 学習

fn run(runtime: &Runtime, program: &CompiledGraph, inputs: &[&Tensor]) -> R<Tensor> {
    Ok(runtime.run_compiled(program, inputs)?.remove(0))
}

fn main() -> Result<(), Error> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out)?;
    let w = load_window(&args.rk4)?;
    let (nx, ny) = (w.xs.len(), w.ys.len());
    println!(
        "窓: 格子 {nx}×{ny}、時刻 {} 個（τ = {} … {}）",
        w.taus.len(),
        w.taus[0],
        w.taus[w.taus.len() - 1]
    );
    let tau_i = 20.0; // シミュレーションの初期時刻（a = τ/τ_i）
    let (lambda, eta) = (1.0, 1.0);

    // 初期条件・境界条件・データ・評価の点
    let is_data_time: Vec<bool> = w
        .taus
        .iter()
        .map(|&t| (t - t.round()).abs() < 1e-9 && t > w.taus[0] + 1e-9)
        .collect();
    let mut ic = Points::default();
    let mut bc = Points::default();
    let mut data = Points::default();
    for (k, &tau) in w.taus.iter().enumerate() {
        for (j, &y) in w.ys.iter().enumerate() {
            for (i, &x) in w.xs.iter().enumerate() {
                let value = w.phi[k][j * nx + i];
                let edge = i == 0 || j == 0 || i == nx - 1 || j == ny - 1;
                if k == 0 {
                    ic.push(x, y, tau, value);
                    ic.pi.push(w.pi0[j * nx + i]);
                }
                if edge {
                    bc.push(x, y, tau, value);
                } else if is_data_time[k] {
                    data.push(x, y, tau, value);
                }
            }
        }
    }

    // ---- 計算グラフを作る
    let mut rng = Rng(args.seed);
    let net = Net::new(&mut rng);
    let mut theta = net.init(&mut rng);
    println!("パラメータ数 {}", net.n_params);
    let theta_in = input(&[net.n_params])?;

    // 物理（方程式の残差）。ℋ = 1/τ と a² = (τ/τ_i)² は点ごとの値として外から与える
    let (xr, yr, tr, hr, ar) = (
        input(&[N_RES, 1])?,
        input(&[N_RES, 1])?,
        input(&[N_RES, 1])?,
        input(&[N_RES, 1])?,
        input(&[N_RES, 1])?,
    );
    let u = net.forward(&theta_in, &xr, &yr, &tr, N_RES)?;
    let ones = constant(N_RES, 1.0)?;
    let u_t = u.jvp(&tr, &ones)?;
    let u_tt = u_t.jvp(&tr, &ones)?;
    let u_xx = u.jvp(&xr, &ones)?.jvp(&xr, &ones)?;
    let u_yy = u.jvp(&yr, &ones)?.jvp(&yr, &ones)?;
    let abs2 = u.mul(&u)?.reduce_sum(Some(&[1]))?.reshape([N_RES, 1])?;
    let potential = ar
        .mul(&abs2.add(&constant(N_RES, -eta * eta)?)?)?
        .scale_real(lambda)?;
    let residual = u_tt
        .add(&hr.mul(&u_t)?.scale_real(2.0)?)?
        .sub(&u_xx)?
        .sub(&u_yy)?
        .add(&potential.mul(&u)?)?;
    let zeros_r = TracedTensor::from_vec_col_major(vec![N_RES, 2], vec![0.0; 2 * N_RES])?;
    let loss_physics = mean_square(&residual, &zeros_r, 2 * N_RES)?;

    // 初期条件
    let (xi, yi, ti, phi_i, pi_i) = (
        input(&[N_IC, 1])?,
        input(&[N_IC, 1])?,
        input(&[N_IC, 1])?,
        input(&[N_IC, 2])?,
        input(&[N_IC, 2])?,
    );
    let u_ic = net.forward(&theta_in, &xi, &yi, &ti, N_IC)?;
    let loss_ic_phi = mean_square(&u_ic, &phi_i, 2 * N_IC)?;
    let loss_ic_pi = mean_square(&u_ic.jvp(&ti, &constant(N_IC, 1.0)?)?, &pi_i, 2 * N_IC)?;

    // 境界条件
    let (xb, yb, tb, phi_b) = (
        input(&[N_BC, 1])?,
        input(&[N_BC, 1])?,
        input(&[N_BC, 1])?,
        input(&[N_BC, 2])?,
    );
    let loss_boundary = mean_square(
        &net.forward(&theta_in, &xb, &yb, &tb, N_BC)?,
        &phi_b,
        2 * N_BC,
    )?;

    // データ
    let (xd, yd, td, phi_d) = (
        input(&[N_DATA, 1])?,
        input(&[N_DATA, 1])?,
        input(&[N_DATA, 1])?,
        input(&[N_DATA, 2])?,
    );
    let loss_data = mean_square(
        &net.forward(&theta_in, &xd, &yd, &td, N_DATA)?,
        &phi_d,
        2 * N_DATA,
    )?;

    let loss_without_data = loss_physics
        .scale_real(W_PHYSICS)?
        .add(&loss_ic_phi.add(&loss_ic_pi)?.scale_real(W_IC)?)?
        .add(&loss_boundary.scale_real(W_BOUNDARY)?)?;
    let loss_with_data = loss_without_data.add(&loss_data.scale_real(W_DATA)?)?;
    let grad_without_data = loss_without_data.grad(&theta_in)?;
    let grad_with_data = loss_with_data.grad(&theta_in)?;

    // 評価（1 時刻分の窓内全格子点）
    let n_eval = nx * ny;
    let (xe, ye, te) = (
        input(&[n_eval, 1])?,
        input(&[n_eval, 1])?,
        input(&[n_eval, 1])?,
    );
    let u_eval = net.forward(&theta_in, &xe, &ye, &te, n_eval)?;

    // ---- コンパイル（入力の順番は bindings の順）
    let spec = |t: &TracedTensor, shape: &[usize]| (t.clone(), shape.to_vec());
    let s_theta = spec(&theta_in, &[net.n_params]);
    let s_res = [
        spec(&xr, &[N_RES, 1]),
        spec(&yr, &[N_RES, 1]),
        spec(&tr, &[N_RES, 1]),
        spec(&hr, &[N_RES, 1]),
        spec(&ar, &[N_RES, 1]),
    ];
    let s_ic = [
        spec(&xi, &[N_IC, 1]),
        spec(&yi, &[N_IC, 1]),
        spec(&ti, &[N_IC, 1]),
        spec(&phi_i, &[N_IC, 2]),
        spec(&pi_i, &[N_IC, 2]),
    ];
    let s_ic_phi = [
        spec(&xi, &[N_IC, 1]),
        spec(&yi, &[N_IC, 1]),
        spec(&ti, &[N_IC, 1]),
        spec(&phi_i, &[N_IC, 2]),
    ];
    let s_ic_pi = [
        spec(&xi, &[N_IC, 1]),
        spec(&yi, &[N_IC, 1]),
        spec(&ti, &[N_IC, 1]),
        spec(&pi_i, &[N_IC, 2]),
    ];
    let s_bc = [
        spec(&xb, &[N_BC, 1]),
        spec(&yb, &[N_BC, 1]),
        spec(&tb, &[N_BC, 1]),
        spec(&phi_b, &[N_BC, 2]),
    ];
    let s_data = [
        spec(&xd, &[N_DATA, 1]),
        spec(&yd, &[N_DATA, 1]),
        spec(&td, &[N_DATA, 1]),
        spec(&phi_d, &[N_DATA, 2]),
    ];
    let s_eval = [
        spec(&xe, &[n_eval, 1]),
        spec(&ye, &[n_eval, 1]),
        spec(&te, &[n_eval, 1]),
    ];
    let mut compiler = GraphCompiler::new();
    let mut compile =
        |out: &TracedTensor, groups: &[&[(TracedTensor, Vec<usize>)]]| -> R<CompiledGraph> {
            let bindings: Vec<(&TracedTensor, DType, &[usize])> = std::iter::once(&s_theta)
                .chain(groups.iter().flat_map(|g| g.iter()))
                .map(|(t, s)| (t, DType::F64, s.as_slice()))
                .collect();
            compiler.compile_with_input_specs(out, &bindings)
        };
    let start = Instant::now();
    let p_grad_data = compile(&grad_with_data, &[&s_res, &s_ic, &s_bc, &s_data])?;
    let p_grad_nodata = compile(&grad_without_data, &[&s_res, &s_ic, &s_bc])?;
    let p_physics = compile(&loss_physics, &[&s_res])?;
    let p_ic_phi = compile(&loss_ic_phi, &[&s_ic_phi])?;
    let p_ic_pi = compile(&loss_ic_pi, &[&s_ic_pi])?;
    let p_boundary = compile(&loss_boundary, &[&s_bc])?;
    let p_data = compile(&loss_data, &[&s_data])?;
    let p_eval = compile(&u_eval, &[&s_eval])?;
    println!("コンパイル {:.1} 秒", start.elapsed().as_secs_f64());

    let backend = CpuBackend::with_threads(args.threads)?;
    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration(&backend)?)?;
    let runtime = builder.build()?;

    // 評価用の入力（時刻ごと）
    let grid_x = column((0..ny).flat_map(|_| w.xs.iter().copied()).collect());
    let grid_y = column(
        w.ys.iter()
            .flat_map(|&y| std::iter::repeat_n(y, nx))
            .collect(),
    );
    let predict = |theta: &Tensor, tau: f64| -> R<Vec<Complex64>> {
        let te = column(vec![tau; n_eval]);
        let out = run(&runtime, &p_eval, &[theta, &grid_x, &grid_y, &te])?;
        let v = out.as_slice::<f64>()?;
        Ok((0..n_eval)
            .map(|k| Complex64::new(v[k], v[n_eval + k]))
            .collect())
    };
    let rel_error = |pred: &[Complex64], exact: &[Complex64]| -> f64 {
        let num: f64 = pred
            .iter()
            .zip(exact)
            .map(|(p, e)| (p - e).norm_sqr())
            .sum();
        let den: f64 = exact.iter().map(|e| e.norm_sqr()).sum();
        (num / den).sqrt()
    };

    // ---- Adam
    let (lr0, beta1, beta2, eps): (f64, f64, f64, f64) = (1e-3, 0.9, 0.999, 1e-8);
    let mut m = vec![0.0; net.n_params];
    let mut v = vec![0.0; net.n_params];
    let mut use_data = args.mode != "physics_only";
    let mut switch_iteration: Option<usize> = None;

    let mut metrics = fs::File::create(args.out.join("metrics.csv"))?;
    writeln!(
        metrics,
        "iteration,use_data,loss_physics,loss_ic_phi,loss_ic_pi,loss_boundary,loss_data,error_data_times,error_heldout_times,error_max,seconds"
    )?;
    let start = Instant::now();
    for it in 0..=args.iterations {
        // この反復で使う点
        let mut res_x = Vec::with_capacity(N_RES);
        let mut res_y = Vec::with_capacity(N_RES);
        let mut res_t = Vec::with_capacity(N_RES);
        for _ in 0..N_RES {
            res_x.push(X_RANGE.0 + (X_RANGE.1 - X_RANGE.0) * rng.uniform());
            res_y.push(Y_RANGE.0 + (Y_RANGE.1 - Y_RANGE.0) * rng.uniform());
            res_t.push(T_RANGE.0 + (T_RANGE.1 - T_RANGE.0) * rng.uniform());
        }
        let res_h: Vec<f64> = res_t.iter().map(|t| 1.0 / t).collect();
        let res_a2: Vec<f64> = res_t.iter().map(|t| (t / tau_i).powi(2)).collect();
        let res = [
            column(res_x),
            column(res_y),
            column(res_t),
            column(res_h),
            column(res_a2),
        ];
        let [ix, iy, it_, iphi, ipi] = sample(&ic, N_IC, &mut rng);
        let [bx, by, bt, bphi, _] = sample(&bc, N_BC, &mut rng);
        let [dx_, dy, dt_, dphi, _] = sample(&data, N_DATA, &mut rng);
        let theta_t = Tensor::from_vec_col_major(vec![net.n_params], theta.clone())?;

        if it % EVAL_EVERY == 0 {
            let r: Vec<&Tensor> = res.iter().collect();
            let scalar =
                |t: Tensor| -> f64 { t.as_slice::<f64>().map(|s| s[0]).unwrap_or(f64::NAN) };
            let lp = scalar(run(&runtime, &p_physics, &[&[&theta_t][..], &r].concat())?);
            let lip = scalar(run(
                &runtime,
                &p_ic_phi,
                &[&theta_t, &ix, &iy, &it_, &iphi],
            )?);
            let lipi = scalar(run(&runtime, &p_ic_pi, &[&theta_t, &ix, &iy, &it_, &ipi])?);
            let lb = scalar(run(
                &runtime,
                &p_boundary,
                &[&theta_t, &bx, &by, &bt, &bphi],
            )?);
            let ld = scalar(run(&runtime, &p_data, &[&theta_t, &dx_, &dy, &dt_, &dphi])?);
            let mut errors = Vec::with_capacity(w.taus.len());
            for (k, &tau) in w.taus.iter().enumerate() {
                errors.push(rel_error(&predict(&theta_t, tau)?, &w.phi[k]));
            }
            let mean = |want: bool| {
                let sel: Vec<f64> = errors
                    .iter()
                    .zip(&is_data_time)
                    .filter(|(_, d)| **d == want)
                    .map(|(e, _)| *e)
                    .collect();
                sel.iter().sum::<f64>() / sel.len() as f64
            };
            let (err_data, err_held) = (mean(true), mean(false));
            let err_max = errors.iter().cloned().fold(0.0, f64::max);
            writeln!(
                metrics,
                "{it},{},{lp:.4e},{lip:.4e},{lipi:.4e},{lb:.4e},{ld:.4e},{err_data:.4e},{err_held:.4e},{err_max:.4e},{:.1}",
                use_data as u8,
                start.elapsed().as_secs_f64()
            )?;
            metrics.flush()?;
            println!(
                "{it:6} {} 残差 {lp:.2e} 誤差(データ時刻) {err_data:.3} 誤差(検証時刻) {err_held:.3} 最大 {err_max:.3} {:6.0} 秒",
                if use_data {
                    "data+物理"
                } else {
                    "物理のみ "
                },
                start.elapsed().as_secs_f64()
            );
            if args.mode == "curriculum" && use_data && err_data < args.switch {
                use_data = false;
                switch_iteration = Some(it);
                println!(
                    "--- 反復 {it} でデータを外す（データ時刻の誤差 {err_data:.3} < {}）",
                    args.switch
                );
            }
        }
        if it == args.iterations {
            break;
        }

        let grad = if use_data {
            run(
                &runtime,
                &p_grad_data,
                &[
                    &theta_t, &res[0], &res[1], &res[2], &res[3], &res[4], &ix, &iy, &it_, &iphi,
                    &ipi, &bx, &by, &bt, &bphi, &dx_, &dy, &dt_, &dphi,
                ],
            )?
        } else {
            run(
                &runtime,
                &p_grad_nodata,
                &[
                    &theta_t, &res[0], &res[1], &res[2], &res[3], &res[4], &ix, &iy, &it_, &iphi,
                    &ipi, &bx, &by, &bt, &bphi,
                ],
            )?
        };
        let g = grad.as_slice::<f64>()?;
        // 学習率は反復とともに指数的に下げ、最後に 1/10 にする（PyTorch 版と同じ）
        let lr = lr0 * 0.1_f64.powf(it as f64 / args.iterations as f64);
        let step = (it + 1) as i32;
        let (c1, c2) = (1.0 - beta1.powi(step), 1.0 - beta2.powi(step));
        for k in 0..net.n_params {
            m[k] = beta1 * m[k] + (1.0 - beta1) * g[k];
            v[k] = beta2 * v[k] + (1.0 - beta2) * g[k] * g[k];
            theta[k] -= lr * (m[k] / c1) / ((v[k] / c2).sqrt() + eps);
        }
    }

    // ---- 学習後の予測を保存する
    let theta_t = Tensor::from_vec_col_major(vec![net.n_params], theta.clone())?;
    let mut prediction = Vec::with_capacity(w.taus.len() * n_eval);
    for &tau in &w.taus {
        prediction.extend(predict(&theta_t, tau)?);
    }
    npy::write_complex(
        &args.out.join("prediction.npy"),
        &[w.taus.len(), ny, nx],
        &prediction,
    )?;
    npy::write_f64_1d(&args.out.join("taus.npy"), &w.taus)?;
    fs::write(
        args.out.join("config.json"),
        format!(
            "{{\n  \"implementation\": \"tenferro-rs\",\n  \"mode\": \"{}\",\n  \"iterations\": {},\n  \
             \"switch_threshold\": {},\n  \"switch_iteration\": {},\n  \"seed\": {},\n  \"threads\": {},\n  \
             \"n_params\": {}\n}}\n",
            args.mode,
            args.iterations,
            args.switch,
            switch_iteration.map_or("null".to_string(), |i| i.to_string()),
            args.seed,
            args.threads,
            net.n_params
        ),
    )?;
    Ok(())
}
