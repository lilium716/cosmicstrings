//! 大域的ひもを、輻射優勢期の FLRW 時空の 3 次元空間で解く。
//!
//! 方程式（共形時間 τ、共動座標、′ = ∂/∂τ、ℋ = a′/a）:
//!   φ′ = π
//!   π′ = −2ℋπ + ∇²φ − a² λ(|φ|² − η²)φ
//! 背景: a(τ) = τ/τ_i（ℋ = 1/τ）。expansion=none なら膨張なし（a = 1、ℋ = 0）
//! 差分: 7 点差分ラプラシアン、時間は古典的 RK4 または 2N-storage の RK4（integrator=lsrk4）。境界は x, y 方向がディリクレ条件（境界の点を初期値に固定）、
//!       z 方向はディリクレ条件または周期境界（periodic_z=true）。計算は rayon で並列化する
//!
//! 配置:
//!   parallel : z 軸に平行な渦（巻き数 +1、x = −d/2）と反渦（−1、x = +d/2）。静止。
//!              2 次元の simulate.rs と同じ初期条件で、z によらない（2 次元との比較用）
//!   loop     : xy 平面（z = 0）に半径 R₀（radius）の円形のひもを静止させて置く。z 軸を含む断面では
//!              ρ = +R₀ の渦と ρ = −R₀ の反渦の組に見えるので、2 次元と同じ形の式を ρ = √(x² + y²) で書く:
//!              θ = atan2(z, ρ − R₀) − atan2(z, ρ + R₀)、振幅は f(d₁) f(d₂)（d は (ρ, z) = (±R₀, 0) からの距離）
//!   crossing : ひも A（z 方向、(x, y) = (−d/2, 0) を通る、巻き数 +1）を +x 方向へ、
//!              ひも B（y 方向、(x, z) = (+d/2, 0) を通る、巻き数 +1）を −x 方向へ、速さ v で動かす。
//!              場は 2 本の積 φ = φ_A φ_B。動くひもはローレンツ収縮させたプロファイルにする
//!
//! 使い方: cargo run --release --bin simulate3d -- <出力ディレクトリ> [key=value ...]
//!   config=crossing  nx=256（x, y 方向の点数）  nz=256  length=32（x, y 方向の箱の一辺、Δx = length/nx）
//!   periodic_z=false  tau_end=36  output_interval=2  separation=8  velocity=0.5  courant=0.2
//!   save_full=false（true なら 3 次元の場全体を snapshot_XXXX.npy、shape (nz, ny, nx) として保存する）
//!   expansion=radiation（none なら膨張なし）  radius=8（loop の半径）
//!   integrator=rk4（rk4: 古典的 4 段 RK4、作業領域を含め 1 点 128 バイト。
//!                   lsrk4: Carpenter–Kennedy の 5 段 4 次 2N-storage RK、1 点 64 バイト）
//! 出力: params.json、timeseries.csv（エネルギー、|φ| の最小値、ひもの長さ）、snapshots.csv、
//!       cores_XXXX.csv（巻き数をもつプラケットの中心 = ひもの位置）、
//!       slice_z_XXXX.npy（z = 0 に最も近い断面の φ）、slice_y_XXXX.npy（y = 0 に最も近い断面の φ）

use std::f64::consts::TAU;
use std::fs;
use std::io::Write;
use std::path::Path;

use num_complex::Complex64;
use rayon::prelude::*;

use cosmicstrings::io::npy;

type Error = Box<dyn std::error::Error>;

/// 場の 4 成分 [Re φ, Im φ, Re π, Im π]。それぞれ index p = (k * ny + j) * nx + i
#[derive(Clone)]
struct State([Vec<f64>; 4]);

impl State {
    fn zeros(n: usize) -> Self {
        State([vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]])
    }
}

struct Setup {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    eta: f64,
    lambda: f64,
    tau_initial: f64,
    periodic_z: bool,
    /// false なら膨張なし（a = 1、ℋ = 0）
    expanding: bool,
}

impl Setup {
    fn scale_factor(&self, tau: f64) -> f64 {
        if self.expanding {
            tau / self.tau_initial
        } else {
            1.0
        }
    }
    fn hubble(&self, tau: f64) -> f64 {
        if self.expanding { 1.0 / tau } else { 0.0 }
    }
    fn plane(&self) -> usize {
        self.nx * self.ny
    }
}

fn main() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = args
        .first()
        .cloned()
        .ok_or("出力ディレクトリを指定してください")?;
    let out = Path::new(&out_dir);
    fs::create_dir_all(out)?;

    let mut config = "crossing".to_string();
    let (mut nx, mut nz, mut length) = (256usize, 256usize, 32.0f64);
    let mut periodic_z = false;
    let (mut tau_end, mut output_interval, mut courant) = (36.0f64, 2.0f64, 0.2f64);
    let (mut separation, mut velocity) = (8.0f64, 0.5f64);
    let mut expansion = "radiation".to_string();
    let mut radius = 8.0f64;
    let mut save_full = false;
    let mut integrator = "rk4".to_string();
    for arg in args.iter().skip(1) {
        match arg.split_once('=') {
            Some(("config", v)) if ["crossing", "parallel", "loop"].contains(&v) => {
                config = v.to_string()
            }
            Some(("expansion", v)) if v == "radiation" || v == "none" => expansion = v.to_string(),
            Some(("radius", v)) => radius = v.parse()?,
            Some(("nx", v)) => nx = v.parse()?,
            Some(("nz", v)) => nz = v.parse()?,
            Some(("length", v)) => length = v.parse()?,
            Some(("periodic_z", v)) => periodic_z = v.parse()?,
            Some(("tau_end", v)) => tau_end = v.parse()?,
            Some(("output_interval", v)) => output_interval = v.parse()?,
            Some(("courant", v)) => courant = v.parse()?,
            Some(("separation", v)) => separation = v.parse()?,
            Some(("velocity", v)) => velocity = v.parse()?,
            Some(("save_full", v)) => save_full = v.parse()?,
            Some(("integrator", v)) if v == "rk4" || v == "lsrk4" => integrator = v.to_string(),
            _ => return Err(format!("未知の引数です: {arg}").into()),
        }
    }

    let ny = nx;
    let dx = length / nx as f64;
    let dt = courant * dx;
    let tau_initial = 20.0;
    let steps = ((tau_end - tau_initial) / dt).round() as usize;
    let output_every = (output_interval / dt).round() as usize;
    let (eta, lambda): (f64, f64) = (1.0, 1.0);
    let core_width = 2.0 / ((2.0 * lambda).sqrt() * eta); // tanh(m r / 2)、m = √(2λ) η
    let center_offset = 1.0 / 24.0; // ひもの中心が格子点に乗らないようにずらす
    let s = Setup {
        nx,
        ny,
        nz,
        dx,
        eta,
        lambda,
        tau_initial,
        periodic_z,
        expanding: expansion == "radiation",
    };
    let n = nx * ny * nz;
    // 座標: x = iΔx − L/2 − δ（y, z も同様、z 方向の長さは nz Δx）
    let coord = |i: usize, len: f64| i as f64 * dx - len / 2.0 - center_offset;
    let lz = nz as f64 * dx;

    // ---- 初期条件
    let pair_field = |x: f64, y: f64, z: f64, t: f64| -> Complex64 {
        match config.as_str() {
            "parallel" => {
                // simulate.rs と同じ式
                let xp = x + separation / 2.0;
                let xm = x - separation / 2.0;
                let phase = y.atan2(xp) - y.atan2(xm);
                let amplitude =
                    eta * (xp.hypot(y) / core_width).tanh() * (xm.hypot(y) / core_width).tanh();
                Complex64::from_polar(amplitude, phase)
            }
            "loop" => {
                let rho = x.hypot(y);
                let phase = z.atan2(rho - radius) - z.atan2(rho + radius);
                let amplitude = eta
                    * ((rho - radius).hypot(z) / core_width).tanh()
                    * ((rho + radius).hypot(z) / core_width).tanh();
                Complex64::from_polar(amplitude, phase)
            }
            _ => {
                let gamma = 1.0 / (1.0 - velocity * velocity).sqrt();
                // ひも A: z 方向、x = −d/2 + v t、y = 0
                let xa = gamma * (x + separation / 2.0 - velocity * t);
                // ひも B: y 方向、x = +d/2 − v t、z = 0
                let xb = gamma * (x - separation / 2.0 + velocity * t);
                let phase = y.atan2(xa) + z.atan2(xb);
                let amplitude =
                    eta * (xa.hypot(y) / core_width).tanh() * (xb.hypot(z) / core_width).tanh();
                Complex64::from_polar(amplitude, phase)
            }
        }
    };
    let mut u = State::zeros(n);
    {
        let h = 1e-5; // π = ∂φ/∂τ を時間の中心差分で求める
        let [re, im, pre, pim] = &mut u.0;
        re.par_iter_mut()
            .zip(im.par_iter_mut())
            .zip(pre.par_iter_mut().zip(pim.par_iter_mut()))
            .enumerate()
            .for_each(|(p, ((r, i), (pr, pi)))| {
                let (ii, jj, kk) = (p % nx, (p / nx) % ny, p / (nx * ny));
                let (x, y, z) = (coord(ii, length), coord(jj, length), coord(kk, lz));
                let phi = pair_field(x, y, z, 0.0);
                let dphi = (pair_field(x, y, z, h) - pair_field(x, y, z, -h)) / (2.0 * h);
                *r = phi.re;
                *i = phi.im;
                *pr = dphi.re;
                *pi = dphi.im;
            });
    }

    fs::write(
        out.join("params.json"),
        format!(
            "{{\n  \"config\": \"{config}\",\n  \"nx\": {nx},\n  \"ny\": {ny},\n  \"nz\": {nz},\n  \"length\": {length},\n  \
             \"dx\": {dx},\n  \"dt\": {dt},\n  \"tau_initial\": {tau_initial},\n  \"tau_end\": {tau_end},\n  \"eta\": {eta},\n  \
             \"lambda\": {lambda},\n  \"core_width\": {core_width},\n  \"separation\": {separation},\n  \"velocity\": {velocity},\n  \
             \"center_offset\": {center_offset},\n  \"periodic_z\": {periodic_z},\n  \"courant\": {courant},\n  \
             \"output_interval\": {output_interval},\n  \"integrator\": \"{integrator}\",\n  \
             \"expansion\": \"{expansion}\",\n  \"radius\": {radius}\n}}\n"
        ),
    )?;
    let mut timeseries = fs::File::create(out.join("timeseries.csv"))?;
    writeln!(
        timeseries,
        "tau,a,energy,min_abs_phi,string_length,n_plaquettes"
    )?;
    let mut snapshots = fs::File::create(out.join("snapshots.csv"))?;
    writeln!(snapshots, "index,tau,a")?;

    // 時間積分の作業領域（古典的 RK4 は 3 組、2N-storage の RK は 1 組）
    let low_storage = integrator == "lsrk4";
    let (mut k, mut stage, mut sum) = if low_storage {
        (State::zeros(0), State::zeros(0), State::zeros(0))
    } else {
        (State::zeros(n), State::zeros(n), State::zeros(n))
    };
    let mut du = if low_storage {
        State::zeros(n)
    } else {
        State::zeros(0)
    };
    let kz0 = ((lz / 2.0 + center_offset) / dx).round() as usize; // z = 0 に最も近い断面
    let jy0 = ((length / 2.0 + center_offset) / dx).round() as usize; // y = 0 に最も近い断面

    for step in 0..=steps {
        let tau = tau_initial + step as f64 * dt;
        let a = s.scale_factor(tau);
        if step % output_every == 0 {
            let index = step / output_every;
            let min_abs = u.0[0]
                .par_iter()
                .zip(u.0[1].par_iter())
                .map(|(r, i)| r.hypot(*i))
                .reduce(|| f64::INFINITY, f64::min);
            let energy = energy(&s, tau, &u);
            let cores = find_cores(&s, &u);
            let string_length = cores.len() as f64 * dx;
            writeln!(
                timeseries,
                "{tau},{a},{energy},{min_abs},{string_length},{}",
                cores.len()
            )?;
            writeln!(snapshots, "{index},{tau},{a}")?;
            let mut f = fs::File::create(out.join(format!("cores_{index:04}.csv")))?;
            writeln!(f, "x,y,z,normal,winding")?;
            for (i, j, kk, axis, w) in &cores {
                // プラケットの中心の座標（法線の向きには格子点の位置、それ以外は半格子ずらす）
                let cx = coord(*i, length) + if *axis == 0 { 0.0 } else { dx / 2.0 };
                let cy = coord(*j, length) + if *axis == 1 { 0.0 } else { dx / 2.0 };
                let cz = coord(*kk, lz) + if *axis == 2 { 0.0 } else { dx / 2.0 };
                writeln!(f, "{cx},{cy},{cz},{},{w}", ["x", "y", "z"][*axis])?;
            }
            if save_full {
                let full: Vec<Complex64> = u.0[0]
                    .iter()
                    .zip(&u.0[1])
                    .map(|(r, i)| Complex64::new(*r, *i))
                    .collect();
                npy::write_complex(
                    &out.join(format!("snapshot_{index:04}.npy")),
                    &[nz, ny, nx],
                    &full,
                )?;
            }
            // 断面
            let plane = s.plane();
            let slice_z: Vec<Complex64> = (0..plane)
                .map(|q| Complex64::new(u.0[0][kz0 * plane + q], u.0[1][kz0 * plane + q]))
                .collect();
            npy::write_complex(
                &out.join(format!("slice_z_{index:04}.npy")),
                &[ny, nx],
                &slice_z,
            )?;
            let slice_y: Vec<Complex64> = (0..nz)
                .flat_map(|kk| (0..nx).map(move |i| (kk * ny + jy0) * nx + i))
                .map(|p| Complex64::new(u.0[0][p], u.0[1][p]))
                .collect();
            npy::write_complex(
                &out.join(format!("slice_y_{index:04}.npy")),
                &[nz, nx],
                &slice_y,
            )?;
            println!(
                "τ = {tau:6.2}, a = {a:.3}, E = {energy:.6}, min|φ| = {min_abs:.3}, ひもの長さ {string_length:.1}"
            );
            if !min_abs.is_finite() {
                return Err("場が発散しました".into());
            }
        }
        if step == steps {
            break;
        }

        if low_storage {
            // Carpenter–Kennedy の 5 段 4 次 2N-storage RK:
            //   du ← A_i du + Δτ f(τ + C_i Δτ, u)、u ← u + B_i du
            for st in 0..5 {
                rhs(&s, tau + LSRK_C[st] * dt, &u, &mut du, LSRK_A[st], dt);
                for c in 0..4 {
                    let dc = &du.0[c];
                    u.0[c]
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(q, v)| *v += LSRK_B[st] * dc[q]);
                }
            }
            continue;
        }
        // 古典的 4 段ルンゲ・クッタ法
        const NODES: [f64; 4] = [0.0, 0.5, 0.5, 1.0];
        const WEIGHTS: [f64; 4] = [1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0];
        stage.clone_from(&u);
        for c in &mut sum.0 {
            c.par_iter_mut().for_each(|v| *v = 0.0);
        }
        for st in 0..4 {
            rhs(&s, tau + NODES[st] * dt, &stage, &mut k, 0.0, 1.0);
            let next = if st < 3 { NODES[st + 1] * dt } else { 0.0 };
            for c in 0..4 {
                let (kc, uc) = (&k.0[c], &u.0[c]);
                sum.0[c]
                    .par_iter_mut()
                    .zip(stage.0[c].par_iter_mut())
                    .enumerate()
                    .for_each(|(q, (sm, sg))| {
                        *sm += WEIGHTS[st] * kc[q];
                        *sg = uc[q] + next * kc[q];
                    });
            }
        }
        for c in 0..4 {
            let sc = &sum.0[c];
            u.0[c]
                .par_iter_mut()
                .enumerate()
                .for_each(|(q, v)| *v += dt * sc[q]);
        }
    }
    Ok(())
}

/// Carpenter & Kennedy (1994) の 5 段 4 次 2N-storage RK の係数
const LSRK_A: [f64; 5] = [
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
];
const LSRK_B: [f64; 5] = [
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
];
const LSRK_C: [f64; 5] = [
    0.0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926251.0,
];

/// 方程式の右辺 f を計算し、du ← keep · du + scale · f とする（keep = 0、scale = 1 なら du = f）。
/// ディリクレ条件の境界の点は固定（du = 0）
fn rhs(s: &Setup, tau: f64, u: &State, du: &mut State, keep: f64, scale: f64) {
    let (nx, ny, nz) = (s.nx, s.ny, s.nz);
    let plane = s.plane();
    let a = s.scale_factor(tau);
    let friction = 2.0 * s.hubble(tau);
    let coupling = a * a * s.lambda;
    let inv_dx2 = 1.0 / (s.dx * s.dx);
    let [re, im, pre, pim] = &u.0;
    let [dre, dim, dpre, dpim] = &mut du.0;
    dre.par_chunks_mut(plane)
        .zip(dim.par_chunks_mut(plane))
        .zip(dpre.par_chunks_mut(plane).zip(dpim.par_chunks_mut(plane)))
        .enumerate()
        .for_each(|(kk, ((d0, d1), (d2, d3)))| {
            let on_z_boundary = !s.periodic_z && (kk == 0 || kk == nz - 1);
            let up = (if kk + 1 == nz { 0 } else { kk + 1 }) * plane;
            let down = (if kk == 0 { nz - 1 } else { kk - 1 }) * plane;
            for j in 0..ny {
                for i in 0..nx {
                    let q = j * nx + i;
                    if on_z_boundary || i == 0 || j == 0 || i == nx - 1 || j == ny - 1 {
                        d0[q] = 0.0;
                        d1[q] = 0.0;
                        d2[q] = 0.0;
                        d3[q] = 0.0;
                        continue;
                    }
                    let p = kk * plane + q;
                    let lap = |f: &[f64]| {
                        (f[p + 1] + f[p - 1] + f[p + nx] + f[p - nx] + f[up + q] + f[down + q]
                            - 6.0 * f[p])
                            * inv_dx2
                    };
                    let (phi_re, phi_im) = (re[p], im[p]);
                    let potential = coupling * (phi_re * phi_re + phi_im * phi_im - s.eta * s.eta);
                    let f = [
                        pre[p],
                        pim[p],
                        -friction * pre[p] + lap(re) - potential * phi_re,
                        -friction * pim[p] + lap(im) - potential * phi_im,
                    ];
                    if keep == 0.0 && scale == 1.0 {
                        (d0[q], d1[q], d2[q], d3[q]) = (f[0], f[1], f[2], f[3]);
                    } else {
                        d0[q] = keep * d0[q] + scale * f[0];
                        d1[q] = keep * d1[q] + scale * f[1];
                        d2[q] = keep * d2[q] + scale * f[2];
                        d3[q] = keep * d3[q] + scale * f[3];
                    }
                }
            }
        });
}

/// 共動体積でのエネルギー（膨張なしなら保存量。simulate.rs の energy の 3 次元版）
fn energy(s: &Setup, tau: f64, u: &State) -> f64 {
    let (nx, ny, nz) = (s.nx, s.ny, s.nz);
    let plane = s.plane();
    let a = s.scale_factor(tau);
    let [re, im, pre, pim] = &u.0;
    let link = |p: usize, q: usize| (re[q] - re[p]).powi(2) + (im[q] - im[p]).powi(2);
    let total: f64 = (0..nz)
        .into_par_iter()
        .map(|kk| {
            let mut sum = 0.0;
            for j in 0..ny {
                for i in 0..nx {
                    let p = kk * plane + j * nx + i;
                    let v = re[p] * re[p] + im[p] * im[p] - s.eta * s.eta;
                    sum += pre[p].powi(2) + pim[p].powi(2) + a * a * 0.5 * s.lambda * v * v;
                    let mut grad = 0.0;
                    if i + 1 < nx {
                        grad += link(p, p + 1);
                    }
                    if j + 1 < ny {
                        grad += link(p, p + nx);
                    }
                    if kk + 1 < nz {
                        grad += link(p, p + plane);
                    } else if s.periodic_z {
                        grad += link(p, j * nx + i);
                    }
                    sum += grad / (s.dx * s.dx);
                }
            }
            sum
        })
        .sum();
    total * s.dx.powi(3)
}

/// 巻き数をもつプラケットを探す。戻り値は (i, j, k, 法線の向き 0/1/2 = x/y/z, 巻き数)。
/// プラケットは格子点 (i, j, k) を角にもち、法線の向き以外の 2 方向に 1 格子ずつ広がる
fn find_cores(s: &Setup, u: &State) -> Vec<(usize, usize, usize, usize, i32)> {
    let (nx, ny, nz) = (s.nx, s.ny, s.nz);
    let plane = s.plane();
    let [re, im, _, _] = &u.0;
    let phase = |p: usize| im[p].atan2(re[p]);
    let wrap = |d: f64| d - TAU * (d / TAU).round();
    let kmax = if s.periodic_z { nz } else { nz - 1 };
    (0..nz)
        .into_par_iter()
        .flat_map_iter(|kk| {
            let mut found = Vec::new();
            let kp = if kk + 1 == nz { 0 } else { kk + 1 };
            for j in 0..ny {
                for i in 0..nx {
                    let p = kk * plane + j * nx + i;
                    // 法線 z: (i, j) → (i+1, j) → (i+1, j+1) → (i, j+1)
                    let candidates = [
                        (
                            2usize,
                            i + 1 < nx && j + 1 < ny,
                            [p, p + 1, p + 1 + nx, p + nx],
                        ),
                        // 法線 x: (j, k) → (j+1, k) → (j+1, k+1) → (j, k+1)
                        (
                            0,
                            j + 1 < ny && kk < kmax,
                            [
                                p,
                                p + nx,
                                kp * plane + (j + 1) * nx + i,
                                kp * plane + j * nx + i,
                            ],
                        ),
                        // 法線 y: (k, i) → (k+1, i) → (k+1, i+1) → (k, i+1)
                        (
                            1,
                            i + 1 < nx && kk < kmax,
                            [
                                p,
                                kp * plane + j * nx + i,
                                kp * plane + j * nx + i + 1,
                                p + 1,
                            ],
                        ),
                    ];
                    for (axis, valid, c) in candidates {
                        if !valid {
                            continue;
                        }
                        let total: f64 = (0..4)
                            .map(|m| wrap(phase(c[(m + 1) % 4]) - phase(c[m])))
                            .sum();
                        let w = (total / TAU).round() as i32;
                        if w != 0 {
                            found.push((i, j, kk, axis, w));
                        }
                    }
                }
            }
            found
        })
        .collect()
}
