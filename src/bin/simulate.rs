//! 大域的ひも（渦と反渦）の対消滅を、輻射優勢期の FLRW 時空の 2 次元断面で解く。
//!
//! 方程式（共形時間 τ、共動座標、′ = ∂/∂τ、ℋ = a′/a）:
//!   φ′ = π
//!   π′ = −2ℋπ + ∇²φ − a² λ(|φ|² − η²)φ
//! 背景: a(τ) = τ/τ_i（ℋ = 1/τ）
//! 差分: 5 点差分ラプラシアン、RK4、境界はディリクレ条件（境界の点を初期値に固定）
//!
//! 使い方: cargo run --release --bin simulate -- <出力ディレクトリ>
//! 出力: params.json、timeseries.csv（|φ| の最小値とひもの本数）、
//!       snapshots.csv、snapshot_XXXX.npy（φ、complex128、shape (nx, nx)、配列[j, i] = φ(x_i, y_j)）

use std::f64::consts::TAU;
use std::fs;
use std::io::Write;
use std::path::Path;

/// 方程式の右辺を計算するのに必要な設定
struct Setup {
    nx: usize,
    dx: f64,
    eta: f64,
    lambda: f64,
    tau_initial: f64,
}

impl Setup {
    /// スケール因子 a(τ) = τ/τ_i
    fn scale_factor(&self, tau: f64) -> f64 {
        tau / self.tau_initial
    }

    /// 共形ハッブルパラメータ ℋ = 1/τ
    fn hubble(&self, tau: f64) -> f64 {
        1.0 / tau
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data".to_string());
    let out = Path::new(&out_dir);
    fs::create_dir_all(out)?;

    // 格子: x = iΔx − L/2（i = 0, …, nx − 1）の nx 点。QTT 化のため 2 のべきにする。
    // 両端の点（i = 0 と nx − 1）は境界値として固定する
    let nx: usize = 512;
    let length: f64 = 64.0;
    let dx = length / nx as f64;
    let dt = 0.2 * dx;
    let tau_initial: f64 = 20.0;
    let tau_end: f64 = 60.0;
    let steps = ((tau_end - tau_initial) / dt).round() as usize;
    let output_every: usize = 80; // Δτ = 2 ごとに出力

    let eta: f64 = 1.0;
    let lambda: f64 = 1.0;
    let core_width: f64 = 2.0 / ((2.0 * lambda).sqrt() * eta); // tanh(m r / 2)、m = √(2λ) η
    let separation: f64 = 16.0;
    // ひもの中心が格子点に乗らないようにずらす（1/24 は dx = 1/8 の格子点に乗らない）
    let center_offset: f64 = 1.0 / 24.0;
    let n = nx * nx;

    let setup = Setup {
        nx,
        dx,
        eta,
        lambda,
        tau_initial,
    };

    // u = [Re φ, Im φ, Re π, Im π]（それぞれ n 個、index p = j * nx + i）
    let mut u = vec![0.0; 4 * n];
    for j in 0..nx {
        for i in 0..nx {
            let p = j * nx + i;
            let x = i as f64 * dx - length / 2.0 - center_offset;
            let y = j as f64 * dx - length / 2.0 - center_offset;
            // 渦（巻き数 +1）は x = −d/2、反渦（−1）は x = +d/2
            let xp = x + separation / 2.0;
            let xm = x - separation / 2.0;
            let r_plus = xp.hypot(y);
            let r_minus = xm.hypot(y);
            let phase = y.atan2(xp) - y.atan2(xm);
            let amplitude = eta * (r_plus / core_width).tanh() * (r_minus / core_width).tanh();
            u[p] = amplitude * phase.cos();
            u[n + p] = amplitude * phase.sin();
            // π = 0（静止した状態から始める）
        }
    }

    fs::write(
        out.join("params.json"),
        format!(
            "{{\n  \"nx\": {nx},\n  \"length\": {length},\n  \"dx\": {dx},\n  \"dt\": {dt},\n  \
             \"tau_initial\": {tau_initial},\n  \"tau_end\": {tau_end},\n  \"eta\": {eta},\n  \
             \"lambda\": {lambda},\n  \"core_width\": {core_width},\n  \"separation\": {separation},\n  \
             \"center_offset\": {center_offset},\n  \"boundary\": \"dirichlet\",\n  \"integrator\": \"rk4\"\n}}\n"
        ),
    )?;
    let mut timeseries = fs::File::create(out.join("timeseries.csv"))?;
    writeln!(timeseries, "tau,a,min_abs_phi,n_plus,n_minus")?;
    let mut snapshots = fs::File::create(out.join("snapshots.csv"))?;
    writeln!(snapshots, "index,tau,a,file")?;

    // RK4 の作業領域
    let mut k = vec![0.0; 4 * n];
    let mut stage = vec![0.0; 4 * n];
    let mut sum = vec![0.0; 4 * n];

    for step in 0..=steps {
        let tau = tau_initial + step as f64 * dt;
        let a = setup.scale_factor(tau);

        if step % output_every == 0 {
            let index = step / output_every;
            let min_abs = (0..n)
                .map(|p| u[p].hypot(u[n + p]))
                .fold(f64::INFINITY, f64::min);
            let (n_plus, n_minus) = count_vortices(&u, nx);
            writeln!(timeseries, "{tau},{a},{min_abs},{n_plus},{n_minus}")?;
            let name = format!("snapshot_{index:04}.npy");
            write_npy(&out.join(&name), &u, nx)?;
            writeln!(snapshots, "{index},{tau},{a},{name}")?;
            println!(
                "τ = {tau:6.2}, a = {a:.3}, min|φ| = {min_abs:.3}, ひも: +{n_plus} / −{n_minus}"
            );
            if !min_abs.is_finite() {
                return Err("場が発散しました".into());
            }
        }
        if step == steps {
            break;
        }

        // 古典的 4 段ルンゲ・クッタ法
        const NODES: [f64; 4] = [0.0, 0.5, 0.5, 1.0];
        const WEIGHTS: [f64; 4] = [1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0];
        stage.copy_from_slice(&u);
        sum.fill(0.0);
        for s in 0..4 {
            rhs(&setup, tau + NODES[s] * dt, &stage, &mut k);
            let next = if s < 3 { NODES[s + 1] * dt } else { 0.0 };
            for q in 0..4 * n {
                sum[q] += WEIGHTS[s] * k[q];
                stage[q] = u[q] + next * k[q];
            }
        }
        for q in 0..4 * n {
            u[q] += dt * sum[q];
        }
    }
    Ok(())
}

/// 方程式の右辺 du/dτ を計算する。境界の点は固定（du = 0）
fn rhs(s: &Setup, tau: f64, u: &[f64], du: &mut [f64]) {
    let nx = s.nx;
    let n = nx * nx;
    let a = s.scale_factor(tau);
    let friction = 2.0 * s.hubble(tau);
    let coupling = a * a * s.lambda;
    let inv_dx2 = 1.0 / (s.dx * s.dx);
    let (re, im, pre, pim) = (0, n, 2 * n, 3 * n);

    du.fill(0.0);
    for j in 1..nx - 1 {
        for i in 1..nx - 1 {
            let p = j * nx + i;
            let lap = |c: usize| {
                (u[c + p + 1] + u[c + p - 1] + u[c + p + nx] + u[c + p - nx] - 4.0 * u[c + p])
                    * inv_dx2
            };
            let (phi_re, phi_im) = (u[re + p], u[im + p]);
            let potential = coupling * (phi_re * phi_re + phi_im * phi_im - s.eta * s.eta);
            du[re + p] = u[pre + p];
            du[im + p] = u[pim + p];
            du[pre + p] = -friction * u[pre + p] + lap(re) - potential * phi_re;
            du[pim + p] = -friction * u[pim + p] + lap(im) - potential * phi_im;
        }
    }
}

/// 各プラケットを一周したときの位相差の和から巻き数を求め、+1 と −1 の個数を返す
fn count_vortices(u: &[f64], nx: usize) -> (usize, usize) {
    let n = nx * nx;
    let phase = |p: usize| u[n + p].atan2(u[p]);
    let wrap = |d: f64| d - TAU * (d / TAU).round();
    let (mut plus, mut minus) = (0, 0);
    for j in 0..nx - 1 {
        for i in 0..nx - 1 {
            let corners = [
                j * nx + i,
                j * nx + i + 1,
                (j + 1) * nx + i + 1,
                (j + 1) * nx + i,
            ];
            let total: f64 = (0..4)
                .map(|c| wrap(phase(corners[(c + 1) % 4]) - phase(corners[c])))
                .sum();
            match (total / TAU).round() as i32 {
                0 => {}
                w if w > 0 => plus += w as usize,
                w => minus += (-w) as usize,
            }
        }
    }
    (plus, minus)
}

/// φ を NumPy の .npy 形式（complex128、shape (nx, nx)）で書き出す
fn write_npy(path: &Path, u: &[f64], nx: usize) -> std::io::Result<()> {
    let n = nx * nx;
    let mut header =
        format!("{{'descr': '<c16', 'fortran_order': False, 'shape': ({nx}, {nx}), }}");
    // マジック(6) + バージョン(2) + ヘッダ長(2) + ヘッダ + 改行 を 64 バイト境界に揃える
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + 16 * n);
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for p in 0..n {
        bytes.extend_from_slice(&u[p].to_le_bytes());
        bytes.extend_from_slice(&u[n + p].to_le_bytes());
    }
    fs::write(path, bytes)
}
