//! NumPy の .npy 形式（バージョン 1.0、C 順序）の読み書き。

use num_complex::Complex64;
use std::fs;
use std::path::Path;

type Error = Box<dyn std::error::Error>;

/// 配列を .npy として書き出す。descr は '<c16'（complex128）や '<f8'（float64）
fn write_raw(path: &Path, descr: &str, shape: &[usize], body: &[u8]) -> std::io::Result<()> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_str = if dims.len() == 1 {
        format!("{},", dims[0])
    } else {
        dims.join(", ")
    };
    let mut header =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_str}), }}");
    // マジック(6) + バージョン(2) + ヘッダ長(2) + ヘッダ + 改行 を 64 バイト境界に揃える
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + body.len());
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(body);
    fs::write(path, bytes)
}

/// 複素数の配列（行優先、shape 任意）を complex128 で書き出す
pub fn write_complex(path: &Path, shape: &[usize], data: &[Complex64]) -> std::io::Result<()> {
    assert_eq!(data.len(), shape.iter().product::<usize>());
    let body: Vec<u8> = data
        .iter()
        .flat_map(|z| z.re.to_le_bytes().into_iter().chain(z.im.to_le_bytes()))
        .collect();
    write_raw(path, "<c16", shape, &body)
}

/// 実数の 1 次元配列を float64 で書き出す
pub fn write_f64_1d(path: &Path, data: &[f64]) -> std::io::Result<()> {
    let body: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_raw(path, "<f8", &[data.len()], &body)
}

/// complex128・C 順序の .npy を読む。戻り値は (shape, data)
pub fn read_complex(path: &Path) -> Result<(Vec<usize>, Vec<Complex64>), Error> {
    let bytes = fs::read(path)?;
    if bytes.len() < 10 || &bytes[..8] != b"\x93NUMPY\x01\x00" {
        return Err(format!(
            "npy バージョン 1.0 のファイルではありません: {}",
            path.display()
        )
        .into());
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
    let body = &bytes[10 + header_len..];
    if body.len() != dims.iter().product::<usize>() * 16 {
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
    Ok((dims, data))
}

/// complex128・C 順序の 2 次元 .npy を読む。戻り値は (rows, cols, data)
pub fn read_complex_2d(path: &Path) -> Result<(usize, usize, Vec<Complex64>), Error> {
    let (dims, data) = read_complex(path)?;
    let [rows, cols] = dims[..] else {
        return Err("2 次元配列ではありません".into());
    };
    Ok((rows, cols, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_complex() {
        let path = std::env::temp_dir().join("cosmicstrings_npy_roundtrip.npy");
        let data: Vec<Complex64> = (0..6)
            .map(|k| Complex64::new(k as f64, -0.5 * k as f64))
            .collect();
        write_complex(&path, &[2, 3], &data).unwrap();
        let (rows, cols, read) = read_complex_2d(&path).unwrap();
        assert_eq!((rows, cols), (2, 3));
        assert_eq!(read, data);
        fs::remove_file(path).unwrap();
    }
}
