use super::*;
use eunomia::assert_abs_diff_eq;

fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|index| (0.17 * index as f64).sin() + 0.25 * (0.031 * index as f64).cos())
        .collect()
}

fn scalar_row(signal: &[f64], kind: DirectBasisKind, row: usize) -> f64 {
    let n = signal.len();
    match kind {
        DirectBasisKind::DctI => {
            if n < 2 {
                return 0.0;
            }
            let factor = std::f64::consts::PI / (n - 1) as f64;
            let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
            let mut sum = signal[0] + sign * signal[n - 1];
            for (index, value) in signal.iter().enumerate().take(n - 1).skip(1) {
                sum += 2.0 * *value * (factor * index as f64 * row as f64).cos();
            }
            sum
        }
        DirectBasisKind::DctII => {
            let factor = std::f64::consts::PI / n as f64;
            signal
                .iter()
                .enumerate()
                .map(|(index, value)| *value * (factor * (index as f64 + 0.5) * row as f64).cos())
                .sum()
        }
        DirectBasisKind::DctIII => {
            let factor = std::f64::consts::PI / n as f64;
            let mut sum = signal[0] * 0.5;
            for (index, value) in signal.iter().enumerate().skip(1) {
                sum += *value * (factor * index as f64 * (row as f64 + 0.5)).cos();
            }
            sum
        }
        DirectBasisKind::DctIV => {
            let factor = std::f64::consts::PI / n as f64;
            signal
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    *value * (factor * (index as f64 + 0.5) * (row as f64 + 0.5)).cos()
                })
                .sum()
        }
        DirectBasisKind::DstI => {
            let factor = std::f64::consts::PI / (n + 1) as f64;
            signal
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    2.0 * *value * (factor * (index as f64 + 1.0) * (row as f64 + 1.0)).sin()
                })
                .sum()
        }
        DirectBasisKind::DstII => {
            let factor = std::f64::consts::PI / n as f64;
            signal
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    *value * (factor * (index as f64 + 0.5) * (row as f64 + 1.0)).sin()
                })
                .sum()
        }
        DirectBasisKind::DstIII => {
            let factor = std::f64::consts::PI / n as f64;
            let sign = if row % 2 == 1 { -1.0 } else { 1.0 };
            let mut sum = sign * signal[n - 1] * 0.5;
            for (index, value) in signal.iter().enumerate().take(n - 1) {
                sum += *value * (factor * (index as f64 + 1.0) * (row as f64 + 0.5)).sin();
            }
            sum
        }
        DirectBasisKind::DstIV => {
            let factor = std::f64::consts::PI / n as f64;
            signal
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    *value * (factor * (index as f64 + 0.5) * (row as f64 + 0.5)).sin()
                })
                .sum()
        }
    }
}

fn execute(kind: DirectBasisKind, input: &[f64], output: &mut [f64]) {
    match kind {
        DirectBasisKind::DctI => dct1(input, output),
        DirectBasisKind::DctII => dct2(input, output),
        DirectBasisKind::DctIII => dct3(input, output),
        DirectBasisKind::DctIV => dct4(input, output),
        DirectBasisKind::DstI => dst1(input, output),
        DirectBasisKind::DstII => dst2(input, output),
        DirectBasisKind::DstIII => dst3(input, output),
        DirectBasisKind::DstIV => dst4(input, output),
    }
}

#[test]
fn hermes_direct_rows_match_scalar_formulas_at_threshold() {
    let n = PAR_THRESHOLD;
    let input = signal(n);
    for kind in [
        DirectBasisKind::DctI,
        DirectBasisKind::DctII,
        DirectBasisKind::DctIII,
        DirectBasisKind::DctIV,
        DirectBasisKind::DstI,
        DirectBasisKind::DstII,
        DirectBasisKind::DstIII,
        DirectBasisKind::DstIV,
    ] {
        let mut actual = vec![0.0; n];
        execute(kind, &input, &mut actual);
        for row in [0_usize, 1, 17, 64, 127, 255] {
            let expected = scalar_row(&input, kind, row);
            assert_abs_diff_eq!(actual[row], expected, epsilon = 1.0e-10);
        }
    }
}

#[test]
fn direct_basis_rows_match_scalar_coefficients() {
    let n = 8;
    for kind in [
        DirectBasisKind::DctI,
        DirectBasisKind::DctII,
        DirectBasisKind::DctIII,
        DirectBasisKind::DctIV,
        DirectBasisKind::DstI,
        DirectBasisKind::DstII,
        DirectBasisKind::DstIII,
        DirectBasisKind::DstIV,
    ] {
        let row = 3;
        let mut basis = vec![0.0; n];
        fill_direct_basis_row(&mut basis, kind, row);

        for col in 0..n {
            let mut unit = vec![0.0; n];
            unit[col] = 1.0;
            let expected = scalar_row(&unit, kind, row);
            assert_eq!(basis[col].to_bits(), expected.to_bits());
        }
    }
}
