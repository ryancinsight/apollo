//! Three-odd-prime + radix-4 tail composite coverage and reduced-precision
//! large-N reference comparison.

use super::{check_forward, check_inverse, check_roundtrip, forward_inplace_32, max_err};
use crate::application::execution::kernel::direct::dft_forward;
use eunomia::{Complex32, Complex64};

#[test]
fn forward_n108() {
    check_forward(108, 1e-11);
} // 3³×4
#[test]
fn forward_n180() {
    check_forward(180, 1e-11);
} // 5×3²×4
#[test]
fn forward_n252() {
    check_forward(252, 1e-11);
} // 7×3²×4
#[test]
fn forward_n420() {
    check_forward(420, 1e-10);
} // 7×5×3×4
#[test]
fn roundtrip_n108() {
    check_roundtrip(108, 1e-11);
}
#[test]
fn roundtrip_n420() {
    check_roundtrip(420, 1e-10);
}
#[test]
fn inverse_n252() {
    check_inverse(252, 1e-11);
}

#[test]
fn forward_n36() {
    check_forward(36, 1e-12);
} // 3²×4
#[test]
fn forward_n60() {
    check_forward(60, 1e-12);
} // 5×3×4
#[test]
fn forward_n84() {
    check_forward(84, 1e-12);
} // 7×3×4
#[test]
fn forward_n140() {
    check_forward(140, 1e-11);
} // 7×5×4
#[test]
fn forward_n196() {
    check_forward(196, 1e-11);
} // 7²×4
#[test]
fn forward_n144() {
    check_forward(144, 1e-11);
} // 3²×16
#[test]
fn roundtrip_n60() {
    check_roundtrip(60, 1e-12);
}
#[test]
fn roundtrip_n144() {
    check_roundtrip(144, 1e-11);
}

#[test]
fn forward_reduced_n1000_matches_precise_reference() {
    let input: Vec<Complex64> = (0..1000usize)
        .map(|k| Complex64::new((k as f64 * 0.13).sin(), (k as f64 * 0.31).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut buf: Vec<Complex32> = input
        .iter()
        .map(|x| Complex32::new(x.re as f32, x.im as f32))
        .collect();
    forward_inplace_32(&mut buf);
    let got: Vec<Complex64> = buf
        .iter()
        .map(|x| Complex64::new(x.re as f64, x.im as f64))
        .collect();
    let err = max_err(&got, &expected);
    assert!(err < 2e-3, "f32 forward N=1000 max_err={err:.2e}");
}
