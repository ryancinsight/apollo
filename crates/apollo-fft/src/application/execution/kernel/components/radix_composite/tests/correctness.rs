//! Forward, roundtrip, and inverse value-semantic correctness across canonical
//! prime-23-smooth sizes, including DC regressions and reduced-precision
//! reference comparisons.

use super::{
    check_forward, check_inverse, check_roundtrip, forward_inplace_32, forward_inplace_64, max_err,
};
use crate::application::execution::kernel::components::radix_composite::forward_inplace_with_radices;
use crate::application::execution::kernel::direct::dft_forward;
use eunomia::{Complex32, Complex64};

#[test]
fn forward_n7() {
    check_forward(7, 1e-13);
}
#[test]
fn forward_n3() {
    check_forward(3, 1e-13);
}
#[test]
fn forward_n5() {
    check_forward(5, 1e-13);
}
#[test]
fn forward_n9() {
    check_forward(9, 1e-12);
}
#[test]
fn forward_n15() {
    check_forward(15, 1e-12);
}
#[test]
fn forward_n25() {
    check_forward(25, 1e-12);
}
#[test]
fn forward_n6() {
    check_forward(6, 1e-13);
}
#[test]
fn forward_n10() {
    check_forward(10, 1e-12);
}
#[test]
fn forward_n14() {
    check_forward(14, 1e-12);
}
#[test]
fn forward_n21() {
    check_forward(21, 1e-11);
}

#[test]
fn forward_n100() {
    check_forward(100, 1e-11);
}
#[test]
fn forward_n1000() {
    check_forward(1000, 1e-9);
}
#[test]
fn forward_n10000() {
    check_forward(10000, 1e-8);
}

#[test]
fn forward_n12() {
    check_forward(12, 1e-13);
}

#[test]
fn twiddle_cache_distinguishes_radix_order_for_same_length() {
    let input: Vec<Complex64> = (0..12)
        .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
        .collect();
    let expected = dft_forward(&input);

    let mut radix_3_4 = input.clone();
    forward_inplace_with_radices(&mut radix_3_4, &[3, 4]);
    assert!(
        max_err(&radix_3_4, &expected) < 1e-12,
        "radix [3,4] cache path must match direct DFT"
    );

    let mut radix_4_3 = input;
    forward_inplace_with_radices(&mut radix_4_3, &[4, 3]);
    assert!(
        max_err(&radix_4_3, &expected) < 1e-12,
        "radix [4,3] cache path must not reuse [3,4] twiddles"
    );
}

#[test]
fn forward_lowered_radix4_tail_n12_matches_direct() {
    let input: Vec<Complex64> = (0..12)
        .map(|i| Complex64::new((i as f64 * 0.41).sin(), (i as f64 * 0.17).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut got = input;
    forward_inplace_with_radices(&mut got, &[3, 4]);
    assert!(
        max_err(&got, &expected) < 1e-12,
        "lowered radix [3,4] path must match direct DFT"
    );
}

#[test]
fn forward_lowered_radix4_tail_n192_matches_direct() {
    let input: Vec<Complex64> = (0..192)
        .map(|i| Complex64::new((i as f64 * 0.23).sin(), (i as f64 * 0.31).cos()))
        .collect();
    let expected = dft_forward(&input);
    let mut got = input;
    forward_inplace_with_radices(&mut got, &[3, 4, 4, 4]);
    assert!(
        max_err(&got, &expected) < 1e-10,
        "lowered radix [3,4,4,4] path must match direct DFT"
    );
}

#[test]
fn forward_n24() {
    check_forward(24, 1e-12);
}
#[test]
fn forward_n48() {
    check_forward(48, 1e-12);
}
#[test]
fn forward_n192() {
    check_forward(192, 1e-11);
}
#[test]
fn forward_n384() {
    check_forward(384, 1e-10);
}

#[test]
fn roundtrip_n100() {
    check_roundtrip(100, 1e-12);
}
#[test]
fn roundtrip_n1000() {
    check_roundtrip(1000, 1e-11);
}
#[test]
fn roundtrip_n14() {
    check_roundtrip(14, 1e-12);
}
#[test]
fn roundtrip_n10000() {
    check_roundtrip(10000, 1e-10);
}

#[test]
fn inverse_n14() {
    check_inverse(14, 1e-12);
}
#[test]
fn inverse_n100() {
    check_inverse(100, 1e-11);
}
#[test]
fn inverse_n1000() {
    check_inverse(1000, 1e-10);
}

#[test]
fn forward_dc_n100() {
    let mut buf = vec![Complex64::new(1.0, 0.0); 100];
    forward_inplace_64(&mut buf);
    assert!((buf[0] - Complex64::new(100.0, 0.0)).norm() < 1e-10);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-10, "non-zero bin: {:?}", x);
    }
}

#[test]
fn forward_dc_n1000() {
    let mut buf = vec![Complex64::new(1.0, 0.0); 1000];
    forward_inplace_64(&mut buf);
    assert!((buf[0] - Complex64::new(1000.0, 0.0)).norm() < 1e-9);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-9, "non-zero bin: {:?}", x);
    }
}

#[test]
fn forward_reduced_n100_matches_precise_reference() {
    let input: Vec<Complex64> = (0..100usize)
        .map(|k| Complex64::new((k as f64 * 0.29).sin(), (k as f64 * 0.47).cos()))
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
    assert!(err < 1e-4, "f32 forward N=100 max_err={err:.2e}");
}
