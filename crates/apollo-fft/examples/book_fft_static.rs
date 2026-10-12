//! Static plans: `StaticFftPlan1D<F, N>` embeds the precision and the
//! transform size as generics, so no plan cache lookup occurs at runtime.

extern crate apollo_fft;
extern crate leto;

use apollo_fft::{Complex64, StaticFftPlan1D};
use leto::Array1;

fn main() {
    let plan: StaticFftPlan1D<f64, 8> = StaticFftPlan1D::new();
    let mut signal = Array1::from_vec(
        [8],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.5, -1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(1.0, -1.0),
        ],
    )
    .expect("valid signal");
    let dc_re: f64 = signal
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|c| c.re)
        .sum();
    let dc_im: f64 = signal
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|c| c.im)
        .sum();

    plan.forward_complex_inplace(&mut signal);
    assert!((signal[[0]].re - dc_re).abs() < 1.0e-9);
    assert!((signal[[0]].im - dc_im).abs() < 1.0e-9);
    println!("static-plan spectrum length : {}", signal.size());

    plan.inverse_complex_inplace(&mut signal);
    let expected = [
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (-1.0, 0.5),
        (0.5, -1.0),
        (2.0, 2.0),
        (1.0, -1.0),
    ];
    for (actual, (re, im)) in signal
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(expected.iter())
    {
        assert!((actual.re - re).abs() < 1.0e-11);
        assert!((actual.im - im).abs() < 1.0e-11);
    }

    println!("static-plan round-trip assertions passed");
}
