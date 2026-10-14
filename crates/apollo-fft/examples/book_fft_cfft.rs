//! Full-spectrum complex FFT in one and three dimensions.
//!
//! A complex input transforms to a full complex spectrum of the same shape.
//! The DC bin holds the complex input sum.

extern crate apollo_fft;
extern crate leto;

use apollo_fft::{fft_1d_complex, fft_3d_complex, Complex64};
use leto::{Array1, Array3};

fn main() {
    let signal = Array1::from_vec(
        [4],
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(-2.0, 1.0),
            Complex64::new(0.5, -0.5),
            Complex64::new(2.25, 0.0),
        ],
    )
    .expect("valid signal");
    let out = fft_1d_complex(&signal);
    println!("signal length   : {}", signal.size());
    println!("spectrum length : {}", out.size());
    assert_eq!(out.size(), 4);
    // DC bin: the complex input sum (1.75 + 1.0i).
    assert!((out[[0]].re - 1.75).abs() < 1.0e-9);
    assert!((out[[0]].im - 1.0).abs() < 1.0e-9);

    let volume_in = Array3::from_vec(
        [2, 2, 2],
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
    .expect("valid volume");
    let out3 = fft_3d_complex(&volume_in);
    println!("volume shape    : {:?}", volume_in.shape());
    println!("spectrum shape  : {:?}", out3.shape());
    assert_eq!(out3.shape(), [2, 2, 2]);
    // DC bin: the complex input sum (4.5 + 2.5i).
    assert!((out3[[0, 0, 0]].re - 4.5).abs() < 1.0e-9);
    assert!((out3[[0, 0, 0]].im - 2.5).abs() < 1.0e-9);

    println!("cfft shape assertions passed");
}
