//! Full-spectrum real FFT in one, two, and three dimensions.
//!
//! A real input of length `N` (or shape `[nx, ny, nz]`) transforms to a full
//! complex spectrum of the same length (or shape). The DC bin holds the
//! input sum; the remaining bins repeat as conjugates.

extern crate apollo_fft;
extern crate leto;

use apollo_fft::{fft_1d_array, fft_3d_array};
use leto::{Array1, Array3};

fn main() {
    let signal: Array1<f64> =
        Array1::from_vec([8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("valid signal");
    let spectrum = fft_1d_array(&signal);
    println!("signal length   : {}", signal.size());
    println!("spectrum length : {}", spectrum.size());
    assert_eq!(spectrum.size(), 8);
    // DC bin: the input sums to 36.
    assert!((spectrum[[0]].re - 36.0).abs() < 1.0e-9);
    assert!(spectrum[[0]].im.abs() < 1.0e-9);

    let field: Array3<f64> =
        Array3::from_vec([2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
            .expect("valid field");
    let volume = fft_3d_array(&field);
    println!("field shape     : {:?}", field.shape());
    println!("spectrum shape  : {:?}", volume.shape());
    assert_eq!(volume.shape(), [2, 2, 2]);
    assert!((volume[[0, 0, 0]].re - 4.0).abs() < 1.0e-9);

    println!("rfft shape assertions passed");
}
