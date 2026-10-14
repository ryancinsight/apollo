//! Leto view overloads: the `_leto` variants accept Leto views and return
//! Mnemosyne-backed Leto arrays holding the same spectra as the slice entries.

extern crate apollo_fft;
extern crate leto;

use apollo_fft::{fft_1d_array, fft_1d_leto, fft_2d_leto};
use leto::{Array1, Array2};

fn main() {
    let signal: Array1<f64> =
        Array1::from_vec([8], vec![1.0, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75])
            .expect("valid signal");
    let out = fft_1d_leto(signal.view()); // Leto view -> Leto output
    println!("leto spectrum length : {}", out.size());
    assert_eq!(out.size(), 8);
    let owned = fft_1d_array(&signal);
    for (via_view, via_slice) in out
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(owned.as_slice().expect("contiguous").iter())
    {
        assert!((via_view.re - via_slice.re).abs() < 1.0e-12);
        assert!((via_view.im - via_slice.im).abs() < 1.0e-12);
    }

    let plane: Array2<f64> =
        Array2::from_vec([4, 4], (0..16).map(|v| v as f64).collect()).expect("valid plane");
    let out2 = fft_2d_leto(plane.view());
    println!("leto spectrum shape  : {:?}", out2.shape());
    assert_eq!(out2.shape(), [4, 4]);

    println!("Leto overloads match the slice entries");
}
