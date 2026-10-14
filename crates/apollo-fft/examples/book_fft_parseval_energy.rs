//! Parseval energy check through the forward FFT.
//!
//! Apollo normalizes the inverse by the volume, so the time-domain energy
//! equals the spectral energy divided by `N`.

extern crate apollo_fft;
extern crate leto;

use apollo_fft::fft_1d_array;
use leto::Array1;

fn main() {
    let signal: Array1<f64> =
        Array1::from_vec([8], vec![1.0, -2.0, 0.5, 2.25, 0.0, -1.0, 3.0, 0.5])
            .expect("valid signal");
    let spectrum = fft_1d_array(&signal);

    let time_energy: f64 = signal.iter().map(|x| x * x).sum();
    let freq_energy: f64 = spectrum
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .sum::<f64>()
        / signal.size() as f64;
    println!("time energy : {time_energy:.12}");
    println!("freq energy : {freq_energy:.12}");
    assert!((time_energy - freq_energy).abs() < 1.0e-10);

    println!("Parseval energy assertions passed");
}
