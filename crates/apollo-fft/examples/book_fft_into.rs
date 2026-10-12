//! Caller-owned output: the `_into` variants write into a caller-provided
//! buffer instead of allocating, and agree with the allocating entries.

extern crate apollo_fft;
extern crate leto;

use apollo_fft::{fft_1d_array, fft_1d_array_into, Complex64};
use leto::Array1;

fn main() {
    let signal: Array1<f64> =
        Array1::from_vec([8], vec![1.0, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75])
            .expect("valid signal");
    let mut output =
        Array1::from_vec([8], vec![Complex64::new(0.0, 0.0); 8]).expect("valid output");
    fft_1d_array_into(&signal, &mut output); // no allocation
    println!("output length   : {}", output.size());

    let owned = fft_1d_array(&signal);
    for (placed, allocated) in output
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(owned.as_slice().expect("contiguous").iter())
    {
        assert!((placed.re - allocated.re).abs() < 1.0e-12);
        assert!((placed.im - allocated.im).abs() < 1.0e-12);
    }

    println!("caller-owned output matches the allocating entry");
}
