//! Slice/`Vec`-based 1D FFT API: round-trip and parity with the `Array1` API.

use apollo_fft::{
    fft_1d_array_typed, fft_1d_slice_typed, ifft_1d_array_typed, ifft_1d_slice_typed,
};
use ndarray::Array1;

#[test]
fn slice_fft_roundtrip_f64() {
    let signal = vec![1.0f64, 2.0, 1.0, -1.0, 1.5, 3.0, 2.2, 0.5];
    let spectrum = fft_1d_slice_typed::<f64>(&signal);
    assert_eq!(spectrum.len(), signal.len());
    let recon = ifft_1d_slice_typed::<f64>(&spectrum);
    assert_eq!(recon.len(), signal.len());
    for (i, (&o, &r)) in signal.iter().zip(&recon).enumerate() {
        assert!(
            (o - r).abs() < 1e-9,
            "roundtrip mismatch at {i}: {o} vs {r}"
        );
    }
}

#[test]
fn slice_matches_array_api_f32() {
    let signal = vec![0.3f32, -1.2, 4.5, 2.0, -0.7, 1.1, 3.3, -2.5];
    let via_slice = fft_1d_slice_typed::<f32>(&signal);
    let via_array = fft_1d_array_typed::<f32>(&Array1::from_vec(signal.clone()));
    assert_eq!(via_slice.len(), via_array.len());
    for i in 0..via_slice.len() {
        assert_eq!(via_slice[i].re, via_array[i].re, "re mismatch at {i}");
        assert_eq!(via_slice[i].im, via_array[i].im, "im mismatch at {i}");
    }
    // Inverse parity too.
    let inv_slice = ifft_1d_slice_typed::<f32>(&via_slice);
    let inv_array = ifft_1d_array_typed::<f32>(&via_array);
    for i in 0..inv_slice.len() {
        assert_eq!(inv_slice[i], inv_array[i], "inverse mismatch at {i}");
    }
}
