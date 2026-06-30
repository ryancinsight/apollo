//! Slice/`Vec`-based 1D FFT API: round-trip and parity with the `Array1` API.

use apollo_fft::{
    f16, fft_1d_array_typed, fft_1d_leto_typed, fft_1d_slice_typed, ifft_1d_array_typed,
    ifft_1d_leto_typed, ifft_1d_slice_typed,
};
use leto::{SliceArg, Storage};
use leto::Array1;

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
    let via_array = fft_1d_array_typed::<f32>(&Array1::from_shape_vec([signal.len()], signal.clone()).unwrap());
    assert_eq!(via_slice.len(), via_array.size());
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

#[test]
fn slice_matches_array_api_f16_storage() {
    let signal = vec![
        f16::from_f32(0.3),
        f16::from_f32(-1.2),
        f16::from_f32(4.5),
        f16::from_f32(2.0),
        f16::from_f32(-0.7),
        f16::from_f32(1.1),
        f16::from_f32(3.3),
        f16::from_f32(-2.5),
    ];
    let via_slice = fft_1d_slice_typed::<f16>(&signal);
    let via_array = fft_1d_array_typed::<f16>(&Array1::from_shape_vec([signal.len()], signal.clone()).unwrap());
    assert_eq!(via_slice.len(), via_array.size());
    for i in 0..via_slice.len() {
        assert_eq!(via_slice[i].re, via_array[i].re, "re mismatch at {i}");
        assert_eq!(via_slice[i].im, via_array[i].im, "im mismatch at {i}");
    }

    let inv_slice = ifft_1d_slice_typed::<f16>(&via_slice);
    let inv_array = ifft_1d_array_typed::<f16>(&via_array);
    for i in 0..inv_slice.len() {
        assert_eq!(inv_slice[i], inv_array[i], "inverse mismatch at {i}");
    }
}

#[test]
fn leto_fft_matches_ndarray_array_api_f32() {
    let signal = vec![0.3f32, -1.2, 4.5, 2.0, -0.7, 1.1, 3.3, -2.5];
    let leto_input = leto::Array1::from_shape_vec([signal.len()], signal.clone()).unwrap();
    let via_leto = fft_1d_leto_typed::<f32>(leto_input.view());
    let via_array = fft_1d_array_typed::<f32>(&Array1::from_shape_vec([signal.len()], signal).unwrap());

    assert_eq!(via_leto.shape(), [via_array.size()]);
    assert_eq!(via_leto.strides(), [1]);
    assert_eq!(via_leto.storage().as_slice(), via_array.as_slice().unwrap());

    let leto_inverse = ifft_1d_leto_typed::<f32>(via_leto.view());
    let array_inverse = ifft_1d_array_typed::<f32>(&via_array);
    assert_eq!(
        leto_inverse.storage().as_slice(),
        array_inverse.as_slice().unwrap()
    );
}

#[test]
fn leto_fft_accepts_strided_view_and_matches_logical_ndarray_values() {
    let logical = vec![0.3f32, -1.2, 4.5, 2.0, -0.7, 1.1, 3.3, -2.5];
    let interleaved = logical
        .iter()
        .flat_map(|&value| [value, 99.0])
        .collect::<Vec<_>>();
    let leto_input = leto::Array1::from_shape_vec([interleaved.len()], interleaved).unwrap();
    let strided = leto_input
        .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
        .unwrap();

    let via_leto = fft_1d_leto_typed::<f32>(strided);
    let via_array = fft_1d_array_typed::<f32>(&Array1::from_shape_vec([logical.len()], logical).unwrap());
    assert_eq!(via_leto.storage().as_slice(), via_array.as_slice().unwrap());
}
