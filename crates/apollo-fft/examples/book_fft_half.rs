//! Half-spectrum real FFT round-trips in one, two, and three dimensions.
//!
//! The half forms keep only the non-repeated bins (`N/2 + 1` along the last
//! axis) and compute only those. Each inverse consumes its half spectrum as
//! scratch and reconstructs the real input.

extern crate apollo_fft;
extern crate leto;

use apollo_fft::{
    fft_1d_slice_half_into, fft_2d_array_half_into, fft_3d_array_half_into,
    ifft_1d_slice_half_into, ifft_2d_array_half_into, ifft_3d_array_half_into,
};
use leto::{Array2, Array3};

const TOL: f64 = 1.0e-11;

fn main() {
    // [N] -> [N/2 + 1], and back.
    let signal = [1.0, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
    let mut half = [zero(); 5];
    fft_1d_slice_half_into(&signal, &mut half);
    assert!((half[0].re - signal.iter().sum::<f64>()).abs() < 1.0e-9);
    let mut back = [0.0; 8];
    ifft_1d_slice_half_into(&mut half, &mut back);
    for (actual, expected) in back.iter().zip(signal.iter()) {
        assert!((actual - expected).abs() < TOL);
    }
    println!("1-D half round-trip error within {TOL:e}");

    // [nx, ny] -> [nx, ny/2 + 1], and back.
    let plane = Array2::from_vec([4, 4], (0..16).map(|v| v as f64).collect()).expect("valid plane");
    let mut half_plane = Array2::from_vec([4, 3], vec![zero(); 12]).expect("valid half plane");
    fft_2d_array_half_into(&plane, &mut half_plane);
    let mut plane_back = Array2::from_vec([4, 4], vec![0.0; 16]).expect("valid output plane");
    ifft_2d_array_half_into(&mut half_plane, &mut plane_back);
    for (actual, expected) in plane_back
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(plane.as_slice().expect("contiguous").iter())
    {
        assert!((actual - expected).abs() < TOL);
    }
    println!("2-D half round-trip error within {TOL:e}");

    // [nx, ny, nz] -> [nx, ny, nz/2 + 1], and back.
    let field =
        Array3::from_vec([2, 2, 4], (0..16).map(|v| v as f64).collect()).expect("valid field");
    let mut half_volume = Array3::from_vec([2, 2, 3], vec![zero(); 12]).expect("valid half volume");
    fft_3d_array_half_into(&field, &mut half_volume);
    let mut field_back = Array3::from_vec([2, 2, 4], vec![0.0; 16]).expect("valid output field");
    ifft_3d_array_half_into(&mut half_volume, &mut field_back);
    for (actual, expected) in field_back
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(field.as_slice().expect("contiguous").iter())
    {
        assert!((actual - expected).abs() < TOL);
    }
    println!("3-D half round-trip error within {TOL:e}");
    println!("half-spectrum assertions passed");
}

fn zero() -> apollo_fft::Complex64 {
    apollo_fft::Complex64::new(0.0, 0.0)
}
