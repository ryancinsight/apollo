//! Two-dimensional rfft frequency grid.
//!
//! Apollo halves the last axis, so a 2-D rfft grid pairs the full `nx`
//! frequencies along x with the `ny/2 + 1` non-repeated frequencies
//! along y.

extern crate apollo_fft;

use apollo_fft::{fftfreq, rfftfreq};

fn main() {
    let fx = fftfreq(4, 1.0);
    let fy = rfftfreq(4, 1.0);
    println!("fx bins : {}", fx.len());
    println!("fy bins : {}", fy.len());
    assert_eq!(fx.len(), 4);
    assert_eq!(fy.len(), 3);
    // Full axis keeps the negative frequencies; the halved axis stops at
    // Nyquist.
    assert!(fx.iter().any(|f| *f < 0.0));
    assert!(fy.iter().all(|f| *f >= 0.0));
    assert!((fy[2] - 0.5).abs() < 1.0e-12);

    println!("2-D rfft grid assertions passed");
}
