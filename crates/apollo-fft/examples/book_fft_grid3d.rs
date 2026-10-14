//! Three-dimensional rfft frequency grid.
//!
//! Like the 2-D grid, only the last axis is halved: full `nx` and `ny`
//! frequencies with `nz/2 + 1` non-repeated frequencies along z.

extern crate apollo_fft;

use apollo_fft::{fftfreq, rfftfreq};

fn main() {
    let fx = fftfreq(4, 1.0);
    let fy = fftfreq(4, 1.0);
    let fz = rfftfreq(8, 0.5);
    println!("fx bins : {}", fx.len());
    println!("fy bins : {}", fy.len());
    println!("fz bins : {}", fz.len());
    assert_eq!(fx.len(), 4);
    assert_eq!(fy.len(), 4);
    assert_eq!(fz.len(), 5);
    // Nyquist bin: 1 / (2 * dz).
    assert!((fz[4] - 1.0).abs() < 1.0e-12);

    println!("3-D rfft grid assertions passed");
}
