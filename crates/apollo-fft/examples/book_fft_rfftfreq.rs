//! One-dimensional rfft frequency grid.
//!
//! `rfftfreq` returns the `N/2 + 1` non-repeated sample frequencies of a
//! real transform in cycles per sample, scaled by the sample spacing.

extern crate apollo_fft;

use apollo_fft::rfftfreq;

fn main() {
    let freqs = rfftfreq(8, 0.001);
    println!("grid length : {}", freqs.len());
    println!("first bin   : {} Hz", freqs[0]);
    println!("last bin    : {} Hz", freqs[freqs.len() - 1]);
    assert_eq!(freqs.len(), 5);
    assert_eq!(freqs[0], 0.0);
    // Nyquist bin: 1 / (2 * dt).
    assert!((freqs[4] - 500.0).abs() < 1.0e-9);

    println!("1-D rfft grid assertions passed");
}
