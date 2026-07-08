use leto::Array1;

/// Symmetric Hann analysis window of length `n`.
///
/// Formula: w\[i\] = 0.5 - 0.5 * cos(2*pi*i / (n-1)), i = 0..n-1.
///
/// For n=1, returns `[1.0]`. For n > 1, w\[0\] = w\[n-1\] = 0.0.
/// Normalization: unnormalized (peak = 1.0 at i=(n-1)/2).
#[must_use]
pub fn hann_window(n: usize) -> Array1<f64> {
    if n == 1 {
        return Array1::from(vec![1.0]);
    }
    Array1::from_shape_fn([n], |[i]| {
        0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
    })
}
