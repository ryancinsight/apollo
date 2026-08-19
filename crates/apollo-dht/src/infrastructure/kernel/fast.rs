//! Fast \(O(N \log N)\) FFT-based kernel execution for large arrays.
//! Leverages the exact correspondence between the Discrete Hartley Transform and the Discrete Fourier Transform.

use apollo_fft::{Complex64, PlanCacheProvider, Shape1D};
use mnemosyne::scratch::ScratchPool;

thread_local! {
    static COMPLEX_SCRATCH_POOL: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Computes the unnormalized DHT via an O(N log N) FFT mapped path using caller-owned scratch.
///
/// `scratch.len()` must equal `signal.len()`.
pub fn dht_fast_with_scratch(signal: &[f64], output: &mut [f64], scratch: &mut [Complex64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }

    debug_assert_eq!(output.len(), n);
    debug_assert_eq!(scratch.len(), n);

    for (slot, &x) in scratch.iter_mut().zip(signal.iter()) {
        *slot = Complex64::new(x, 0.0);
    }

    let plan = f64::get_1d_plan(Shape1D::new(n).expect("Shape1D"));
    plan.forward_complex_slice_inplace(scratch);

    for (out, x_k) in output.iter_mut().zip(scratch.iter()) {
        *out = x_k.re - x_k.im;
    }
}

/// Computes the unnormalized DHT via an \(O(N \log N)\) FFT mapped path.
///
/// Ensures exact mathematical equivalence:
/// \(H_k = \operatorname{Re}(X_k) - \operatorname{Im}(X_k)\) where \(X = \operatorname{FFT}(x)\).
pub fn dht_fast(signal: &[f64], output: &mut [f64]) {
    let n = signal.len();
    if n == 0 {
        return;
    }

    COMPLEX_SCRATCH_POOL.with(|pool| {
        pool.with_scratch(n, |scratch| {
            dht_fast_with_scratch(signal, output, scratch);
        });
    });
}
