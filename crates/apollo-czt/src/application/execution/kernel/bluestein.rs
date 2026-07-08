use apollo_fft::FftPlan1D;
use eunomia::Complex64;
use leto::Array1;
use mnemosyne::scratch::ScratchPool;
use moirai::ParallelSliceMut;

/// Below this contiguous element count, serial loops avoid scheduling overhead.
const BLUESTEIN_PAR_LEN_THRESHOLD: usize = 16_384;

thread_local! {
    static COMPLEX_SCRATCH_POOL: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Evaluates Bluestein's fast algorithm over precomputed chirp variables
/// using an optimized invariant `fft_kernel`.
///
/// # Theorem
///
/// For the chirp z-transform
/// `X_k = sum_n x_n a^-n w^(nk)`, the identity
/// `nk = (n^2 + k^2 - (k - n)^2) / 2` rewrites the transform as a linear
/// convolution between `x_n a^-n w^(n^2/2)` and `w^(-(k-n)^2/2)`, followed by
/// multiplication by `w^(k^2/2)`. This function evaluates that convolution
/// with Apollo FFT plans over zero-padded buffers.
///
/// # Proof sketch
///
/// Substituting the identity into `w^(nk)` factors all `n`-only and `k`-only
/// terms out of the summation. The remaining `(k-n)` term is Toeplitz and
/// becomes a cyclic convolution after zero-padding to at least `n + m - 1`.
/// The FFT convolution theorem then gives the same result as direct CZT in
/// exact arithmetic.
#[must_use]
pub fn czt_bluestein_forward(
    input: &Array1<Complex64>,
    output_len: usize,
    convolution_len: usize,
    chirp_n: &[Complex64],
    chirp_k: &[Complex64],
    fft_kernel: &Array1<Complex64>,
    fft_plan: &FftPlan1D<f64>,
) -> Array1<Complex64> {
    let mut output = Array1::<Complex64>::zeros([output_len]);
    czt_bluestein_forward_into(
        input,
        &mut output,
        convolution_len,
        chirp_n,
        chirp_k,
        fft_kernel,
        fft_plan,
    );
    output
}

/// Evaluates Bluestein's fast CZT into caller-owned output storage.
///
/// This path uses one convolution workspace. The workspace is transformed,
/// multiplied by the precomputed FFT kernel in place, inverse transformed in
/// place, and then sampled into `output`.
pub fn czt_bluestein_forward_into(
    input: &Array1<Complex64>,
    output: &mut Array1<Complex64>,
    convolution_len: usize,
    chirp_n: &[Complex64],
    chirp_k: &[Complex64],
    fft_kernel: &Array1<Complex64>,
    fft_plan: &FftPlan1D<f64>,
) {
    COMPLEX_SCRATCH_POOL.with(|pool| {
        pool.with_scratch(convolution_len, |workspace| {
            czt_bluestein_forward_into_with_workspace(
                input.as_slice().expect("CZT input must be contiguous"),
                output
                    .as_slice_mut()
                    .expect("CZT output must be contiguous"),
                workspace,
                convolution_len,
                chirp_n,
                chirp_k,
                fft_kernel
                    .as_slice()
                    .expect("CZT FFT kernel must be contiguous"),
                fft_plan,
            );
        });
    });
}

/// Evaluates Bluestein's fast CZT into caller-owned output using caller-owned workspace.
///
/// The workspace length must equal `convolution_len`. Reusing it across calls
/// removes the O(P) allocation from the hot path while preserving the same
/// Bluestein convolution identity as `czt_bluestein_forward_into`.
pub(crate) fn czt_bluestein_forward_into_with_workspace(
    input: &[Complex64],
    output: &mut [Complex64],
    workspace: &mut [Complex64],
    convolution_len: usize,
    chirp_n: &[Complex64],
    chirp_k: &[Complex64],
    fft_kernel: &[Complex64],
    fft_plan: &FftPlan1D<f64>,
) {
    assert_eq!(
        output.len(),
        chirp_k.len(),
        "CZT output length must match chirp_k length"
    );
    assert_eq!(
        input.len(),
        chirp_n.len(),
        "CZT input length must match chirp_n length"
    );
    assert_eq!(
        fft_kernel.len(),
        convolution_len,
        "CZT FFT kernel length must match convolution length"
    );
    assert_eq!(
        workspace.len(),
        convolution_len,
        "CZT workspace length must match convolution length"
    );

    prepare_workspace(input, workspace, chirp_n);

    fft_plan.forward_complex_slice_inplace(workspace);

    multiply_kernel(workspace, fft_kernel);

    fft_plan.inverse_complex_slice_inplace(workspace);

    sample_output(output, workspace, chirp_k);
}

fn prepare_workspace(input: &[Complex64], workspace: &mut [Complex64], chirp_n: &[Complex64]) {
    if workspace.len() >= BLUESTEIN_PAR_LEN_THRESHOLD {
        workspace.par_mut().enumerate(|index, value| {
            *value = if index < input.len() {
                input[index] * chirp_n[index]
            } else {
                Complex64::new(0.0, 0.0)
            };
        });
    } else {
        workspace.fill(Complex64::new(0.0, 0.0));
        for n_idx in 0..input.len() {
            workspace[n_idx] = input[n_idx] * chirp_n[n_idx];
        }
    }
}

fn multiply_kernel(workspace: &mut [Complex64], fft_kernel: &[Complex64]) {
    if workspace.len() >= BLUESTEIN_PAR_LEN_THRESHOLD {
        workspace.par_mut().enumerate(|index, value| {
            *value *= fft_kernel[index];
        });
    } else {
        for (value, &kernel_value) in workspace.iter_mut().zip(fft_kernel.iter()) {
            *value *= kernel_value;
        }
    }
}

fn sample_output(output: &mut [Complex64], workspace: &[Complex64], chirp_k: &[Complex64]) {
    if output.len() >= BLUESTEIN_PAR_LEN_THRESHOLD {
        output.par_mut().enumerate(|k, out| {
            *out = chirp_k[k] * workspace[k];
        });
    } else {
        for (k, out) in output.iter_mut().enumerate() {
            *out = chirp_k[k] * workspace[k];
        }
    }
}

/// Computes the inverse chirp z-transform via the Björck–Pereyra Vandermonde solve.
///
/// # Mathematical contract
///
/// The forward CZT evaluates `X[k] = sum_{n=0}^{N-1} x[n] A^{-n} W^{nk}` for
/// `k = 0..N-1`.  Writing `y[n] = x[n] A^{-n}` this is the Vandermonde system
/// `V y = X` where `V[k,n] = z_k^n` and `z_k = W^k`.  Recovering `y` from `X`
/// then gives `x[n] = y[n] A^n`.
///
/// # Algorithm
///
/// Björck & Pereyra (1970) "Solution of Vandermonde Systems of Equations",
/// Math. Comput. 24(112): 893-903.  The algorithm is O(N²) in time and O(N)
/// in additional space, producing the exact polynomial coefficients of the
/// unique interpolating polynomial through the N Vandermonde nodes `z_k`.
///
/// # Errors
///
/// Returns `CztError::NotInvertible` when two evaluation points coincide
/// (`z_k = z_j` for some `k ≠ j`), which makes the Vandermonde matrix singular.
/// This occurs exactly when `W` is a root of unity of order `d ≤ N`.
pub fn czt_bjork_pereyra_inverse(
    spectrum: &[Complex64],
    a: Complex64,
    w: Complex64,
) -> Result<Vec<Complex64>, crate::domain::contracts::error::CztError> {
    let n = spectrum.len();
    let z = czt_inverse_nodes(n, w);
    let mut output = vec![Complex64::new(0.0, 0.0); n];
    czt_bjork_pereyra_inverse_into(spectrum, &mut output, &z, a)?;
    Ok(output)
}

/// Precompute Vandermonde nodes `z_k = W^k` for inverse CZT plans.
#[must_use]
pub(crate) fn czt_inverse_nodes(n: usize, w: Complex64) -> Vec<Complex64> {
    let mut z: Vec<Complex64> = Vec::with_capacity(n);
    let mut wk = Complex64::new(1.0, 0.0);
    for _ in 0..n {
        z.push(wk);
        wk *= w;
    }
    z
}

/// Computes the inverse chirp z-transform into caller-owned output storage.
///
/// `nodes[k]` must equal `W^k`. Passing precomputed nodes removes one O(N)
/// allocation from repeated inverse calls on the same square CZT plan.
pub(crate) fn czt_bjork_pereyra_inverse_into(
    spectrum: &[Complex64],
    output: &mut [Complex64],
    nodes: &[Complex64],
    a: Complex64,
) -> Result<(), crate::domain::contracts::error::CztError> {
    use crate::domain::contracts::error::CztError;
    let n = spectrum.len();
    if output.len() != n || nodes.len() != n {
        return Err(CztError::LengthMismatch);
    }

    // Björck-Pereyra phase 1: forward divided-differences
    output.copy_from_slice(spectrum);
    for j in 0..(n.saturating_sub(1)) {
        for k in (j + 1..n).rev() {
            let denom = nodes[k] - nodes[k - j - 1];
            if denom.norm() < f64::EPSILON * 1024.0 {
                return Err(CztError::NotInvertible {
                    reason: "Vandermonde nodes z_k collide; W is a root of unity of order <= N",
                });
            }
            output[k] = (output[k] - output[k - 1]) / denom;
        }
    }

    // Björck-Pereyra phase 2: Newton evaluation (reverse)
    for j in (0..n.saturating_sub(1)).rev() {
        for k in j..(n - 1) {
            let ck1 = output[k + 1];
            output[k] -= nodes[j] * ck1;
        }
    }

    // x[n] = y[n] * A^n  (undo the A^{-n} scaling from the forward CZT)
    let mut a_pow = Complex64::new(1.0, 0.0);
    for xn in output.iter_mut() {
        *xn *= a_pow;
        a_pow *= a;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn moirai_parallel_prepare_workspace_matches_serial_formula_at_threshold() {
        let input_len = 128usize;
        let mut workspace = vec![Complex64::new(9.0, -9.0); BLUESTEIN_PAR_LEN_THRESHOLD];
        let input = (0..input_len)
            .map(|index| Complex64::new(index as f64 * 0.25, -(index as f64) * 0.125))
            .collect::<Vec<_>>();
        let chirp_n = (0..input_len)
            .map(|index| Complex64::from_polar(1.0, index as f64 * 0.03125))
            .collect::<Vec<_>>();

        prepare_workspace(&input, &mut workspace, &chirp_n);

        for index in 0..workspace.len() {
            let expected = if index < input_len {
                input[index] * chirp_n[index]
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert_abs_diff_eq!(workspace[index].re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(workspace[index].im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn moirai_parallel_kernel_multiply_matches_elementwise_formula_at_threshold() {
        let mut workspace = (0..BLUESTEIN_PAR_LEN_THRESHOLD)
            .map(|index| Complex64::new(index as f64 * 0.125, index as f64 * -0.0625))
            .collect::<Vec<_>>();
        let before = workspace.clone();
        let fft_kernel = (0..BLUESTEIN_PAR_LEN_THRESHOLD)
            .map(|index| Complex64::from_polar(1.0, index as f64 * 0.015625))
            .collect::<Vec<_>>();

        multiply_kernel(&mut workspace, &fft_kernel);

        for index in 0..workspace.len() {
            let expected = before[index] * fft_kernel[index];
            assert_abs_diff_eq!(workspace[index].re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(workspace[index].im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn moirai_parallel_sample_output_matches_chirp_formula_at_threshold() {
        let workspace = (0..BLUESTEIN_PAR_LEN_THRESHOLD)
            .map(|index| Complex64::new(index as f64 * 0.125, index as f64 * 0.03125))
            .collect::<Vec<_>>();
        let chirp_k = (0..BLUESTEIN_PAR_LEN_THRESHOLD)
            .map(|index| Complex64::from_polar(1.0, -(index as f64) * 0.0078125))
            .collect::<Vec<_>>();
        let mut output = vec![Complex64::new(0.0, 0.0); BLUESTEIN_PAR_LEN_THRESHOLD];

        sample_output(&mut output, &workspace, &chirp_k);

        for index in 0..output.len() {
            let expected = chirp_k[index] * workspace[index];
            assert_abs_diff_eq!(output[index].re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(output[index].im, expected.im, epsilon = 1.0e-12);
        }
    }
}
