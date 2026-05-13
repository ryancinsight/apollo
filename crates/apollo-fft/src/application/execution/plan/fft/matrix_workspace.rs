//! Matrix-factorized FFT orchestration over planar working storage.
//!
//! # Six-step factorization theorem
//!
//! Let `N = N1 * N2` and write each input index as `n = n1 * N2 + n2`
//! with `0 <= n1 < N1`, `0 <= n2 < N2`. For each output index
//! `k = k2 * N1 + k1`, the DFT is
//!
//! `X[k1 + N1*k2] = sum_n x[n] exp(-2πi n k / N)`.
//!
//! Substituting the index decompositions and separating factors gives:
//!
//! `X[k1 + N1*k2] = sum_{n2} exp(-2πi n2*k2/N2)
//!                  sum_{n1} x[n1,n2] exp(-2πi n1*k1/N1)
//!                  exp(-2πi n1*n2/N)`.
//!
//! Therefore the 1D DFT is computed by:
//! 1. `N2` independent `N1`-point column DFTs.
//! 2. Multiplication by twiddle `exp(-2πi n1*n2/N)`.
//! 3. `N1` independent `N2`-point row DFTs.
//! 4. Transposed natural-order write `out[k1 + N1*k2]`.
//!
//! For the current vertical slice `N1 = 5`, so the infrastructure radix-5
//! batched kernel consumes one matrix column per SIMD lane.

#![allow(clippy::uninit_vec)]

use num_complex::Complex32;

use std::marker::PhantomData;

/// Zero-cost execution contract for six-step f32 column and row kernels.
///
/// # Dependency inversion contract
///
/// The application layer owns the six-step matrix factorization and calls this
/// trait for concrete short-column and row FFT execution. Infrastructure
/// implements the trait for CPU or accelerator kernels. Because the workspace
/// is generic over `K`, every call is monomorphized and inlined at compile time;
/// no vtable or heap allocation is introduced.
pub(crate) trait SixStepF32Kernel {
    /// Build a contiguous row FFT twiddle table for a power-of-two row length.
    fn build_forward_row_twiddles(n: usize) -> Vec<Complex32>;

    /// Build a contiguous inverse row FFT twiddle table for a power-of-two row length.
    fn build_inverse_row_twiddles(n: usize) -> Vec<Complex32>;

    /// Execute the forward column sweep for `R`-point batches.
    fn column_forward_real<const R: usize>(
        input: &[f32],
        n2: usize,
        re: &mut [f32],
        im: &mut [f32],
    );

    /// Execute the unnormalized inverse column sweep and write scaled real output.
    fn column_inverse_real<const R: usize>(
        re: &[f32],
        im: &[f32],
        n2: usize,
        output: &mut [f32],
        scale: f32,
    );

    /// Load contiguous planar workspace into a complex row while applying forward six-step twiddles.
    fn load_row_and_twiddle_forward(
        re: &[f32],
        im: &[f32],
        twiddle_re: &[f32],
        twiddle_im: &[f32],
        row: &mut [Complex32],
        radix_idx: usize,
        n2: usize,
    );

    /// Store complex row into contiguous planar workspace while applying inverse six-step twiddles.
    fn store_row_and_twiddle_inverse(
        row: &[Complex32],
        re: &mut [f32],
        im: &mut [f32],
        twiddle_re: &[f32],
        twiddle_im: &[f32],
        radix_idx: usize,
        n2: usize,
    );

    /// Execute one forward row FFT using plan-owned scratch.
    fn row_forward(row: &mut [Complex32], scratch: &mut [Complex32], twiddles: &[Complex32]);

    /// Execute one unnormalized inverse row FFT using plan-owned scratch.
    fn row_inverse_unnorm(row: &mut [Complex32], scratch: &mut [Complex32], twiddles: &[Complex32]);

    /// Write one row into the transposed natural-order output.
    fn write_transposed_row(
        row: &[Complex32],
        output: &mut [Complex32],
        radix_row: usize,
        radix: usize,
    );
}

/// Plan-owned native-f32 six-step workspace.
///
/// All heap storage is allocated during plan construction. `forward_real`
/// mutates only these owned buffers and caller-owned input/output slices.
pub(crate) struct SixStepF32Workspace<K: SixStepF32Kernel> {
    radix: usize,
    n2: usize,
    re: Vec<f32>,
    im: Vec<f32>,
    twiddle_re: Vec<f32>,
    twiddle_im: Vec<f32>,
    row: Vec<Complex32>,
    row_scratch: Vec<Complex32>,
    row_twiddles: Vec<Complex32>,
    inverse_row_twiddles: Vec<Complex32>,
    kernel: PhantomData<K>,
}

fn uninit_f32_vec(len: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    // SAFETY: f32 has no drop glue. Callers only use this for work buffers
    // whose full contents are overwritten before any read.
    unsafe { values.set_len(len) };
    values
}

fn uninit_complex32_vec(len: usize) -> Vec<Complex32> {
    let mut values = Vec::with_capacity(len);
    // SAFETY: Complex32 is two f32 values and has no drop glue. Callers only
    // use this for work buffers whose full contents are overwritten first.
    unsafe { values.set_len(len) };
    values
}

impl<K: SixStepF32Kernel> SixStepF32Workspace<K> {
    /// Create workspace for lengths `N = R * N2` where `R ∈ {3,5,7,11}` and
    /// `N2` is power-of-two.
    pub(crate) fn new(n: usize) -> Option<Self> {
        let radix = [3_usize, 5, 7, 11]
            .into_iter()
            .find(|&candidate| n % candidate == 0 && (n / candidate).is_power_of_two())?;
        let n2 = n / radix;
        if !n2.is_power_of_two() {
            return None;
        }
        Some(Self {
            radix,
            n2,
            re: uninit_f32_vec(n),
            im: uninit_f32_vec(n),
            twiddle_re: build_twiddle_component(radix, n2, false, true),
            twiddle_im: build_twiddle_component(radix, n2, false, false),
            row: uninit_complex32_vec(n2),
            row_scratch: uninit_complex32_vec(n2),
            row_twiddles: K::build_forward_row_twiddles(n2),
            inverse_row_twiddles: K::build_inverse_row_twiddles(n2),
            kernel: PhantomData,
        })
    }

    /// Execute an unnormalized forward FFT and write natural-order spectrum.
    ///
    /// # Allocation contract
    ///
    /// This function performs no heap allocation. The input and output are
    /// caller-owned contiguous slices; all intermediate storage is owned by the
    /// plan workspace.
    pub(crate) fn forward_real(&mut self, input: &[f32], output: &mut [Complex32]) {
        let n = self.radix * self.n2;
        assert_eq!(input.len(), n, "six-step input length mismatch");
        assert_eq!(output.len(), n, "six-step output length mismatch");
        match self.radix {
            3 => K::column_forward_real::<3>(input, self.n2, &mut self.re, &mut self.im),
            5 => K::column_forward_real::<5>(input, self.n2, &mut self.re, &mut self.im),
            7 => K::column_forward_real::<7>(input, self.n2, &mut self.re, &mut self.im),
            11 => K::column_forward_real::<11>(input, self.n2, &mut self.re, &mut self.im),
            _ => unreachable!("unsupported six-step radix"),
        }
        self.row_fft_and_transposed_write(output);
    }

    /// Execute a normalized inverse FFT from natural-order spectrum to real output.
    ///
    /// # Inverse six-step theorem
    ///
    /// With `N = R*N2`, write frequency as `k = k1 + R*k2` and time as
    /// `n = n1*N2 + n2`. The inverse DFT is
    /// `x[n] = (1/N) Σ_k X[k] exp(+2πi nk/N)`. Substitution gives
    /// `exp(+2πi n1*k1/R) * exp(+2πi n2*k1/N) *
    /// exp(+2πi n2*k2/N2)` because `n1*k2` is an integer phase. Therefore the
    /// inverse is computed by unnormalized inverse row FFTs over `k2`,
    /// multiplication by `exp(+2πi n2*k1/N)`, unnormalized inverse column FFTs
    /// over `k1`, and one final multiplication by `1/N`.
    ///
    /// # Normalization proof
    ///
    /// The row inverse contributes no scale, and the prime column inverse
    /// contributes no scale. The only scaling in this routine is `1/(R*N2)`,
    /// so the result matches Apollo's FFTW-compatible normalized inverse
    /// convention.
    pub(crate) fn inverse_real(&mut self, input: &[Complex32], output: &mut [f32]) {
        let n = self.radix * self.n2;
        assert_eq!(input.len(), n, "six-step inverse input length mismatch");
        assert_eq!(output.len(), n, "six-step inverse output length mismatch");
        self.inverse_rows_from_transposed_input(input);
        let scale = 1.0 / n as f32;
        match self.radix {
            3 => K::column_inverse_real::<3>(&self.re, &self.im, self.n2, output, scale),
            5 => K::column_inverse_real::<5>(&self.re, &self.im, self.n2, output, scale),
            7 => K::column_inverse_real::<7>(&self.re, &self.im, self.n2, output, scale),
            11 => K::column_inverse_real::<11>(&self.re, &self.im, self.n2, output, scale),
            _ => unreachable!("unsupported six-step radix"),
        }
    }

    /// Run the `R` independent `N2`-point inverse row transforms.
    ///
    /// Input is in natural 1D frequency order `k1 + R*k2`; rows are therefore
    /// gathered with stride `R` into contiguous row scratch before the inverse
    /// row FFT. The row result is written back as `(k1, n2)` planar workspace.
    ///
    fn inverse_rows_from_transposed_input(&mut self, input: &[Complex32]) {
        for r in 0..self.radix {
            for c in 0..self.n2 {
                self.row[c] = input[r + self.radix * c];
            }
            K::row_inverse_unnorm(
                &mut self.row,
                &mut self.row_scratch,
                &self.inverse_row_twiddles,
            );
            K::store_row_and_twiddle_inverse(
                &self.row,
                &mut self.re,
                &mut self.im,
                &self.twiddle_re,
                &self.twiddle_im,
                r,
                self.n2,
            );
        }
    }

    /// Run row FFTs and write the transposed natural-order spectrum.
    ///
    /// The row FFT produces `Y[k2]` for fixed `k1`. Natural 1D order for
    /// `k = k1 + N1*k2` is therefore a matrix transpose from row-major
    /// `(k1, k2)` to flat output.
    ///
    fn row_fft_and_transposed_write(&mut self, output: &mut [Complex32]) {
        for r in 0..self.radix {
            K::load_row_and_twiddle_forward(
                &self.re,
                &self.im,
                &self.twiddle_re,
                &self.twiddle_im,
                &mut self.row,
                r,
                self.n2,
            );
            K::row_forward(&mut self.row, &mut self.row_scratch, &self.row_twiddles);
            K::write_transposed_row(&self.row, output, r, self.radix);
        }
    }
}

fn build_twiddle_component(radix: usize, n2: usize, inverse: bool, real: bool) -> Vec<f32> {
    let len = radix * n2;
    let n = len as f32;
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut values = uninit_f32_vec(len);
    let mut idx = 0;
    for r in 0..radix {
        for c in 0..n2 {
            let angle = sign * 2.0 * std::f32::consts::PI * (r * c) as f32 / n;
            // SAFETY: idx increments once for each element in the radix x n2
            // rectangular iteration and therefore stays below len.
            unsafe {
                *values.get_unchecked_mut(idx) = if real { angle.cos() } else { angle.sin() };
            }
            idx += 1;
        }
    }
    debug_assert_eq!(idx, len);
    values
}

#[cfg(test)]
mod tests {
    use crate::application::execution::kernel::cpu_matrix_fft::CpuSixStepF32Plan;
    use crate::application::execution::kernel::FftPrecision;
    use num_complex::Complex32;

    #[test]
    fn workspace_selects_supported_prime_radix_times_power_of_two() {
        assert!(CpuSixStepF32Plan::new(12).is_some());
        assert!(CpuSixStepF32Plan::new(20).is_some());
        assert!(CpuSixStepF32Plan::new(28).is_some());
        assert!(CpuSixStepF32Plan::new(44).is_some());
    }

    fn assert_six_step_matches_direct(n: usize, tolerance: f32) {
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32;
                (0.11 * x).sin() + 0.25 * (0.37 * x).cos()
            })
            .collect();
        let mut expected: Vec<Complex32> = input
            .iter()
            .map(|value| Complex32::new(*value, 0.0))
            .collect();
        Complex32::fft_forward(&mut expected);
        let mut actual = vec![Complex32::new(0.0, 0.0); n];
        CpuSixStepF32Plan::new(n)
            .expect("six-step workspace")
            .forward_real(&input, &mut actual);
        for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            let err = (*got - *want).norm();
            assert!(
                err <= tolerance,
                "n={n} idx={idx} err={err:.4e} got={got:?} expected={want:?}"
            );
        }
    }

    #[test]
    fn six_step_radix3_7_11_match_direct_f32_reference() {
        assert_six_step_matches_direct(12, 2.0e-2);
        assert_six_step_matches_direct(28, 3.0e-2);
        assert_six_step_matches_direct(44, 4.0e-2);
    }

    #[test]
    fn six_step_inverse_applies_one_over_n_normalization() {
        let n = 5 * 64;
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32;
                (0.07 * x).sin() + 0.35 * (0.19 * x).cos()
            })
            .collect();
        let mut spectrum = vec![Complex32::new(0.0, 0.0); n];
        let mut recovered = vec![0.0_f32; n];
        let mut plan = CpuSixStepF32Plan::new(n).expect("six-step workspace");

        plan.forward_real(&input, &mut spectrum);
        plan.inverse_real(&spectrum, &mut recovered);

        for (idx, (got, want)) in recovered.iter().zip(input.iter()).enumerate() {
            let err = (*got - *want).abs();
            assert!(
                err <= 8.0e-4,
                "idx={idx} err={err:.4e} got={got:.8e} expected={want:.8e}"
            );
        }
    }

    #[test]
    fn six_step_twiddle_and_transpose_are_backend_hooks() {
        let source = include_str!("matrix_workspace.rs");
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source, |(production, _)| production);
        for forbidden in [
            "fn apply_forward_twiddles(&mut self",
            "fn apply_inverse_twiddles(&mut self",
            "fn blocked_transposed_row_write",
        ] {
            assert!(
                !production.contains(forbidden),
                "application matrix workspace must not own hardware-specific `{forbidden}`"
            );
        }
    }

    #[test]
    fn six_step_column_sweep_does_not_materialize_stack_lane_tiles() {
        let source = include_str!("matrix_workspace.rs");
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source, |(production, _)| production);
        for forbidden in ["[[0.0_f32; LANES]", "FftPlanarMut::new"] {
            assert!(
                !production.contains(forbidden),
                "application column sweep must not materialize `{forbidden}`"
            );
        }
    }
}

