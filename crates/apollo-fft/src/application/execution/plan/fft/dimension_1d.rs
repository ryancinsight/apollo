//! 1D FFT plan.

use crate::application::execution::kernel::mixed_radix::{
    cached_twiddle_fwd_32, cached_twiddle_fwd_64, cached_twiddle_inv_32, cached_twiddle_inv_64,
    dispatch_inplace, with_stockham_scratch_64,
};
use crate::application::execution::kernel::real_fft::RealFft;
use crate::application::execution::plan::fft::real_storage::RealFftData;
use crate::application::execution::plan::fft::workspace::uninit_copy_vec;
use crate::domain::metadata::precision::PrecisionProfile;
use crate::domain::metadata::shape::Shape1D;
use ndarray::{Array1, Zip};
use num_complex::Complex32;
use num_complex::Complex64;
use std::sync::Arc;

mod precision;

/// Reusable 1D FFT plan.
///
/// The plan owns the transform length and precision contract. Forward and
/// inverse transforms use Apollo's in-repo auto-selected FFT kernel, with
/// radix-2 execution for power-of-two lengths and Bluestein execution for
/// non-power-of-two lengths. The inverse path uses FFTW-compatible
/// normalization.
///
/// # Theorem
///
/// Let `F` denote the forward discrete Fourier transform and `F⁻¹` denote the
/// inverse path with `1 / n` normalization. For every signal `x ∈ ℂⁿ`,
/// `F⁻¹(F(x)) = x` in exact arithmetic. The implementation is linear because
/// both passes are matrix-vector products against the DFT basis. The inverse
/// normalization matches the standard FFTW convention used throughout Apollo.
///
/// # Proof sketch
///
/// The forward kernel evaluates the Vandermonde form
/// `X_k = Σ_n x_n exp(-2πi kn / N)`.
/// The inverse kernel evaluates
/// `x_n = (1/N) Σ_k X_k exp(2πi kn / N)`.
/// These are exact inverses over the complex field. Floating-point roundoff is
/// bounded by the chosen precision profile and the input length.
///
/// # Complexity
///
/// Power-of-two lengths use `O(N log N)` time and `O(N)` additional plan state
/// (for the precomputed twiddle table). Non-power-of-two lengths use Bluestein
/// convolution with `O(M log M)` time and `O(M)` plan state, where `M` is the
/// next convolution length selected by the Bluestein plan.
///
/// # Precomputed twiddle tables (power-of-two optimization)
///
/// For power-of-two `N`, the plan precomputes a contiguous per-stage forward and
/// inverse twiddle table of total size `N-1` entries each. Stage s (group length
/// `len = 2^s`) occupies `len/2` contiguous entries in the table; the butterfly
/// loop reads them sequentially with no stride. This eliminates both:
/// 1. The per-call `Vec<Complex64>` allocation (N/2 entries) that the naïve path
///    allocates on every `forward_inplace_64` call.
/// 2. The strided twiddle access `T[j * stride]` with `stride = N/len`, which at
///    early stages (len=2, stride=N/2) reads elements separated by N/2 entries,
///    causing L1-cache misses for N ≥ 256.
///
/// The contiguous layout stores stage twiddles sequentially, so the butterfly
/// inner loop reads `cache_line_size / sizeof(Complex64)` twiddles per cache line
/// regardless of N. This is the same principle used by FFTW's split-radix planner.
///
/// # Bluestein scratch reuse
///
/// For non-power-of-two lengths the plan caches a `Mutex<Vec<Complex64>>`
/// scratch buffer of length `M` alongside the Bluestein plan. On each call the
/// mutex is locked, the scratch is reused for the convolution, and the lock is
/// released. This eliminates the per-call `Vec` allocation that would otherwise
/// occur at O(M log M) work boundary — the critical section overhead is O(1).
///
/// # Failure modes
///
/// - zero-length shapes are rejected by `Shape1D::new`
/// - caller-supplied buffers must match the plan length
/// - non-contiguous ndarray buffers are rejected by `expect` on slice access
pub struct FftPlan1D {
    n: usize,
    precision: PrecisionProfile,
    /// Precomputed contiguous per-stage forward twiddle table for power-of-two N.
    /// Layout: stage s occupies entries [base..base+half] where half = 2^(s-1).
    /// Total length = N-1. `None` for non-power-of-two (Bluestein handles those).
    twiddle_fwd_64: Option<Arc<[Complex64]>>,
    /// Precomputed contiguous per-stage inverse twiddle table for power-of-two N.
    twiddle_inv_64: Option<Arc<[Complex64]>>,
    /// Precomputed f32 forward twiddle table (used in LOW_PRECISION_F32 mode).
    twiddle_fwd_32: Option<Arc<[Complex32]>>,
    /// Precomputed f32 inverse twiddle table (used in LOW_PRECISION_F32 mode).
    twiddle_inv_32: Option<Arc<[Complex32]>>,
    /// Entry k = exp(-2πi·k/N) for k = 0..=N/2. Length N/2+1.
    /// `Some` iff `N` is a power of two ≥ 4; `None` otherwise.
    real_fwd_post_twiddles: Option<Vec<Complex64>>,
}

impl std::fmt::Debug for FftPlan1D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPlan1D")
            .field("n", &self.n)
            .field("precision", &self.precision)
            .finish()
    }
}

impl FftPlan1D {
    /// Create a new 1D plan.
    #[must_use]
    pub fn new(shape: Shape1D) -> Self {
        Self::with_precision(shape, PrecisionProfile::HIGH_ACCURACY_F64)
    }

    /// Create a new 1D plan with an explicit precision profile.
    #[must_use]
    pub fn with_precision(shape: Shape1D, precision: PrecisionProfile) -> Self {
        // Precompute contiguous per-stage twiddle tables for power-of-two sizes.
        // For non-power-of-two (Bluestein), these are None; Bluestein has its own tables.
        let is_pow2 = shape.n.is_power_of_two() && shape.n > 1;
        let twiddle_fwd_64 = if is_pow2 {
            Some(cached_twiddle_fwd_64(shape.n))
        } else {
            None
        };
        let twiddle_inv_64 = if is_pow2 {
            Some(cached_twiddle_inv_64(shape.n))
        } else {
            None
        };
        let twiddle_fwd_32 = if is_pow2 {
            Some(cached_twiddle_fwd_32(shape.n))
        } else {
            None
        };
        let twiddle_inv_32 = if is_pow2 {
            Some(cached_twiddle_inv_32(shape.n))
        } else {
            None
        };
        Self {
            n: shape.n,
            precision,
            twiddle_fwd_64,
            twiddle_inv_64,
            twiddle_fwd_32,
            twiddle_inv_32,
            real_fwd_post_twiddles: if shape.n >= 4 && shape.n.is_power_of_two() {
                Some(<f64 as RealFft>::build_real_fwd_post_twiddles(shape.n))
            } else {
                None
            },
        }
    }

    /// Return the plan length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Return the validated shape owned by this plan.
    #[must_use]
    pub fn shape(&self) -> Shape1D {
        Shape1D { n: self.n }
    }

    /// Return the precision profile used by this plan.
    #[must_use]
    pub fn precision_profile(&self) -> PrecisionProfile {
        self.precision
    }

    /// Whether the plan length is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward transform of a real signal.
    #[must_use]
    pub fn forward(&self, input: &Array1<f64>) -> Array1<Complex64> {
        self.forward_real_to_complex(input)
    }

    /// Inverse transform returning a real signal.
    #[must_use]
    pub fn inverse(&self, input: &Array1<Complex64>) -> Array1<f64> {
        self.inverse_complex_to_real(input)
    }

    /// Forward transform of a real signal using generic storage dispatch.
    #[must_use]
    pub fn forward_typed<T: RealFftData>(&self, input: &Array1<T>) -> Array1<T::Spectrum> {
        T::forward_1d(self, input)
    }

    /// Inverse transform of a complex spectrum using generic storage dispatch.
    #[must_use]
    pub fn inverse_typed<T: RealFftData>(&self, input: &Array1<T::Spectrum>) -> Array1<T> {
        T::inverse_1d(self, input)
    }

    /// Forward transform of a real signal.
    #[must_use]
    pub fn forward_real_to_complex(&self, input: &Array1<f64>) -> Array1<Complex64> {
        assert_eq!(input.len(), self.n, "forward input length mismatch");
        // For power-of-two N ≥ 4: use the half-length complex packing trick.
        // Pack x into N/2 complex samples, run N/2-point FFT (using the first
        // N/2-1 entries of the N-point twiddle table), then unpack via conjugate
        // symmetry. This halves the FFT work relative to a full N-point complex FFT.
        if let (Some(fft_tw), Some(post_tw)) = (&self.twiddle_fwd_64, &self.real_fwd_post_twiddles)
        {
            let input_slice = input.as_slice().expect("input must be contiguous");
            let mut output = Array1::<Complex64>::from_shape_vec(self.n, uninit_copy_vec(self.n))
                .expect("uninit Complex64 1D buffer length must match plan n");
            <f64 as RealFft>::forward_real_inplace(
                input_slice,
                output.as_slice_mut().expect("output must be contiguous"),
                fft_tw.as_ref(),
                post_tw,
            );
            output
        } else {
            let mut output = Array1::<Complex64>::from_shape_vec(self.n, uninit_copy_vec(self.n))
                .expect("uninit Complex64 1D buffer length must match plan n");
            ndarray::Zip::from(&mut output)
                .and(input)
                .for_each(|out, &value| {
                    *out = Complex64::new(value, 0.0);
                });
            self.forward_complex_inplace(&mut output);
            output
        }
    }

    /// Forward transform of a real signal into a caller-supplied complex buffer.
    pub fn forward_real_to_complex_into(
        &self,
        input: &Array1<f64>,
        output: &mut Array1<Complex64>,
    ) {
        self.forward_real_to_complex_slice_into(
            input.as_slice().expect("input must be contiguous"),
            output.as_slice_mut().expect("output must be contiguous"),
        );
    }

    /// Forward transform of a real signal slice into caller-owned complex storage.
    pub fn forward_real_to_complex_slice_into(&self, input: &[f64], output: &mut [Complex64]) {
        assert_eq!(input.len(), self.n, "forward input length mismatch");
        assert_eq!(output.len(), self.n, "forward output length mismatch");
        if let (Some(fft_tw), Some(post_tw)) = (&self.twiddle_fwd_64, &self.real_fwd_post_twiddles)
        {
            <f64 as RealFft>::forward_real_inplace(input, output, fft_tw.as_ref(), post_tw);
        } else {
            for (out, &value) in output.iter_mut().zip(input.iter()) {
                *out = Complex64::new(value, 0.0);
            }
            self.forward_complex_slice_inplace(output);
        }
    }

    /// Inverse transform returning a real-valued signal.
    #[must_use]
    pub fn inverse_complex_to_real(&self, input: &Array1<Complex64>) -> Array1<f64> {
        let mut output = Array1::<f64>::from_shape_vec(self.n, uninit_copy_vec(self.n))
            .expect("uninit f64 1D buffer length must match plan n");
        self.inverse_complex_to_real_with_workspace(input, &mut output);
        output
    }

    /// Inverse transform into caller-owned real and scratch buffers.
    pub fn inverse_complex_to_real_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<f64>,
        scratch: &mut Array1<Complex64>,
    ) {
        assert_eq!(input.len(), self.n, "inverse input length mismatch");
        assert_eq!(output.len(), self.n, "inverse output length mismatch");
        assert_eq!(scratch.len(), self.n, "inverse scratch length mismatch");
        // Fast path: iRFFT trick. The caller-supplied scratch is N elements but
        // we only need M=N/2; use the lower half to avoid a lock when the caller
        // already owns a buffer.
        if let (Some(inv_tw), Some(post_tw)) = (&self.twiddle_inv_64, &self.real_fwd_post_twiddles)
        {
            let input_slice = input.as_slice().expect("input must be contiguous");
            let output_slice = output.as_slice_mut().expect("output must be contiguous");
            let scratch_slice = scratch.as_slice_mut().expect("scratch must be contiguous");
            let m = self.n >> 1;
            <f64 as RealFft>::inverse_real_inplace(
                input_slice,
                output_slice,
                &mut scratch_slice[..m],
                inv_tw.as_ref(),
                post_tw,
            );
        } else {
            scratch.assign(input);
            self.inverse_complex_inplace(scratch);
            Zip::from(output)
                .and(scratch.view())
                .for_each(|out, value| {
                    *out = value.re;
                });
        }
    }

    fn inverse_complex_to_real_with_workspace(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<f64>,
    ) {
        assert_eq!(input.len(), self.n, "inverse input length mismatch");
        assert_eq!(output.len(), self.n, "inverse output length mismatch");
        // Fast path: iRFFT half-spectrum trick for PoT N ≥ 4.
        // Uses the M-point inverse FFT (M = N/2) instead of full N-point,
        // halving the FFT work and eliminating the per-call N-element allocation.
        if let (Some(inv_tw), Some(post_tw)) = (&self.twiddle_inv_64, &self.real_fwd_post_twiddles)
        {
            let input_slice = input.as_slice().expect("input must be contiguous");
            let output_slice = output.as_slice_mut().expect("output must be contiguous");
            let m = self.n >> 1;
            with_stockham_scratch_64(m, |scratch| {
                <f64 as RealFft>::inverse_real_inplace(
                    input_slice,
                    output_slice,
                    scratch,
                    inv_tw.as_ref(),
                    post_tw,
                );
            });
        } else {
            let mut spectrum = input.to_owned();
            self.inverse_complex_inplace(&mut spectrum);
            Zip::from(output)
                .and(spectrum.view())
                .for_each(|out, value| {
                    *out = value.re;
                });
        }
    }

    /// Forward transform of a complex signal in-place using a slice.
    pub fn forward_complex_slice_inplace(&self, data: &mut [Complex64]) {
        assert_eq!(data.len(), self.n, "complex forward length mismatch");
        if let Some(twiddles) = &self.twiddle_fwd_64 {
            dispatch_inplace::<f64, false, false>(data, Some(twiddles.as_ref()));
        } else {
            crate::application::execution::kernel::mixed_radix::forward_inplace::<f64>(data);
        }
    }

    /// Inverse transform of a complex signal in-place with normalization using a slice.
    pub fn inverse_complex_slice_inplace(&self, data: &mut [Complex64]) {
        assert_eq!(data.len(), self.n, "complex inverse length mismatch");
        if let Some(twiddles) = &self.twiddle_inv_64 {
            dispatch_inplace::<f64, true, true>(data, Some(twiddles.as_ref()));
        } else {
            crate::application::execution::kernel::mixed_radix::inverse_inplace::<f64>(data);
        }
    }

    /// Forward transform of a complex signal in-place.
    pub fn forward_complex_inplace(&self, data: &mut Array1<Complex64>) {
        self.forward_complex_slice_inplace(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Inverse transform of a complex signal in-place with normalization.
    pub fn inverse_complex_inplace(&self, data: &mut Array1<Complex64>) {
        self.inverse_complex_slice_inplace(data.as_slice_mut().expect("Array must be contiguous"));
    }

    /// Forward transform of a complex signal (allocating).
    #[must_use]
    pub fn forward_complex(&self, input: &Array1<Complex64>) -> Array1<Complex64> {
        let mut output = input.clone();
        self.forward_complex_inplace(&mut output);
        output
    }

    /// Inverse transform of a complex signal (allocating).
    #[must_use]
    pub fn inverse_complex(&self, input: &Array1<Complex64>) -> Array1<Complex64> {
        let mut output = input.clone();
        self.inverse_complex_inplace(&mut output);
        output
    }
}

#[cfg(test)]
mod tests;
