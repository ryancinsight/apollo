//! Reusable Chirp Z-Transform Plan

use super::helpers::{leto_array1_from_vec, leto_view1_cow, with_forward_workspace};
use super::typed::CztStorage;
use crate::application::execution::kernel::bluestein::{
    czt_bjork_pereyra_inverse_into, czt_bluestein_forward_into_with_workspace, czt_inverse_nodes,
};
use crate::application::execution::kernel::direct::{
    czt_direct_forward, czt_direct_forward_slice_into,
};
use crate::domain::contracts::error::CztError;
use apollo_fft::{FftPlan1D, PrecisionProfile, Shape1D};
use leto::Array1;
use eunomia::Complex64;

/// Reusable chirp z-transform plan.
///
/// The plan is the single source of truth for CZT dimensions, spiral
/// parameters, chirp factors, convolution kernel, and FFT plan. Construction
/// validates non-zero lengths, finite non-zero `a` and `w`, and precomputes
/// all input-independent terms.
///
/// # Mathematical contract
///
/// For input `x[0..n)` this plan computes
/// `X[k] = sum_n x[n] a^-n w^(n k)` for `k in 0..m`.
/// `forward_direct` evaluates that definition directly. `forward` and
/// `forward_into` evaluate the same map through Bluestein convolution.
///
/// # Complexity
///
/// `forward_direct` costs `O(nm)` time. `forward` and `forward_into` cost
/// `O(p log p)` time with one `O(p)` convolution workspace, where
/// `p = convolution_len() >= n + m - 1`.
#[derive(Debug)]
pub struct CztPlan {
    n: usize,
    m: usize,
    a: Complex64,
    w: Complex64,
    convolution_len: usize,
    chirp_n: Vec<Complex64>,
    chirp_k: Vec<Complex64>,
    fft_kernel: Array1<Complex64>,
    fft_plan: FftPlan1D<f64>,
    inverse_nodes: Option<Vec<Complex64>>,
}

impl CztPlan {
    /// Create a validated CZT plan precomputing and caching the $O(P \log P)$ convolution kernel.
    pub fn new(n: usize, m: usize, a: Complex64, w: Complex64) -> Result<Self, CztError> {
        if n == 0 || m == 0 {
            return Err(CztError::EmptyLength);
        }
        if !a.re.is_finite() || !a.im.is_finite() || !w.re.is_finite() || !w.im.is_finite() {
            return Err(CztError::InvalidParameters);
        }
        if a == Complex64::new(0.0, 0.0) || w == Complex64::new(0.0, 0.0) {
            return Err(CztError::InvalidParameters);
        }

        let convolution_len = (n + m - 1).next_power_of_two();
        let fft_plan = FftPlan1D::<f64>::new(
            Shape1D::new(convolution_len).expect("CZT convolution length must be valid"),
        );

        let mut chirp_n = Vec::with_capacity(n);
        let mut chirp_k = Vec::with_capacity(m);
        let mut kernel = vec![Complex64::new(0.0, 0.0); convolution_len];

        for n_idx in 0..n {
            let nn = n_idx as f64;
            let phase = 0.5 * nn * nn;
            chirp_n.push((a.powf(-nn)) * w.powf(phase));
        }

        for k_idx in 0..m {
            let kk = k_idx as f64;
            let phase = 0.5 * kk * kk;
            chirp_k.push(w.powf(phase));
        }

        for k_idx in 0..m {
            kernel[k_idx] = w.powf(-0.5 * (k_idx as f64) * (k_idx as f64));
        }
        for k_idx in 1..n {
            kernel[convolution_len - k_idx] = w.powf(-0.5 * (k_idx as f64) * (k_idx as f64));
        }

        let mut fft_kernel = Array1::from(kernel);
        fft_plan.forward_complex_inplace(&mut fft_kernel);
        let inverse_nodes = if n == m {
            Some(czt_inverse_nodes(n, w))
        } else {
            None
        };

        Ok(Self {
            n,
            m,
            a,
            w,
            convolution_len,
            chirp_n,
            chirp_k,
            fft_kernel,
            fft_plan,
            inverse_nodes,
        })
    }

    /// Return the input length.
    #[must_use]
    pub const fn input_len(&self) -> usize {
        self.n
    }

    /// Return the output length.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.m
    }

    /// Return the convolution length used by the fast path.
    #[must_use]
    pub const fn convolution_len(&self) -> usize {
        self.convolution_len
    }

    /// Forward direct CZT evaluation.
    pub fn forward_direct(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, CztError> {
        if input.size() != self.n {
            return Err(CztError::LengthMismatch);
        }
        czt_direct_forward(input, self.m, self.a, self.w)
    }

    /// Forward direct CZT evaluation over a Leto complex view.
    ///
    /// Contiguous views are borrowed. Strided views copy once into logical
    /// order before entering the canonical direct slice kernel.
    pub fn forward_direct_leto(
        &self,
        input: leto::ArrayView1<'_, Complex64>,
    ) -> Result<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>, CztError> {
        if input.shape()[0] != self.n {
            return Err(CztError::LengthMismatch);
        }
        let signal = leto_view1_cow(&input);
        let mut output = vec![Complex64::new(0.0, 0.0); self.output_len()];
        czt_direct_forward_slice_into(&signal, &mut output, self.a, self.w)?;
        Ok(leto_array1_from_vec(output))
    }

    /// Forward CZT using Bluestein's convolution identity with precomputed caching.
    pub fn forward(&self, input: &Array1<Complex64>) -> Result<Array1<Complex64>, CztError> {
        if input.size() != self.n {
            return Err(CztError::LengthMismatch);
        }

        let mut output = Array1::<Complex64>::zeros([self.m]);
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Forward CZT over a Leto complex view.
    ///
    /// Contiguous views are borrowed. Strided views copy once into logical
    /// order before entering the canonical contiguous slice kernel.
    pub fn forward_leto(
        &self,
        input: leto::ArrayView1<'_, Complex64>,
    ) -> Result<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>, CztError> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![Complex64::new(0.0, 0.0); self.output_len()];
        self.forward_complex64_slice_into(&signal, &mut output)?;
        Ok(leto_array1_from_vec(output))
    }

    /// Forward CZT into caller-owned output storage.
    pub fn forward_into(
        &self,
        input: &Array1<Complex64>,
        output: &mut Array1<Complex64>,
    ) -> Result<(), CztError> {
        if input.size() != self.n || output.size() != self.m {
            return Err(CztError::LengthMismatch);
        }
        self.forward_complex64_slice_into(
            input.as_slice().expect("CZT input must be contiguous"),
            output
                .as_slice_mut()
                .expect("CZT output must be contiguous"),
        )
    }

    /// Forward CZT over contiguous Complex64 slices.
    pub(crate) fn forward_complex64_slice_into(
        &self,
        input: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), CztError> {
        if input.len() != self.n || output.len() != self.m {
            return Err(CztError::LengthMismatch);
        }
        with_forward_workspace(self.convolution_len, |workspace| {
            czt_bluestein_forward_into_with_workspace(
                input,
                output,
                workspace,
                self.convolution_len,
                &self.chirp_n,
                &self.chirp_k,
                self.fft_kernel
                    .as_slice()
                    .expect("CZT FFT kernel must be contiguous"),
                &self.fft_plan,
            );
        });
        Ok(())
    }

    /// Forward CZT for `Complex64`, `Complex32`, or mixed two-lane `f16` storage.
    ///
    /// `Complex64` uses the native high-accuracy path. `Complex32` and mixed
    /// `[f16; 2]` storage convert through the owner kernel and quantize once
    /// into the caller-owned output.
    pub fn forward_typed_into<T: CztStorage>(
        &self,
        input: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        T::forward_into(self, input, output, profile)
    }

    /// Forward CZT over a typed Leto complex-storage view.
    pub fn forward_leto_typed<T: CztStorage>(
        &self,
        input: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> Result<leto::Array<T, leto::MnemosyneStorage<T>, 1>, CztError> {
        let signal = leto_view1_cow(&input);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); self.output_len()];
        T::forward_slice_into(self, &signal, &mut output, profile)?;
        Ok(leto_array1_from_vec(output))
    }

    /// In-place forward CZT.
    pub fn forward_inplace(&self, data: &mut Array1<Complex64>) -> Result<(), CztError> {
        let transformed = self.forward(data)?;
        *data = transformed;
        Ok(())
    }

    /// Inverse chirp z-transform via the Björck–Pereyra Vandermonde solve.
    ///
    /// # Mathematical contract
    ///
    /// Given a spectrum `X[k]` of length `M`, recovers `x[n]` of length `N`
    /// such that `forward(x) == X` (in exact arithmetic).  Inversion requires
    /// `M == N`; a rectangular (M ≠ N) CZT is not uniquely invertible.
    ///
    /// The evaluation points are `z_k = W^k` for `k = 0..N-1`.  If two points
    /// coincide the Vandermonde matrix is singular and `CztError::NotInvertible`
    /// is returned.
    ///
    /// # Complexity
    ///
    /// O(N²) time, O(N) additional space.
    pub fn inverse(&self, spectrum: &Array1<Complex64>) -> Result<Array1<Complex64>, CztError> {
        if spectrum.size() != self.m {
            return Err(CztError::LengthMismatch);
        }
        if self.n != self.m {
            return Err(CztError::NotInvertible {
                reason: "inverse is only defined for square (M == N) CZT plans",
            });
        }
        let mut output = Array1::<Complex64>::zeros([self.n]);
        self.inverse_complex64_slice_into(
            spectrum
                .as_slice()
                .expect("CZT spectrum must be contiguous"),
            output
                .as_slice_mut()
                .expect("CZT inverse output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Inverse CZT over a Leto complex spectrum view.
    ///
    /// Inversion is available only for square plans (`M == N`), matching the
    /// existing Leto inverse contract.
    pub fn inverse_leto(
        &self,
        spectrum: leto::ArrayView1<'_, Complex64>,
    ) -> Result<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>, CztError> {
        let spectrum = leto_view1_cow(&spectrum);
        let mut output = vec![Complex64::new(0.0, 0.0); self.input_len()];
        self.inverse_complex64_slice_into(&spectrum, &mut output)?;
        Ok(leto_array1_from_vec(output))
    }

    /// Inverse CZT over contiguous Complex64 slices.
    pub(crate) fn inverse_complex64_slice_into(
        &self,
        spectrum: &[Complex64],
        output: &mut [Complex64],
    ) -> Result<(), CztError> {
        if spectrum.len() != self.m || output.len() != self.n {
            return Err(CztError::LengthMismatch);
        }
        let nodes = self
            .inverse_nodes
            .as_deref()
            .ok_or(CztError::NotInvertible {
                reason: "inverse is only defined for square (M == N) CZT plans",
            })?;
        czt_bjork_pereyra_inverse_into(spectrum, output, nodes, self.a)
    }

    /// Inverse CZT for typed `Complex64`, `Complex32`, or mixed `[f16; 2]` storage.
    ///
    /// Converts input once to `Complex64`, applies the exact Vandermonde solve,
    /// and quantizes back to the requested storage type.
    pub fn inverse_typed_into<T: CztStorage>(
        &self,
        spectrum: &Array1<T>,
        output: &mut Array1<T>,
        profile: PrecisionProfile,
    ) -> Result<(), CztError> {
        T::inverse_into(self, spectrum, output, profile)
    }

    /// Inverse CZT over a typed Leto complex-storage spectrum view.
    pub fn inverse_leto_typed<T: CztStorage>(
        &self,
        spectrum: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> Result<leto::Array<T, leto::MnemosyneStorage<T>, 1>, CztError> {
        let spectrum = leto_view1_cow(&spectrum);
        let mut output = vec![T::from_complex64(Complex64::new(0.0, 0.0)); self.input_len()];
        T::inverse_slice_into(self, &spectrum, &mut output, profile)?;
        Ok(leto_array1_from_vec(output))
    }
}
