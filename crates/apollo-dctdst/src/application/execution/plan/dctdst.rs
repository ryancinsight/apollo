//! Reusable DCT/DST plan metadata surface.

use crate::domain::contracts::error::{DctDstError, DctDstResult};
use crate::domain::metadata::kind::{RealTransformConfig, RealTransformKind};
use crate::infrastructure::kernel::direct::{dct1, dct2, dct3, dct4, dst1, dst2, dst3, dst4};
use crate::infrastructure::kernel::fast::{
    dct2_fast, dct3_fast, dst2_fast, dst3_fast, FAST_THRESHOLD,
};
use apollo_fft::{f16, PrecisionProfile};
use ndarray::{Array2, Array3};
use std::borrow::Cow;

/// Reusable DCT/DST plan.
///
/// The plan owns a validated real-to-real transform length and kind.
///
/// # Theorem
///
/// The DCT-II/DCT-III pair and DST-II/DST-III pair are biorthogonal under the
/// unnormalized conventions implemented in this crate:
///
/// ```text
/// DCT-III(DCT-II(x)) = (N / 2) x
/// DST-III(DST-II(x)) = (N / 2) x
/// ```
///
/// DCT-I, DCT-IV, DST-I, and DST-IV are each self-inverse under the following scales:
///
/// ```text
/// DCT-I(DCT-I(x))   = 2(N−1) · x    (N ≥ 2)
/// DCT-IV(DCT-IV(x)) = (N/2)  · x
/// DST-I(DST-I(x))   = 2(N+1) · x
/// DST-IV(DST-IV(x)) = (N/2)  · x
/// ```
///
/// Therefore `inverse` scales by `2 / N` for all type-II/III/IV pairs and by
/// `1 / (2(N−1))` or `1 / (2(N+1))` for DCT-I and DST-I respectively.
///
/// # Proof sketch
///
/// The cosine and sine basis functions used by the type-II/type-III pairs are
/// orthogonal over the half-sample shifted grid. The cross terms vanish by
/// finite trigonometric sum identities, and the diagonal terms evaluate to
/// `N / 2` under Apollo's unnormalized convention. DCT-I and DST-I carry an
/// explicit factor of 2 in their definitions; their orthogonality diagonals
/// evaluate to `(N−1)` and `(N+1)` respectively, yielding the stated scales.
///
/// # Complexity
///
/// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
/// analytical kernel). Both paths use O(1) auxiliary storage for caller-owned
/// `*_into` paths (the fast path allocates a 2N complex buffer internally for
/// the FFT work area).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DctDstPlan {
    config: RealTransformConfig,
}

impl DctDstPlan {
    /// Create a validated DCT/DST plan.
    pub fn new(len: usize, kind: RealTransformKind) -> DctDstResult<Self> {
        Ok(Self {
            config: RealTransformConfig::new(len, kind)?,
        })
    }

    /// Return the validated configuration.
    #[must_use]
    pub const fn config(self) -> RealTransformConfig {
        self.config
    }

    /// Return transform length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.config.len()
    }

    /// Return true when transform length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.config.is_empty()
    }

    /// Return transform kind.
    #[must_use]
    pub const fn kind(self) -> RealTransformKind {
        self.config.kind()
    }

    /// Execute the forward transform and return allocated coefficients.
    ///
    /// Returns `LengthMismatch` when the input slice length differs from the
    /// plan length.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn forward(&self, signal: &[f64]) -> DctDstResult<Vec<f64>> {
        let mut output = vec![0.0_f64; self.len()];
        self.forward_into(signal, &mut output)?;
        Ok(output)
    }

    /// Execute the forward transform over a Leto real-valued 1D view.
    ///
    /// Contiguous views are borrowed. Strided views copy once into logical
    /// order before entering the canonical slice execution path.
    pub fn forward_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>> {
        let signal = leto_view1_cow(&signal);
        let mut output = vec![0.0_f64; self.len()];
        self.forward_into(&signal, &mut output)?;
        Ok(leto_array1_from_slice(&output))
    }

    /// Execute a separable 2D forward transform over a square `N x N` field.
    ///
    /// The plan length `N` is applied to both axes; each row is transformed
    /// first, then each column.
    ///
    /// Returns `LengthMismatch` unless `input.dim() == (N, N)`.
    pub fn forward_2d(&self, input: &Array2<f64>) -> DctDstResult<Array2<f64>> {
        let (rows, cols) = input.dim();
        if rows != self.len() || cols != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let mut output = Array2::<f64>::zeros((rows, cols));
        self.forward_2d_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute a separable 2D forward transform over a Leto `N x N` view.
    pub fn forward_2d_leto(
        &self,
        input: leto::ArrayView2<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let input = array2_from_leto_view(input);
        let output = self.forward_2d(&input)?;
        Ok(leto_array2_from_ndarray(&output))
    }

    /// Execute a separable 2D forward transform into caller-owned output.
    ///
    /// Returns `LengthMismatch` unless both `input` and `output` are square
    /// `N x N` arrays matching the plan length `N`.
    pub fn forward_2d_into(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
    ) -> DctDstResult<()> {
        let n = self.len();
        if input.dim() != (n, n) || output.dim() != (n, n) {
            return Err(DctDstError::LengthMismatch);
        }

        let mut stage = Array2::<f64>::zeros((n, n));
        let mut line_in = vec![0.0_f64; n];
        let mut line_out = vec![0.0_f64; n];

        for i in 0..n {
            for j in 0..n {
                line_in[j] = input[(i, j)];
            }
            self.forward_into(&line_in, &mut line_out)?;
            for j in 0..n {
                stage[(i, j)] = line_out[j];
            }
        }

        for j in 0..n {
            for i in 0..n {
                line_in[i] = stage[(i, j)];
            }
            self.forward_into(&line_in, &mut line_out)?;
            for i in 0..n {
                output[(i, j)] = line_out[i];
            }
        }

        Ok(())
    }

    /// Execute a separable 3D forward transform over a cubic `N x N x N` field.
    ///
    /// The plan length `N` is applied to all three axes in z, then y, then x
    /// order.
    ///
    /// Returns `LengthMismatch` unless `input.dim() == (N, N, N)`.
    pub fn forward_3d(&self, input: &Array3<f64>) -> DctDstResult<Array3<f64>> {
        let dims = input.dim();
        if dims.0 != self.len() || dims.1 != self.len() || dims.2 != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let mut output = Array3::<f64>::zeros(dims);
        self.forward_3d_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute a separable 3D forward transform over a Leto `N x N x N` view.
    pub fn forward_3d_leto(
        &self,
        input: leto::ArrayView3<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let input = array3_from_leto_view(input);
        let output = self.forward_3d(&input)?;
        Ok(leto_array3_from_ndarray(&output))
    }

    /// Execute a separable 3D forward transform into caller-owned output.
    ///
    /// Returns `LengthMismatch` unless both `input` and `output` are cubic
    /// `N x N x N` arrays matching the plan length `N`.
    pub fn forward_3d_into(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
    ) -> DctDstResult<()> {
        let n = self.len();
        if input.dim() != (n, n, n) || output.dim() != (n, n, n) {
            return Err(DctDstError::LengthMismatch);
        }

        let mut stage1 = Array3::<f64>::zeros((n, n, n));
        let mut stage2 = Array3::<f64>::zeros((n, n, n));
        let mut line_in = vec![0.0_f64; n];
        let mut line_out = vec![0.0_f64; n];

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    line_in[k] = input[(i, j, k)];
                }
                self.forward_into(&line_in, &mut line_out)?;
                for k in 0..n {
                    stage1[(i, j, k)] = line_out[k];
                }
            }
        }

        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    line_in[j] = stage1[(i, j, k)];
                }
                self.forward_into(&line_in, &mut line_out)?;
                for j in 0..n {
                    stage2[(i, j, k)] = line_out[j];
                }
            }
        }

        for j in 0..n {
            for k in 0..n {
                for i in 0..n {
                    line_in[i] = stage2[(i, j, k)];
                }
                self.forward_into(&line_in, &mut line_out)?;
                for i in 0..n {
                    output[(i, j, k)] = line_out[i];
                }
            }
        }

        Ok(())
    }

    /// Execute the forward transform into a caller-supplied buffer.
    ///
    /// Returns `LengthMismatch` when either slice length differs from the plan
    /// length.
    ///
    /// Dispatches to the O(N log N) FFT fast path for N ≥ 16, and to the
    /// direct O(N²) analytical kernel for N < 16.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn forward_into(&self, signal: &[f64], output: &mut [f64]) -> DctDstResult<()> {
        if signal.len() != self.len() || output.len() != self.len() {
            return Err(DctDstError::LengthMismatch);
        }

        let n = self.len();
        if n >= FAST_THRESHOLD {
            match self.kind() {
                RealTransformKind::DctII => dct2_fast(signal, output),
                RealTransformKind::DctIII => dct3_fast(signal, output),
                RealTransformKind::DstII => dst2_fast(signal, output),
                RealTransformKind::DstIII => dst3_fast(signal, output),
                // No fast path implemented yet; fall through to direct O(N²) kernel.
                RealTransformKind::DctI => dct1(signal, output),
                RealTransformKind::DctIV => dct4(signal, output),
                RealTransformKind::DstI => dst1(signal, output),
                RealTransformKind::DstIV => dst4(signal, output),
            }
        } else {
            match self.kind() {
                RealTransformKind::DctII => dct2(signal, output),
                RealTransformKind::DctIII => dct3(signal, output),
                RealTransformKind::DstII => dst2(signal, output),
                RealTransformKind::DstIII => dst3(signal, output),
                RealTransformKind::DctI => dct1(signal, output),
                RealTransformKind::DctIV => dct4(signal, output),
                RealTransformKind::DstI => dst1(signal, output),
                RealTransformKind::DstIV => dst4(signal, output),
            }
        }

        Ok(())
    }

    /// Compute the inverse of the given forward transform.
    ///
    /// DCT-III is the inverse of DCT-II (up to a 2/N scaling factor).
    /// DST-III is the inverse of DST-II (up to a 2/N scaling factor).
    /// DCT-I, DCT-IV, DST-I, and DST-IV are self-inverse; each is scaled by
    /// `1/(2(N−1))`, `2/N`, `1/(2(N+1))`, and `2/N` respectively.
    /// The result is scaled to recover the original signal.
    ///
    /// Returns `LengthMismatch` when the input slice length differs from the
    /// plan length.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn inverse(&self, signal: &[f64]) -> DctDstResult<Vec<f64>> {
        let mut output = vec![0.0_f64; self.len()];
        self.inverse_into(signal, &mut output)?;
        Ok(output)
    }

    /// Execute the inverse transform over a Leto real-valued 1D view.
    pub fn inverse_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>> {
        let signal = leto_view1_cow(&signal);
        let mut output = vec![0.0_f64; self.len()];
        self.inverse_into(&signal, &mut output)?;
        Ok(leto_array1_from_slice(&output))
    }

    /// Execute a separable 2D inverse transform over a square `N x N` field.
    ///
    /// Returns `LengthMismatch` unless `input.dim() == (N, N)`.
    pub fn inverse_2d(&self, input: &Array2<f64>) -> DctDstResult<Array2<f64>> {
        let (rows, cols) = input.dim();
        if rows != self.len() || cols != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let mut output = Array2::<f64>::zeros((rows, cols));
        self.inverse_2d_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute a separable 2D inverse transform over a Leto `N x N` view.
    pub fn inverse_2d_leto(
        &self,
        input: leto::ArrayView2<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let input = array2_from_leto_view(input);
        let output = self.inverse_2d(&input)?;
        Ok(leto_array2_from_ndarray(&output))
    }

    /// Execute a separable 2D inverse transform into caller-owned output.
    ///
    /// Returns `LengthMismatch` unless both `input` and `output` are square
    /// `N x N` arrays matching the plan length `N`.
    pub fn inverse_2d_into(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
    ) -> DctDstResult<()> {
        let n = self.len();
        if input.dim() != (n, n) || output.dim() != (n, n) {
            return Err(DctDstError::LengthMismatch);
        }

        let mut stage = Array2::<f64>::zeros((n, n));
        let mut line_in = vec![0.0_f64; n];
        let mut line_out = vec![0.0_f64; n];

        for i in 0..n {
            for j in 0..n {
                line_in[j] = input[(i, j)];
            }
            self.inverse_into(&line_in, &mut line_out)?;
            for j in 0..n {
                stage[(i, j)] = line_out[j];
            }
        }

        for j in 0..n {
            for i in 0..n {
                line_in[i] = stage[(i, j)];
            }
            self.inverse_into(&line_in, &mut line_out)?;
            for i in 0..n {
                output[(i, j)] = line_out[i];
            }
        }

        Ok(())
    }

    /// Execute a separable 3D inverse transform over a cubic `N x N x N` field.
    ///
    /// Returns `LengthMismatch` unless `input.dim() == (N, N, N)`.
    pub fn inverse_3d(&self, input: &Array3<f64>) -> DctDstResult<Array3<f64>> {
        let dims = input.dim();
        if dims.0 != self.len() || dims.1 != self.len() || dims.2 != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let mut output = Array3::<f64>::zeros(dims);
        self.inverse_3d_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute a separable 3D inverse transform over a Leto `N x N x N` view.
    pub fn inverse_3d_leto(
        &self,
        input: leto::ArrayView3<'_, f64>,
    ) -> DctDstResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let input = array3_from_leto_view(input);
        let output = self.inverse_3d(&input)?;
        Ok(leto_array3_from_ndarray(&output))
    }

    /// Execute a separable 3D inverse transform into caller-owned output.
    ///
    /// Returns `LengthMismatch` unless both `input` and `output` are cubic
    /// `N x N x N` arrays matching the plan length `N`.
    pub fn inverse_3d_into(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<f64>,
    ) -> DctDstResult<()> {
        let n = self.len();
        if input.dim() != (n, n, n) || output.dim() != (n, n, n) {
            return Err(DctDstError::LengthMismatch);
        }

        let mut stage1 = Array3::<f64>::zeros((n, n, n));
        let mut stage2 = Array3::<f64>::zeros((n, n, n));
        let mut line_in = vec![0.0_f64; n];
        let mut line_out = vec![0.0_f64; n];

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    line_in[k] = input[(i, j, k)];
                }
                self.inverse_into(&line_in, &mut line_out)?;
                for k in 0..n {
                    stage1[(i, j, k)] = line_out[k];
                }
            }
        }

        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    line_in[j] = stage1[(i, j, k)];
                }
                self.inverse_into(&line_in, &mut line_out)?;
                for j in 0..n {
                    stage2[(i, j, k)] = line_out[j];
                }
            }
        }

        for j in 0..n {
            for k in 0..n {
                for i in 0..n {
                    line_in[i] = stage2[(i, j, k)];
                }
                self.inverse_into(&line_in, &mut line_out)?;
                for i in 0..n {
                    output[(i, j, k)] = line_out[i];
                }
            }
        }

        Ok(())
    }

    /// Compute the inverse of the configured transform into caller-owned output.
    ///
    /// DCT-III is the inverse of DCT-II (up to a 2/N scaling factor).
    /// DST-III is the inverse of DST-II (up to a 2/N scaling factor).
    /// DCT-I, DCT-IV, DST-I, and DST-IV are self-inverse; each is scaled by
    /// `1/(2(N−1))`, `2/N`, `1/(2(N+1))`, and `2/N` respectively.
    /// The result is scaled to recover the original signal.
    ///
    /// Returns `LengthMismatch` when either slice length differs from the plan
    /// length.
    ///
    /// Dispatches to the O(N log N) FFT fast path for N ≥ 16, and to the
    /// direct O(N²) analytical kernel for N < 16.
    ///
    /// # Complexity
    ///
    /// O(N log N) for N ≥ 16 (2N-point FFT fast path); O(N²) for N < 16 (direct
    /// analytical kernel).
    pub fn inverse_into(&self, signal: &[f64], output: &mut [f64]) -> DctDstResult<()> {
        if signal.len() != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        if output.len() != self.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let n = self.len();
        let mut raw = vec![0.0_f64; n];

        if n >= FAST_THRESHOLD {
            match self.kind() {
                RealTransformKind::DctII => dct3_fast(signal, &mut raw),
                RealTransformKind::DstII => dst3_fast(signal, &mut raw),
                RealTransformKind::DctIII => dct2_fast(signal, &mut raw),
                RealTransformKind::DstIII => dst2_fast(signal, &mut raw),
                // No fast path implemented yet; fall through to direct O(N²) kernel.
                RealTransformKind::DctI => dct1(signal, &mut raw),
                RealTransformKind::DctIV => dct4(signal, &mut raw),
                RealTransformKind::DstI => dst1(signal, &mut raw),
                RealTransformKind::DstIV => dst4(signal, &mut raw),
            }
        } else {
            match self.kind() {
                RealTransformKind::DctII => dct3(signal, &mut raw),
                RealTransformKind::DstII => dst3(signal, &mut raw),
                RealTransformKind::DctIII => dct2(signal, &mut raw),
                RealTransformKind::DstIII => dst2(signal, &mut raw),
                RealTransformKind::DctI => dct1(signal, &mut raw),
                RealTransformKind::DctIV => dct4(signal, &mut raw),
                RealTransformKind::DstI => dst1(signal, &mut raw),
                RealTransformKind::DstIV => dst4(signal, &mut raw),
            }
        }

        // Scale factor derived from the self-inverse identity of each transform kind:
        //   DCT-II/III, DST-II/III, DCT-IV, DST-IV: paired/self-inverse scale = 2/N
        //   DCT-I: C₁·C₁ = 2(N−1)·I  →  scale = 1/(2(N−1))
        //   DST-I: S₁·S₁ = 2(N+1)·I  →  scale = 1/(2(N+1))
        let scale = match self.kind() {
            RealTransformKind::DctI => 1.0 / (2.0 * (n - 1) as f64),
            RealTransformKind::DstI => 1.0 / (2.0 * (n + 1) as f64),
            _ => 2.0 / n as f64,
        };
        for (slot, value) in output.iter_mut().zip(raw.into_iter()) {
            *slot = value * scale;
        }
        Ok(())
    }

    /// Execute the forward transform for `f64`, `f32`, or mixed `f16` storage.
    ///
    /// Lower storage profiles reuse the crate's authoritative `f64` transform
    /// and quantize once into the caller-owned output slice. This avoids
    /// precision-specific algorithm forks and preserves the DCT/DST theorem
    /// surface.
    pub fn forward_typed_into<T: RealTransformStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        T::forward_into(self, signal, output, profile)
    }

    /// Execute the forward transform over a typed Leto real-valued 1D view.
    pub fn forward_leto_typed<T: RealTransformStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> DctDstResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = leto_view1_cow(&signal);
        let mut output = vec![T::from_f64(0.0); self.len()];
        T::forward_into(self, &signal, &mut output, profile)?;
        Ok(leto_array1_from_slice(&output))
    }

    /// Execute the inverse transform for `f64`, `f32`, or mixed `f16` storage.
    pub fn inverse_typed_into<T: RealTransformStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        T::inverse_into(self, signal, output, profile)
    }

    /// Execute the inverse transform over a typed Leto real-valued 1D view.
    pub fn inverse_leto_typed<T: RealTransformStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> DctDstResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = leto_view1_cow(&signal);
        let mut output = vec![T::from_f64(0.0); self.len()];
        T::inverse_into(self, &signal, &mut output, profile)?;
        Ok(leto_array1_from_slice(&output))
    }
}

/// Real storage accepted by typed DCT/DST paths.
pub trait RealTransformStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage to owner arithmetic.
    fn to_f64(self) -> f64;
    /// Convert owner arithmetic to storage.
    fn from_f64(value: f64) -> Self;

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let input64: Vec<f64> = signal.iter().map(|value| value.to_f64()).collect();
        let mut output64 = vec![0.0_f64; plan.len()];
        plan.forward_into(&input64, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.into_iter()) {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(DctDstError::LengthMismatch);
        }
        let input64: Vec<f64> = signal.iter().map(|value| value.to_f64()).collect();
        let mut output64 = vec![0.0_f64; plan.len()];
        plan.inverse_into(&input64, &mut output64)?;
        for (slot, value) in output.iter_mut().zip(output64.into_iter()) {
            *slot = Self::from_f64(value);
        }
        Ok(())
    }
}

impl RealTransformStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(signal, output)
    }

    fn inverse_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(signal, output)
    }
}

impl RealTransformStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl RealTransformStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> DctDstResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(DctDstError::PrecisionMismatch)
    }
}

fn leto_view1_cow<'a, T: Copy>(view: &leto::ArrayView1<'a, T>) -> Cow<'a, [T]> {
    match view.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(
            (0..view.shape()[0])
                .map(|index| {
                    *view
                        .get([index])
                        .expect("Leto DCT/DST view index must be valid after shape validation")
                })
                .collect(),
        ),
    }
}

fn array2_from_leto_view(input: leto::ArrayView2<'_, f64>) -> Array2<f64> {
    let [rows, cols] = input.shape();
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        *input
            .get([row, col])
            .expect("Leto 2D DCT/DST view index must be valid after shape validation")
    })
}

fn array3_from_leto_view(input: leto::ArrayView3<'_, f64>) -> Array3<f64> {
    let [d0, d1, d2] = input.shape();
    Array3::from_shape_fn((d0, d1, d2), |(i, j, k)| {
        *input
            .get([i, j, k])
            .expect("Leto 3D DCT/DST view index must be valid after shape validation")
    })
}

fn leto_array1_from_slice<T: Copy>(output: &[T]) -> leto::Array<T, leto::MnemosyneStorage<T>, 1> {
    leto::Array::<T, leto::MnemosyneStorage<T>, 1>::from_mnemosyne_slice([output.len()], output)
        .expect("DCT/DST output length must match Leto output shape")
}

fn leto_array2_from_ndarray(
    output: &Array2<f64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 2> {
    let (rows, cols) = output.dim();
    leto::Array::<f64, leto::MnemosyneStorage<f64>, 2>::from_mnemosyne_slice(
        [rows, cols],
        output
            .as_slice()
            .expect("DCT/DST-owned 2D ndarray output must be contiguous"),
    )
    .expect("DCT/DST 2D output length must match Leto output shape")
}

fn leto_array3_from_ndarray(
    output: &Array3<f64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 3> {
    let (d0, d1, d2) = output.dim();
    leto::Array::<f64, leto::MnemosyneStorage<f64>, 3>::from_mnemosyne_slice(
        [d0, d1, d2],
        output
            .as_slice()
            .expect("DCT/DST-owned 3D ndarray output must be contiguous"),
    )
    .expect("DCT/DST 3D output length must match Leto output shape")
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use leto::{SliceArg, Storage};

    #[test]
    fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
        let plan = DctDstPlan::new(8, RealTransformKind::DctII).expect("valid plan");
        let signal64 = [1.0_f64, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let expected = plan.forward(&signal64).expect("forward");

        let mut out64 = [0.0_f64; 8];
        plan.forward_typed_into(&signal64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed f64 forward");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let signal32 = signal64.map(|value| value as f32);
        let mut out32 = [0.0_f32; 8];
        plan.forward_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed f32 forward");
        for (actual, expected) in out32.iter().zip(expected.iter()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
        }

        let signal16 = signal64.map(|value| f16::from_f32(value as f32));
        let mut out16 = [f16::from_f32(0.0); 8];
        plan.forward_typed_into(
            &signal16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed mixed f16 forward");
        for (actual, expected) in out16.iter().zip(expected.iter()) {
            assert!((f64::from(actual.to_f32()) - *expected).abs() < 2.0e-3);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = DctDstPlan::new(4, RealTransformKind::DstII).expect("valid plan");
        let signal = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        assert!(matches!(
            plan.forward_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(DctDstError::PrecisionMismatch)
        ));
    }

    #[test]
    fn leto_1d_forward_and_inverse_match_slice_reference() {
        let plan = DctDstPlan::new(8, RealTransformKind::DctII).expect("valid plan");
        let signal = vec![1.0_f64, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let expected_forward = plan.forward(&signal).expect("slice forward");
        let expected_inverse = plan.inverse(&expected_forward).expect("slice inverse");
        let leto_signal = leto::Array1::from_shape_vec([signal.len()], signal).expect("leto input");

        let actual_forward = plan.forward_leto(leto_signal.view()).expect("leto forward");
        let actual_inverse = plan
            .inverse_leto(actual_forward.view())
            .expect("leto inverse");

        for (actual, expected) in actual_forward
            .storage()
            .as_slice()
            .iter()
            .zip(expected_forward.iter())
        {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
        for (actual, expected) in actual_inverse
            .storage()
            .as_slice()
            .iter()
            .zip(expected_inverse.iter())
        {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_strided_1d_forward_matches_slice_reference() {
        let plan = DctDstPlan::new(4, RealTransformKind::DstII).expect("valid plan");
        let signal = [1.0_f64, -2.0, 0.5, 2.25];
        let expected = plan.forward(&signal).expect("slice forward");
        let interleaved = vec![1.0, -999.0, -2.0, -999.0, 0.5, -999.0, 2.25, -999.0];
        let leto_signal =
            leto::Array1::from_shape_vec([interleaved.len()], interleaved).expect("leto input");
        let strided = leto_signal
            .view()
            .slice_with::<1>(&[SliceArg::range(Some(0), Some(8), 2)])
            .expect("strided input");

        let actual = plan.forward_leto(strided).expect("leto forward");

        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_typed_forward_matches_typed_slice_reference() {
        let plan = DctDstPlan::new(8, RealTransformKind::DctII).expect("valid plan");
        let signal = vec![1.0_f32, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let mut expected = vec![0.0_f32; signal.len()];
        plan.forward_typed_into(&signal, &mut expected, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed slice forward");
        let leto_signal =
            leto::Array1::from_shape_vec([signal.len()], signal).expect("typed leto input");

        let actual = plan
            .forward_leto_typed(leto_signal.view(), PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed leto forward");

        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn leto_2d_forward_matches_ndarray_reference() {
        let plan = DctDstPlan::new(3, RealTransformKind::DctII).expect("valid plan");
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, 3.0, -0.75],
        )
        .expect("ndarray input");
        let expected = plan.forward_2d(&input).expect("ndarray forward");
        let leto_input = leto::Array2::from_shape_vec([3, 3], input.iter().copied().collect())
            .expect("leto input");

        let actual = plan
            .forward_2d_leto(leto_input.view())
            .expect("leto forward");

        assert_eq!(actual.shape(), [3, 3]);
        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_2d_strided_inverse_matches_ndarray_reference() {
        let plan = DctDstPlan::new(3, RealTransformKind::DstII).expect("valid plan");
        let input = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, 3.0, -0.75],
        )
        .expect("ndarray input");
        let spectrum = plan.forward_2d(&input).expect("ndarray forward");
        let expected = plan.inverse_2d(&spectrum).expect("ndarray inverse");
        let mut interleaved = Vec::with_capacity(18);
        for value in spectrum.iter() {
            interleaved.push(*value);
            interleaved.push(-999.0);
        }
        let leto_input =
            leto::Array2::from_shape_vec([3, 6], interleaved).expect("leto interleaved");
        let strided = leto_input
            .view()
            .slice_with::<2>(&[
                SliceArg::range(Some(0), Some(3), 1),
                SliceArg::range(Some(0), Some(6), 2),
            ])
            .expect("strided spectrum");

        let actual = plan.inverse_2d_leto(strided).expect("leto inverse");

        assert_eq!(actual.shape(), [3, 3]);
        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_3d_forward_and_inverse_match_ndarray_reference() {
        let plan = DctDstPlan::new(2, RealTransformKind::DctIV).expect("valid plan");
        let input =
            Array3::from_shape_vec((2, 2, 2), vec![1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, 3.0])
                .expect("ndarray input");
        let expected_forward = plan.forward_3d(&input).expect("ndarray forward");
        let expected_inverse = plan.inverse_3d(&expected_forward).expect("ndarray inverse");
        let leto_input = leto::Array3::from_shape_vec([2, 2, 2], input.iter().copied().collect())
            .expect("leto input");

        let actual_forward = plan
            .forward_3d_leto(leto_input.view())
            .expect("leto forward");
        let actual_inverse = plan
            .inverse_3d_leto(actual_forward.view())
            .expect("leto inverse");

        assert_eq!(actual_forward.shape(), [2, 2, 2]);
        assert_eq!(actual_inverse.shape(), [2, 2, 2]);
        for (actual, expected) in actual_forward
            .storage()
            .as_slice()
            .iter()
            .zip(expected_forward.iter())
        {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
        for (actual, expected) in actual_inverse
            .storage()
            .as_slice()
            .iter()
            .zip(expected_inverse.iter())
        {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }
}
