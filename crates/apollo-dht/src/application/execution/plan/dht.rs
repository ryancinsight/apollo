//! Reusable Discrete Hartley Transform plan.

use crate::domain::contracts::error::{DhtError, DhtResult};
use crate::domain::metadata::length::HartleyLength;
use crate::domain::spectrum::coefficients::HartleySpectrum;
use crate::infrastructure::kernel::direct::transform_real;
use crate::infrastructure::kernel::fast::dht_fast_with_scratch;
use apollo_fft::{f16, PrecisionProfile};
use mnemosyne::scratch::ScratchPool;
use ndarray::{Array2, Array3};
use num_complex::Complex64;

const FAST_KERNEL_THRESHOLD: usize = 512;

// ── Thread-local scratch buffers ────────────────────────────────────────────
//
// The plan previously stored three `Mutex<Vec<_>>` fields (`fast_scratch`,
// `lane_scratch`, `typed_scratch`) that serialised every call on a mutex even
// for single-threaded use. Replaced by `thread_local!  RefCell<Vec<_>>` buffers
// that grow on demand and are reused across calls, matching the CZT and FFT
// plan scratch precedents.

thread_local! {
    static FAST_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    static LANE_IN_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static LANE_OUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static TYPED_INPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static TYPED_OUTPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Reusable 1D real-to-real DHT plan.
#[derive(Debug)]
pub struct DhtPlan {
    length: HartleyLength,
}

impl DhtPlan {
    fn forward_2d_impl(&self, input: &Array2<f64>, output: &mut Array2<f64>) -> DhtResult<()> {
        let n = self.len();
        let (rows, cols) = input.dim();
        let (out_rows, out_cols) = output.dim();
        if rows != n || cols != n {
            return Err(DhtError::ShapeMismatch2d {
                expected: n,
                rows,
                cols,
            });
        }
        if out_rows != n || out_cols != n {
            return Err(DhtError::ShapeMismatch2d {
                expected: n,
                rows: out_rows,
                cols: out_cols,
            });
        }

        LANE_IN_SCRATCH.with(|in_pool| {
            in_pool.with_scratch(n, |lane_in| {
                LANE_OUT_SCRATCH.with(|out_pool| {
                    out_pool.with_scratch(n, |lane_out| {
                        for r in 0..n {
                            for c in 0..n {
                                lane_in[c] = input[[r, c]];
                            }
                            self.forward_into(lane_in, lane_out)?;
                            for c in 0..n {
                                output[[r, c]] = lane_out[c];
                            }
                        }

                        for c in 0..n {
                            for r in 0..n {
                                lane_in[r] = output[[r, c]];
                            }
                            self.forward_into(lane_in, lane_out)?;
                            for r in 0..n {
                                output[[r, c]] = lane_out[r];
                            }
                        }

                        Ok(())
                    })
                })
            })
        })
    }

    fn forward_3d_impl(&self, input: &Array3<f64>, output: &mut Array3<f64>) -> DhtResult<()> {
        let n = self.len();
        let (d0, d1, d2) = input.dim();
        let (o0, o1, o2) = output.dim();
        if d0 != n || d1 != n || d2 != n {
            return Err(DhtError::ShapeMismatch3d {
                expected: n,
                d0,
                d1,
                d2,
            });
        }
        if o0 != n || o1 != n || o2 != n {
            return Err(DhtError::ShapeMismatch3d {
                expected: n,
                d0: o0,
                d1: o1,
                d2: o2,
            });
        }

        LANE_IN_SCRATCH.with(|in_pool| {
            in_pool.with_scratch(n, |lane_in| {
                LANE_OUT_SCRATCH.with(|out_pool| {
                    out_pool.with_scratch(n, |lane_out| {
                        for j in 0..n {
                            for k in 0..n {
                                for i in 0..n {
                                    lane_in[i] = input[[i, j, k]];
                                }
                                self.forward_into(lane_in, lane_out)?;
                                for i in 0..n {
                                    output[[i, j, k]] = lane_out[i];
                                }
                            }
                        }

                        for i in 0..n {
                            for k in 0..n {
                                for j in 0..n {
                                    lane_in[j] = output[[i, j, k]];
                                }
                                self.forward_into(lane_in, lane_out)?;
                                for j in 0..n {
                                    output[[i, j, k]] = lane_out[j];
                                }
                            }
                        }

                        for i in 0..n {
                            for j in 0..n {
                                for k in 0..n {
                                    lane_in[k] = output[[i, j, k]];
                                }
                                self.forward_into(lane_in, lane_out)?;
                                for k in 0..n {
                                    output[[i, j, k]] = lane_out[k];
                                }
                            }
                        }

                        Ok(())
                    })
                })
            })
        })
    }

    /// Create a DHT plan for a non-empty signal length.
    pub fn new(len: usize) -> DhtResult<Self> {
        let length = HartleyLength::new(len)?;
        Ok(Self { length })
    }

    /// Return validated transform length.
    #[must_use]
    pub const fn length(&self) -> HartleyLength {
        self.length
    }

    /// Return transform length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length.get()
    }

    /// Return true when transform length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.length.is_empty()
    }

    /// Execute the unnormalized forward DHT.
    pub fn forward(&self, signal: &[f64]) -> DhtResult<HartleySpectrum> {
        let mut output = vec![0.0; self.len()];
        self.forward_into(signal, &mut output)?;
        Ok(HartleySpectrum::new(output))
    }

    /// Execute the unnormalized forward DHT into a zero-allocation buffer.
    pub fn forward_into(&self, signal: &[f64], output: &mut [f64]) -> DhtResult<()> {
        if signal.len() != self.len() || output.len() != self.len() {
            return Err(DhtError::LengthMismatch);
        }
        if self.len() >= FAST_KERNEL_THRESHOLD {
            FAST_SCRATCH.with(|pool| {
                pool.with_scratch(self.len(), |scratch| {
                    dht_fast_with_scratch(signal, output, scratch);
                });
            });
            Ok(())
        } else {
            transform_real(signal, output)
        }
    }

    /// Execute the inverse DHT by reusing the forward kernel and applying `1 / N`.
    pub fn inverse(&self, spectrum: &HartleySpectrum) -> DhtResult<Vec<f64>> {
        let mut output = vec![0.0; self.len()];
        self.inverse_into(spectrum.values(), &mut output)?;
        Ok(output)
    }

    /// Execute the inverse DHT securely into a zero-allocation buffer.
    pub fn inverse_into(&self, spectrum: &[f64], output: &mut [f64]) -> DhtResult<()> {
        if spectrum.len() != self.len() || output.len() != self.len() {
            return Err(DhtError::LengthMismatch);
        }
        self.forward_into(spectrum, output)?;
        let scale = 1.0 / self.len() as f64;
        output.iter_mut().for_each(|value| *value *= scale);
        Ok(())
    }

    /// Apply one raw unnormalized DHT pass.
    pub fn transform_unscaled(&self, input: &[f64]) -> DhtResult<Vec<f64>> {
        let mut output = vec![0.0; self.len()];
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// Execute the unnormalized separable 2D forward DHT on an N×N array.
    ///
    /// The 2D DHT is separable: apply the 1D DHT to every row, then to every
    /// column. The mathematical identity is
    /// `H_{2D}[m,n] = Σ_j Σ_k x[j,k] cas(2πjm/N) cas(2πkn/N)`,
    /// which equals the row-DHT applied column-wise (order-invariant).
    /// Requires a square `N×N` input where `N == self.len()`.
    pub fn forward_2d(&self, input: &Array2<f64>) -> DhtResult<Array2<f64>> {
        let n = self.len();
        let mut result = Array2::<f64>::zeros((n, n));
        self.forward_2d_impl(input, &mut result)?;
        Ok(result)
    }

    /// Execute the unnormalized separable 2D forward DHT on a Leto N×N view.
    ///
    /// Leto validates the input shape and stride contract. Contiguous and
    /// strided inputs enter the same canonical separable kernel after copying
    /// into the row-major workspace required by the existing lane scheduler.
    pub fn forward_2d_leto(
        &self,
        input: leto::ArrayView2<'_, f64>,
    ) -> DhtResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let n = self.len();
        let input = array2_from_leto_view(input);
        let mut output = Array2::<f64>::zeros((n, n));
        self.forward_2d_impl(&input, &mut output)?;
        Ok(leto_array2_from_ndarray(&output))
    }

    /// Execute the unnormalized separable 2D forward DHT into a caller-owned buffer.
    pub fn forward_2d_into(&self, input: &Array2<f64>, output: &mut Array2<f64>) -> DhtResult<()> {
        self.forward_2d_impl(input, output)
    }

    /// Execute the normalized separable 2D inverse DHT on an N×N spectrum.
    ///
    /// Since DHT is involutory (DHT∘DHT = N·I), the 2D inverse is
    /// `(1/N²) · forward_2d`. This is mathematically exact with no
    /// additional kernel required.
    pub fn inverse_2d(&self, input: &Array2<f64>) -> DhtResult<Array2<f64>> {
        let n = self.len();
        let mut result = Array2::<f64>::zeros((n, n));
        self.forward_2d_impl(input, &mut result)?;
        let scale = 1.0 / (n * n) as f64;
        result.mapv_inplace(|v| v * scale);
        Ok(result)
    }

    /// Execute the normalized separable 2D inverse DHT on a Leto N×N spectrum.
    pub fn inverse_2d_leto(
        &self,
        input: leto::ArrayView2<'_, f64>,
    ) -> DhtResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 2>> {
        let n = self.len();
        let input = array2_from_leto_view(input);
        let mut output = Array2::<f64>::zeros((n, n));
        self.forward_2d_impl(&input, &mut output)?;
        let scale = 1.0 / (n * n) as f64;
        output.mapv_inplace(|v| v * scale);
        Ok(leto_array2_from_ndarray(&output))
    }

    /// Execute the normalized separable 2D inverse DHT into a caller-owned buffer.
    pub fn inverse_2d_into(&self, input: &Array2<f64>, output: &mut Array2<f64>) -> DhtResult<()> {
        self.forward_2d_impl(input, output)?;
        let scale = 1.0 / (self.len() * self.len()) as f64;
        output.mapv_inplace(|v| v * scale);
        Ok(())
    }

    /// Execute the unnormalized separable 3D forward DHT on an N×N×N array.
    ///
    /// Applies the 1D DHT along axis 0, then axis 1, then axis 2.
    /// Requires a cubic `N×N×N` input where `N == self.len()`.
    pub fn forward_3d(&self, input: &Array3<f64>) -> DhtResult<Array3<f64>> {
        let n = self.len();
        let mut result = Array3::<f64>::zeros((n, n, n));
        self.forward_3d_impl(input, &mut result)?;
        Ok(result)
    }

    /// Execute the unnormalized separable 3D forward DHT on a Leto N×N×N view.
    pub fn forward_3d_leto(
        &self,
        input: leto::ArrayView3<'_, f64>,
    ) -> DhtResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let n = self.len();
        let input = array3_from_leto_view(input);
        let mut output = Array3::<f64>::zeros((n, n, n));
        self.forward_3d_impl(&input, &mut output)?;
        Ok(leto_array3_from_ndarray(&output))
    }

    /// Execute the unnormalized separable 3D forward DHT into a caller-owned buffer.
    pub fn forward_3d_into(&self, input: &Array3<f64>, output: &mut Array3<f64>) -> DhtResult<()> {
        self.forward_3d_impl(input, output)
    }

    /// Execute the normalized separable 3D inverse DHT on an N×N×N spectrum.
    ///
    /// `inverse_3d = (1/N³) · forward_3d` by DHT involutory property.
    pub fn inverse_3d(&self, input: &Array3<f64>) -> DhtResult<Array3<f64>> {
        let n = self.len();
        let mut result = Array3::<f64>::zeros((n, n, n));
        self.forward_3d_impl(input, &mut result)?;
        let scale = 1.0 / (n * n * n) as f64;
        result.mapv_inplace(|v| v * scale);
        Ok(result)
    }

    /// Execute the normalized separable 3D inverse DHT on a Leto N×N×N spectrum.
    pub fn inverse_3d_leto(
        &self,
        input: leto::ArrayView3<'_, f64>,
    ) -> DhtResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 3>> {
        let n = self.len();
        let input = array3_from_leto_view(input);
        let mut output = Array3::<f64>::zeros((n, n, n));
        self.forward_3d_impl(&input, &mut output)?;
        let scale = 1.0 / (n * n * n) as f64;
        output.mapv_inplace(|v| v * scale);
        Ok(leto_array3_from_ndarray(&output))
    }

    /// Execute the normalized separable 3D inverse DHT into a caller-owned buffer.
    pub fn inverse_3d_into(&self, input: &Array3<f64>, output: &mut Array3<f64>) -> DhtResult<()> {
        self.forward_3d_impl(input, output)?;
        let n = self.len();
        let scale = 1.0 / (n * n * n) as f64;
        output.mapv_inplace(|v| v * scale);
        Ok(())
    }

    /// Execute the unnormalized DHT for `f64`, `f32`, or mixed `f16` storage.
    ///
    /// `f64` uses the native high-accuracy path. `f32` and mixed `f16` storage
    /// convert through the `f64` owner kernel and quantize once into the caller
    /// supplied output. This preserves a single mathematical implementation and
    /// avoids duplicated precision-specific kernels.
    pub fn forward_typed_into<T: HartleyStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        T::forward_into(self, signal, output, profile)
    }

    /// Execute the normalized inverse DHT for `f64`, `f32`, or mixed `f16` storage.
    pub fn inverse_typed_into<T: HartleyStorage>(
        &self,
        spectrum: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        T::inverse_into(self, spectrum, output, profile)
    }
}

/// Real storage accepted by typed DHT paths.
pub trait HartleyStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage value to the owner `f64` arithmetic path.
    fn to_f64(self) -> f64;
    /// Convert owner arithmetic result back to storage.
    fn from_f64(value: f64) -> Self;

    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &DhtPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(DhtError::LengthMismatch);
        }
        let n = plan.len();
        TYPED_INPUT64_SCRATCH.with(|in_pool| {
            in_pool.with_scratch(n, |input64| {
                TYPED_OUTPUT64_SCRATCH.with(|out_pool| {
                    out_pool.with_scratch(n, |output64| {
                        for (slot, value) in input64.iter_mut().zip(signal.iter()) {
                            *slot = value.to_f64();
                        }
                        plan.forward_into(input64, output64)?;
                        for (slot, value) in output.iter_mut().zip(output64.iter()) {
                            *slot = Self::from_f64(*value);
                        }
                        Ok(())
                    })
                })
            })
        })
    }

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &DhtPlan,
        spectrum: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if spectrum.len() != plan.len() || output.len() != plan.len() {
            return Err(DhtError::LengthMismatch);
        }
        let n = plan.len();
        TYPED_INPUT64_SCRATCH.with(|in_pool| {
            in_pool.with_scratch(n, |input64| {
                TYPED_OUTPUT64_SCRATCH.with(|out_pool| {
                    out_pool.with_scratch(n, |output64| {
                        for (slot, value) in input64.iter_mut().zip(spectrum.iter()) {
                            *slot = value.to_f64();
                        }
                        plan.inverse_into(input64, output64)?;
                        for (slot, value) in output.iter_mut().zip(output64.iter()) {
                            *slot = Self::from_f64(*value);
                        }
                        Ok(())
                    })
                })
            })
        })
    }
}

impl HartleyStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn forward_into(
        plan: &DhtPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(signal, output)
    }

    fn inverse_into(
        plan: &DhtPlan,
        spectrum: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DhtResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(spectrum, output)
    }
}

impl HartleyStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl HartleyStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f32(value as f32)
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> DhtResult<()> {
    if actual.storage == expected.storage && actual.compute == expected.compute {
        Ok(())
    } else {
        Err(DhtError::PrecisionMismatch)
    }
}

fn array2_from_leto_view(input: leto::ArrayView2<'_, f64>) -> Array2<f64> {
    let [rows, cols] = input.shape();
    Array2::from_shape_fn((rows, cols), |(row, col)| {
        *input
            .get([row, col])
            .expect("Leto 2D DHT view index must be valid after shape validation")
    })
}

fn array3_from_leto_view(input: leto::ArrayView3<'_, f64>) -> Array3<f64> {
    let [d0, d1, d2] = input.shape();
    Array3::from_shape_fn((d0, d1, d2), |(i, j, k)| {
        *input
            .get([i, j, k])
            .expect("Leto 3D DHT view index must be valid after shape validation")
    })
}

fn leto_array2_from_ndarray(
    output: &Array2<f64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 2> {
    let (rows, cols) = output.dim();
    leto::Array::<f64, leto::MnemosyneStorage<f64>, 2>::from_mnemosyne_slice(
        [rows, cols],
        output
            .as_slice()
            .expect("DHT-owned 2D ndarray output must be contiguous"),
    )
    .expect("DHT 2D output length must match Leto output shape")
}

fn leto_array3_from_ndarray(
    output: &Array3<f64>,
) -> leto::Array<f64, leto::MnemosyneStorage<f64>, 3> {
    let (d0, d1, d2) = output.dim();
    leto::Array::<f64, leto::MnemosyneStorage<f64>, 3>::from_mnemosyne_slice(
        [d0, d1, d2],
        output
            .as_slice()
            .expect("DHT-owned 3D ndarray output must be contiguous"),
    )
    .expect("DHT 3D output length must match Leto output shape")
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use leto::{SliceArg, Storage};

    #[test]
    fn typed_paths_support_f64_f32_and_mixed_f16_storage() {
        let plan = DhtPlan::new(8).expect("valid plan");
        let signal64 = [1.0_f64, -2.0, 0.5, 2.25, -4.0, 1.5, 0.0, -0.75];
        let expected = plan.forward(&signal64).expect("forward");
        let expected_inverse = plan.inverse(&expected).expect("inverse");

        let mut out64 = [0.0_f64; 8];
        plan.forward_typed_into(&signal64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed f64 forward");
        for (actual, expected) in out64.iter().zip(expected.values()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }

        let signal32 = signal64.map(|value| value as f32);
        let mut out32 = [0.0_f32; 8];
        plan.forward_typed_into(&signal32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed f32 forward");
        for (actual, expected) in out32.iter().zip(expected.values()) {
            assert!((f64::from(*actual) - *expected).abs() < 1.0e-5);
        }
        let mut inv32 = [0.0_f32; 8];
        plan.inverse_typed_into(&out32, &mut inv32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed f32 inverse");
        for (actual, expected) in inv32.iter().zip(expected_inverse.iter()) {
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
        for (actual, expected) in out16.iter().zip(expected.values()) {
            let quantization_bound = expected.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual.to_f32()) - *expected).abs() <= quantization_bound);
        }
    }

    #[test]
    fn typed_path_rejects_profile_storage_mismatch() {
        let plan = DhtPlan::new(4).expect("valid plan");
        let signal = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        assert!(matches!(
            plan.forward_typed_into(&signal, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(DhtError::PrecisionMismatch)
        ));
    }

    #[test]
    fn leto_2d_forward_matches_ndarray_reference() {
        let plan = DhtPlan::new(3).expect("valid plan");
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
        let plan = DhtPlan::new(3).expect("valid plan");
        let dense = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, 3.0, -0.75],
        )
        .expect("dense input");
        let spectrum = plan.forward_2d(&dense).expect("ndarray forward");
        let expected = plan.inverse_2d(&spectrum).expect("ndarray inverse");

        let mut interleaved = Vec::with_capacity(18);
        for value in spectrum.iter() {
            interleaved.push(*value);
            interleaved.push(-999.0);
        }
        let leto_input =
            leto::Array2::from_shape_vec([3, 6], interleaved).expect("leto interleaved input");
        let strided = leto_input
            .view()
            .slice_with::<2>(&[
                SliceArg::range(Some(0), Some(3), 1),
                SliceArg::range(Some(0), Some(6), 2),
            ])
            .expect("strided Leto view");
        let actual = plan.inverse_2d_leto(strided).expect("leto inverse");

        assert_eq!(actual.shape(), [3, 3]);
        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_3d_forward_matches_ndarray_reference() {
        let plan = DhtPlan::new(2).expect("valid plan");
        let input =
            Array3::from_shape_vec((2, 2, 2), vec![1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, 3.0])
                .expect("ndarray input");
        let expected = plan.forward_3d(&input).expect("ndarray forward");

        let leto_input = leto::Array3::from_shape_vec([2, 2, 2], input.iter().copied().collect())
            .expect("leto input");
        let actual = plan
            .forward_3d_leto(leto_input.view())
            .expect("leto forward");

        assert_eq!(actual.shape(), [2, 2, 2]);
        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_3d_inverse_matches_ndarray_reference() {
        let plan = DhtPlan::new(2).expect("valid plan");
        let input =
            Array3::from_shape_vec((2, 2, 2), vec![1.0, -2.0, 0.5, 4.0, 0.25, -1.5, 2.0, 3.0])
                .expect("ndarray input");
        let spectrum = plan.forward_3d(&input).expect("ndarray forward");
        let expected = plan.inverse_3d(&spectrum).expect("ndarray inverse");

        let leto_input =
            leto::Array3::from_shape_vec([2, 2, 2], spectrum.iter().copied().collect())
                .expect("leto input");
        let actual = plan
            .inverse_3d_leto(leto_input.view())
            .expect("leto inverse");

        assert_eq!(actual.shape(), [2, 2, 2]);
        for (actual, expected) in actual.storage().as_slice().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
        }
    }
}
