//! WGPU device acquisition for this transform backend.

use std::{borrow::Cow, sync::Arc};

use apollo_dctdst::{RealTransformKind, RealTransformStorage};
use apollo_fft::PrecisionProfile;
use ndarray::{Array2, Array3};

use crate::application::plan::DctDstWgpuPlan;
use crate::domain::capabilities::WgpuCapabilities;
use crate::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::kernel::{DctGpuKernel, DctMode};
use apollo_wgpu_helpers::WgpuDevice;

/// Return whether a default WGPU adapter/device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    DctDstWgpuBackend::try_default().is_ok()
}

/// WGPU backend descriptor.
#[derive(Debug, Clone)]
pub struct DctDstWgpuBackend {
    device: WgpuDevice,
    kernel: Arc<DctGpuKernel>,
}

impl DctDstWgpuBackend {
    /// Create a backend from an existing device and queue.
    pub fn new(device: WgpuDevice) -> WgpuResult<Self> {
        let kernel = Arc::new(DctGpuKernel::new(device.inner()));
        Ok(Self { device, kernel })
    }

    /// Create a backend by requesting a default adapter and device.
    pub fn try_default() -> WgpuResult<Self> {
        Self::new(WgpuDevice::try_default("apollo-dctdst-wgpu")?)
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::full(true)
    }

    /// Return the acquired WGPU device.
    #[must_use]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        self.device.device()
    }

    /// Return the acquired WGPU queue.
    #[must_use]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        self.device.queue()
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize, kind: RealTransformKind) -> DctDstWgpuPlan {
        DctDstWgpuPlan::new(len, kind)
    }

    /// Execute the unnormalized configured real-to-real transform for a real-valued `f32` signal.
    ///
    /// Supported kinds: DCT-I, DCT-II, DCT-III, DCT-IV, DST-I, DST-II, DST-III, and DST-IV.
    /// DCT-I requires length >= 2 and returns [`WgpuError::InvalidPlan`] otherwise.
    pub fn execute_forward(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, input)?;
        let mode = match plan.kind() {
            RealTransformKind::DctII => DctMode::Dct2,
            RealTransformKind::DctIII => DctMode::Dct3,
            RealTransformKind::DstII => DctMode::Dst2,
            RealTransformKind::DstIII => DctMode::Dst3,
            RealTransformKind::DctI => {
                if plan.len() < 2 {
                    return Err(WgpuError::InvalidPlan {
                        message: format!(
                            "invalid length {}: DCT-I requires length >= 2",
                            plan.len()
                        ),
                    });
                }
                DctMode::Dct1
            }
            RealTransformKind::DctIV => DctMode::Dct4,
            RealTransformKind::DstI => DctMode::Dst1,
            RealTransformKind::DstIV => DctMode::Dst4,
        };
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            mode,
            1.0,
        )
    }

    /// Execute the unnormalized forward transform from a Leto 1D host view.
    ///
    /// Contiguous views are borrowed without copying. Strided views are
    /// materialized once into logical order before GPU upload.
    pub fn execute_forward_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = leto_view1_cow(input)?;
        let output = self.execute_forward(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the normalized inverse of the configured real-to-real transform for a real-valued `f32` signal.
    ///
    /// Supported kinds: DCT-I, DCT-II, DCT-III, DCT-IV, DST-I, DST-II, DST-III, and DST-IV.
    /// DCT-I requires length >= 2 and returns [`WgpuError::InvalidPlan`] otherwise.
    /// Inverse scales: DCT-I = 1/(2*(N-1)), DCT-IV = 2/N, DST-I = 1/(2*(N+1)), DST-IV = 2/N.
    pub fn execute_inverse(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        Self::validate_plan_input(plan, input)?;
        let (mode, scale) = match plan.kind() {
            RealTransformKind::DctII => (DctMode::Dct3, 2.0 / plan.len() as f32),
            RealTransformKind::DctIII => (DctMode::Dct2, 2.0 / plan.len() as f32),
            RealTransformKind::DstII => (DctMode::Dst3, 2.0 / plan.len() as f32),
            RealTransformKind::DstIII => (DctMode::Dst2, 2.0 / plan.len() as f32),
            RealTransformKind::DctI => {
                if plan.len() < 2 {
                    return Err(WgpuError::InvalidPlan {
                        message: format!(
                            "invalid length {}: DCT-I requires length >= 2",
                            plan.len()
                        ),
                    });
                }
                (DctMode::Dct1, 1.0 / (2.0 * (plan.len() - 1) as f32))
            }
            RealTransformKind::DctIV => (DctMode::Dct4, 2.0 / plan.len() as f32),
            RealTransformKind::DstI => (DctMode::Dst1, 1.0 / (2.0 * (plan.len() + 1) as f32)),
            RealTransformKind::DstIV => (DctMode::Dst4, 2.0 / plan.len() as f32),
        };
        self.kernel.execute(
            self.device.inner(),
            self.device.queue().as_ref(),
            input,
            mode,
            scale,
        )
    }

    /// Execute the normalized inverse transform from a Leto 1D host view.
    pub fn execute_inverse_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = leto_view1_cow(input)?;
        let output = self.execute_inverse(plan, &input)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the forward real-to-real transform with typed storage.
    ///
    /// WGPU arithmetic is `f32`; mixed `f16` storage is promoted once to `f32` at
    /// the dispatch boundary and quantized at the output boundary.
    pub fn execute_forward_typed_into<T: RealTransformStorage>(
        &self,
        plan: &DctDstWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_dct_typed_precision::<T>(precision)?;
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        if output.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_forward(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
        Ok(())
    }

    /// Execute typed forward transform from a Leto 1D host view.
    pub fn execute_forward_leto_typed<T: RealTransformStorage>(
        &self,
        plan: &DctDstWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = leto_view1_cow(input)?;
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_forward_typed_into(plan, precision, &input, &mut output)?;
        leto_array1_from_slice(&output)
    }

    /// Execute the normalized inverse real-to-real transform with typed storage.
    pub fn execute_inverse_typed_into<T: RealTransformStorage>(
        &self,
        plan: &DctDstWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_dct_typed_precision::<T>(precision)?;
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        if output.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: output.len(),
            });
        }
        let represented: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
        let computed = self.execute_inverse(plan, &represented)?;
        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
            *slot = T::from_f64(f64::from(value));
        }
        Ok(())
    }

    /// Execute typed inverse transform from a Leto 1D host view.
    pub fn execute_inverse_leto_typed<T: RealTransformStorage>(
        &self,
        plan: &DctDstWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = leto_view1_cow(input)?;
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_inverse_typed_into(plan, precision, &input, &mut output)?;
        leto_array1_from_slice(&output)
    }

    fn validate_dct_typed_precision<T: RealTransformStorage>(
        precision: PrecisionProfile,
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Ok(())
    }

    fn validate_plan_input(plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<()> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        if input.len() != len {
            return Err(WgpuError::LengthMismatch {
                expected: len,
                actual: input.len(),
            });
        }
        Ok(())
    }

    /// Execute the unnormalized 2D separable forward real-to-real transform.
    ///
    /// Applies the 1D forward transform along each row then each column.
    /// Requires a square `n × n` input where `n == plan.len()`.
    pub fn execute_forward_2d(
        &self,
        plan: &DctDstWgpuPlan,
        input: &Array2<f32>,
    ) -> WgpuResult<Array2<f32>> {
        let n = plan.len();
        let (rows, cols) = input.dim();
        if rows != n || cols != n {
            return Err(WgpuError::ShapeMismatch {
                message: format!("2D input expected {n}x{n}, got {rows}x{cols}"),
            });
        }
        // Row pass.
        let mut tmp = Array2::<f32>::zeros((n, n));
        for r in 0..n {
            let row: Vec<f32> = input.row(r).iter().copied().collect();
            let out = self.execute_forward(plan, &row)?;
            for c in 0..n {
                tmp[[r, c]] = out[c];
            }
        }
        // Column pass.
        let mut result = Array2::<f32>::zeros((n, n));
        for c in 0..n {
            let col: Vec<f32> = tmp.column(c).iter().copied().collect();
            let out = self.execute_forward(plan, &col)?;
            for r in 0..n {
                result[[r, c]] = out[r];
            }
        }
        Ok(result)
    }

    /// Execute the unnormalized 2D separable forward transform from a Leto view.
    pub fn execute_forward_2d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let input = array2_from_leto_view(input)?;
        let output = self.execute_forward_2d(plan, &input)?;
        leto_array2_from_ndarray(&output)
    }

    /// Execute the normalized 2D separable inverse real-to-real transform.
    ///
    /// Applies the 1D inverse transform along each column then each row.
    /// Requires a square `n × n` input where `n == plan.len()`.
    pub fn execute_inverse_2d(
        &self,
        plan: &DctDstWgpuPlan,
        input: &Array2<f32>,
    ) -> WgpuResult<Array2<f32>> {
        let n = plan.len();
        let (rows, cols) = input.dim();
        if rows != n || cols != n {
            return Err(WgpuError::ShapeMismatch {
                message: format!("2D input expected {n}x{n}, got {rows}x{cols}"),
            });
        }
        // Column pass (inverse).
        let mut tmp = Array2::<f32>::zeros((n, n));
        for c in 0..n {
            let col: Vec<f32> = input.column(c).iter().copied().collect();
            let out = self.execute_inverse(plan, &col)?;
            for r in 0..n {
                tmp[[r, c]] = out[r];
            }
        }
        // Row pass (inverse).
        let mut result = Array2::<f32>::zeros((n, n));
        for r in 0..n {
            let row: Vec<f32> = tmp.row(r).iter().copied().collect();
            let out = self.execute_inverse(plan, &row)?;
            for c in 0..n {
                result[[r, c]] = out[c];
            }
        }
        Ok(result)
    }

    /// Execute the normalized 2D separable inverse transform from a Leto view.
    pub fn execute_inverse_2d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let input = array2_from_leto_view(input)?;
        let output = self.execute_inverse_2d(plan, &input)?;
        leto_array2_from_ndarray(&output)
    }

    /// Execute the unnormalized 3D separable forward real-to-real transform.
    ///
    /// Applies the 1D forward transform along axes 0, 1, and 2 in sequence.
    /// Requires a cubic `n × n × n` input where `n == plan.len()`.
    pub fn execute_forward_3d(
        &self,
        plan: &DctDstWgpuPlan,
        input: &Array3<f32>,
    ) -> WgpuResult<Array3<f32>> {
        let n = plan.len();
        let (d0, d1, d2) = input.dim();
        if d0 != n || d1 != n || d2 != n {
            return Err(WgpuError::ShapeMismatch {
                message: format!("3D input expected {n}x{n}x{n}, got {d0}x{d1}x{d2}"),
            });
        }
        // Axis-0 pass.
        let mut tmp0 = Array3::<f32>::zeros((n, n, n));
        for j in 0..n {
            for k in 0..n {
                let fiber: Vec<f32> = (0..n).map(|i| input[[i, j, k]]).collect();
                let out = self.execute_forward(plan, &fiber)?;
                for i in 0..n {
                    tmp0[[i, j, k]] = out[i];
                }
            }
        }
        // Axis-1 pass.
        let mut tmp1 = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for k in 0..n {
                let fiber: Vec<f32> = (0..n).map(|j| tmp0[[i, j, k]]).collect();
                let out = self.execute_forward(plan, &fiber)?;
                for j in 0..n {
                    tmp1[[i, j, k]] = out[j];
                }
            }
        }
        // Axis-2 pass.
        let mut result = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for j in 0..n {
                let fiber: Vec<f32> = (0..n).map(|k| tmp1[[i, j, k]]).collect();
                let out = self.execute_forward(plan, &fiber)?;
                for k in 0..n {
                    result[[i, j, k]] = out[k];
                }
            }
        }
        Ok(result)
    }

    /// Execute the unnormalized 3D separable forward transform from a Leto view.
    pub fn execute_forward_3d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView3<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>> {
        let input = array3_from_leto_view(input)?;
        let output = self.execute_forward_3d(plan, &input)?;
        leto_array3_from_ndarray(&output)
    }

    /// Execute the normalized 3D separable inverse real-to-real transform.
    ///
    /// Applies the 1D inverse transform along axes 2, 1, and 0 in sequence.
    /// Requires a cubic `n × n × n` input where `n == plan.len()`.
    pub fn execute_inverse_3d(
        &self,
        plan: &DctDstWgpuPlan,
        input: &Array3<f32>,
    ) -> WgpuResult<Array3<f32>> {
        let n = plan.len();
        let (d0, d1, d2) = input.dim();
        if d0 != n || d1 != n || d2 != n {
            return Err(WgpuError::ShapeMismatch {
                message: format!("3D input expected {n}x{n}x{n}, got {d0}x{d1}x{d2}"),
            });
        }
        // Axis-2 pass (inverse).
        let mut tmp0 = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for j in 0..n {
                let fiber: Vec<f32> = (0..n).map(|k| input[[i, j, k]]).collect();
                let out = self.execute_inverse(plan, &fiber)?;
                for k in 0..n {
                    tmp0[[i, j, k]] = out[k];
                }
            }
        }
        // Axis-1 pass (inverse).
        let mut tmp1 = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for k in 0..n {
                let fiber: Vec<f32> = (0..n).map(|j| tmp0[[i, j, k]]).collect();
                let out = self.execute_inverse(plan, &fiber)?;
                for j in 0..n {
                    tmp1[[i, j, k]] = out[j];
                }
            }
        }
        // Axis-0 pass (inverse).
        let mut result = Array3::<f32>::zeros((n, n, n));
        for j in 0..n {
            for k in 0..n {
                let fiber: Vec<f32> = (0..n).map(|i| tmp1[[i, j, k]]).collect();
                let out = self.execute_inverse(plan, &fiber)?;
                for i in 0..n {
                    result[[i, j, k]] = out[i];
                }
            }
        }
        Ok(result)
    }

    /// Execute the normalized 3D separable inverse transform from a Leto view.
    pub fn execute_inverse_3d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView3<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>> {
        let input = array3_from_leto_view(input)?;
        let output = self.execute_inverse_3d(plan, &input)?;
        leto_array3_from_ndarray(&output)
    }
}

fn leto_view1_cow<T: Copy>(view: leto::ArrayView1<'_, T>) -> WgpuResult<Cow<'_, [T]>> {
    if let Some(slice) = view.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }
    let len = view.shape()[0];
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        values.push(*view.get([index]).map_err(|err| WgpuError::ShapeMismatch {
            message: format!("invalid Leto DCT/DST 1D view: {err:?}"),
        })?);
    }
    Ok(Cow::Owned(values))
}

fn array2_from_leto_view(view: leto::ArrayView2<'_, f32>) -> WgpuResult<Array2<f32>> {
    let shape = view.shape();
    let rows = shape[0];
    let cols = shape[1];
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            values.push(
                *view
                    .get([row, col])
                    .map_err(|err| WgpuError::ShapeMismatch {
                        message: format!("invalid Leto DCT/DST 2D view: {err:?}"),
                    })?,
            );
        }
    }
    Array2::from_shape_vec((rows, cols), values).map_err(|err| WgpuError::ShapeMismatch {
        message: format!("failed to materialize Leto DCT/DST 2D view: {err}"),
    })
}

fn array3_from_leto_view(view: leto::ArrayView3<'_, f32>) -> WgpuResult<Array3<f32>> {
    let shape = view.shape();
    let d0 = shape[0];
    let d1 = shape[1];
    let d2 = shape[2];
    let mut values = Vec::with_capacity(d0 * d1 * d2);
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                values.push(
                    *view
                        .get([i, j, k])
                        .map_err(|err| WgpuError::ShapeMismatch {
                            message: format!("invalid Leto DCT/DST 3D view: {err:?}"),
                        })?,
                );
            }
        }
    }
    Array3::from_shape_vec((d0, d1, d2), values).map_err(|err| WgpuError::ShapeMismatch {
        message: format!("failed to materialize Leto DCT/DST 3D view: {err}"),
    })
}

fn leto_array1_from_slice<T: Copy>(
    values: &[T],
) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
    leto::Array::from_mnemosyne_slice([values.len()], values).map_err(|err| {
        WgpuError::InvalidPlan {
            message: format!("failed to allocate Mnemosyne-backed Leto DCT/DST 1D output: {err:?}"),
        }
    })
}

fn leto_array2_from_ndarray(
    values: &Array2<f32>,
) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
    let (rows, cols) = values.dim();
    let flat: Vec<f32> = values.iter().copied().collect();
    leto::Array::from_mnemosyne_slice([rows, cols], &flat).map_err(|err| WgpuError::InvalidPlan {
        message: format!("failed to allocate Mnemosyne-backed Leto DCT/DST 2D output: {err:?}"),
    })
}

fn leto_array3_from_ndarray(
    values: &Array3<f32>,
) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>> {
    let (d0, d1, d2) = values.dim();
    let flat: Vec<f32> = values.iter().copied().collect();
    leto::Array::from_mnemosyne_slice([d0, d1, d2], &flat).map_err(|err| WgpuError::InvalidPlan {
        message: format!("failed to allocate Mnemosyne-backed Leto DCT/DST 3D output: {err:?}"),
    })
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use leto::SliceArg;

    use super::leto_view1_cow;

    #[test]
    fn leto_view1_cow_borrows_contiguous_views() {
        let input = leto::Array1::from_shape_vec([4], vec![1.0_f32, 2.0, 3.0, 4.0]).expect("input");
        let cow = leto_view1_cow(input.view()).expect("contiguous view");
        assert!(matches!(cow, Cow::Borrowed(_)));
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn leto_view1_cow_materializes_strided_views() {
        let input =
            leto::Array1::from_shape_vec([8], vec![1.0_f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0])
                .expect("input");
        let view = input
            .slice_with::<1>(&[SliceArg::range(Some(0), None, 2)])
            .expect("strided view");
        let cow = leto_view1_cow(view).expect("strided view");
        assert!(matches!(cow, Cow::Owned(_)));
        assert_eq!(cow.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
    }
}
