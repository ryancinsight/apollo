use std::borrow::Cow;

use crate::{RealTransformKind, RealTransformStorage};
use apollo_fft::PrecisionProfile;
use ndarray::{Array2, Array3};

use crate::infrastructure::transport::gpu::application::plan::DctDstWgpuPlan;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::device::helpers::{
    array2_from_leto_view, array3_from_leto_view, leto_array1_from_slice, leto_array2_from_ndarray,
    leto_array3_from_ndarray, leto_view1_cow,
};
use crate::infrastructure::transport::gpu::infrastructure::device::DctDstWgpuBackend;
use crate::infrastructure::transport::gpu::infrastructure::kernel::DctMode;

impl DctDstWgpuBackend {
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
            &self.device,
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
        let input = leto_view1_cow(input);
        let output = self.execute_forward(plan, &input)?;
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
        let represented = if let Some(slice_f32) = T::as_f32_slice(input) {
            Cow::Borrowed(slice_f32)
        } else {
            let vec: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
            Cow::Owned(vec)
        };
        let computed = self.execute_forward(plan, &represented)?;
        if let Some(slice_f32) = T::as_f32_slice_mut(output) {
            slice_f32.copy_from_slice(&computed);
        } else {
            for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
                *slot = T::from_f64(f64::from(value));
            }
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
        let input = leto_view1_cow(input);
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_forward_typed_into(plan, precision, &input, &mut output)?;
        leto_array1_from_slice(&output)
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
        // Row pass: contiguous rows are borrowed; non-contiguous layouts fall
        // back to one reused lane buffer instead of a per-row allocation.
        let mut lane = Vec::with_capacity(n);
        let mut tmp = Array2::<f32>::zeros((n, n));
        for r in 0..n {
            let row = input.row(r);
            let out = match row.as_slice() {
                Some(slice) => self.execute_forward(plan, slice)?,
                None => {
                    lane.clear();
                    lane.extend(row.iter().copied());
                    self.execute_forward(plan, &lane)?
                }
            };
            tmp.row_mut(r)
                .iter_mut()
                .zip(&out)
                .for_each(|(s, v)| *s = *v);
        }
        // Column pass: columns are strided, so the single lane buffer is reused.
        let mut result = Array2::<f32>::zeros((n, n));
        for c in 0..n {
            lane.clear();
            lane.extend(tmp.column(c).iter().copied());
            let out = self.execute_forward(plan, &lane)?;
            result
                .column_mut(c)
                .iter_mut()
                .zip(&out)
                .for_each(|(s, v)| *s = *v);
        }
        Ok(result)
    }

    /// Execute the unnormalized 2D separable forward transform from a Leto view.
    pub fn execute_forward_2d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let input = array2_from_leto_view(input);
        let output = self.execute_forward_2d(plan, &input)?;
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
        // One lane buffer reused across all three axis passes; fibers are
        // strided so each is gathered into it instead of a fresh allocation.
        let mut lane = Vec::with_capacity(n);
        // Axis-0 pass.
        let mut tmp0 = Array3::<f32>::zeros((n, n, n));
        for j in 0..n {
            for k in 0..n {
                lane.clear();
                lane.extend((0..n).map(|i| input[[i, j, k]]));
                let out = self.execute_forward(plan, &lane)?;
                for i in 0..n {
                    tmp0[[i, j, k]] = out[i];
                }
            }
        }
        // Axis-1 pass.
        let mut tmp1 = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for k in 0..n {
                lane.clear();
                lane.extend((0..n).map(|j| tmp0[[i, j, k]]));
                let out = self.execute_forward(plan, &lane)?;
                for j in 0..n {
                    tmp1[[i, j, k]] = out[j];
                }
            }
        }
        // Axis-2 pass.
        let mut result = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for j in 0..n {
                lane.clear();
                lane.extend((0..n).map(|k| tmp1[[i, j, k]]));
                let out = self.execute_forward(plan, &lane)?;
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
        let input = array3_from_leto_view(input);
        let output = self.execute_forward_3d(plan, &input)?;
        leto_array3_from_ndarray(&output)
    }
}
