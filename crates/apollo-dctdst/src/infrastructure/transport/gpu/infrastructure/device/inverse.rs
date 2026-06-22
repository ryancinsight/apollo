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
            &self.device,
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
        let input = leto_view1_cow(input);
        let output = self.execute_inverse(plan, &input)?;
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
        let represented = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safety: T is f32, so &[T] is layout-compatible with &[f32].
            let slice_f32 =
                unsafe { std::slice::from_raw_parts(input.as_ptr().cast::<f32>(), input.len()) };
            Cow::Borrowed(slice_f32)
        } else {
            let vec: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
            Cow::Owned(vec)
        };
        let computed = self.execute_inverse(plan, &represented)?;
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safety: T is f32, so &mut [T] is layout-compatible with &mut [f32].
            let slice_f32 = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<f32>(), output.len())
            };
            slice_f32.copy_from_slice(&computed);
        } else {
            for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
                *slot = T::from_f64(f64::from(value));
            }
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
        let input = leto_view1_cow(input);
        let mut output = vec![T::from_f64(0.0); plan.len()];
        self.execute_inverse_typed_into(plan, precision, &input, &mut output)?;
        leto_array1_from_slice(&output)
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
        // Column pass (inverse): columns are strided, so a single lane buffer
        // is reused instead of a per-column allocation.
        let mut lane = Vec::with_capacity(n);
        let mut tmp = Array2::<f32>::zeros((n, n));
        for c in 0..n {
            lane.clear();
            lane.extend(input.column(c).iter().copied());
            let out = self.execute_inverse(plan, &lane)?;
            tmp.column_mut(c)
                .iter_mut()
                .zip(&out)
                .for_each(|(s, v)| *s = *v);
        }
        // Row pass (inverse): contiguous rows are borrowed without copying.
        let mut result = Array2::<f32>::zeros((n, n));
        for r in 0..n {
            let row = tmp.row(r);
            let out = match row.as_slice() {
                Some(slice) => self.execute_inverse(plan, slice)?,
                None => {
                    lane.clear();
                    lane.extend(row.iter().copied());
                    self.execute_inverse(plan, &lane)?
                }
            };
            result
                .row_mut(r)
                .iter_mut()
                .zip(&out)
                .for_each(|(s, v)| *s = *v);
        }
        Ok(result)
    }

    /// Execute the normalized 2D separable inverse transform from a Leto view.
    pub fn execute_inverse_2d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let input = array2_from_leto_view(input);
        let output = self.execute_inverse_2d(plan, &input)?;
        leto_array2_from_ndarray(&output)
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
        // One lane buffer reused across all three axis passes; fibers are
        // strided so each is gathered into it instead of a fresh allocation.
        let mut lane = Vec::with_capacity(n);
        // Axis-2 pass (inverse).
        let mut tmp0 = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for j in 0..n {
                lane.clear();
                lane.extend((0..n).map(|k| input[[i, j, k]]));
                let out = self.execute_inverse(plan, &lane)?;
                for k in 0..n {
                    tmp0[[i, j, k]] = out[k];
                }
            }
        }
        // Axis-1 pass (inverse).
        let mut tmp1 = Array3::<f32>::zeros((n, n, n));
        for i in 0..n {
            for k in 0..n {
                lane.clear();
                lane.extend((0..n).map(|j| tmp0[[i, j, k]]));
                let out = self.execute_inverse(plan, &lane)?;
                for j in 0..n {
                    tmp1[[i, j, k]] = out[j];
                }
            }
        }
        // Axis-0 pass (inverse).
        let mut result = Array3::<f32>::zeros((n, n, n));
        for j in 0..n {
            for k in 0..n {
                lane.clear();
                lane.extend((0..n).map(|i| tmp1[[i, j, k]]));
                let out = self.execute_inverse(plan, &lane)?;
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
        let input = array3_from_leto_view(input);
        let output = self.execute_inverse_3d(plan, &input)?;
        leto_array3_from_ndarray(&output)
    }
}
