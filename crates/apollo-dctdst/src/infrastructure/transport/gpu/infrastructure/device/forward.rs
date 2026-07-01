use std::borrow::Cow;

use crate::{RealTransformKind, RealTransformStorage};
use apollo_fft::PrecisionProfile;

use crate::infrastructure::transport::gpu::application::plan::DctDstWgpuPlan;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::device::helpers::{
    leto_array1_from_slice, leto_view1_cow,
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
        self.kernel.execute(&self.device, input, mode, 1.0)
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
    /// Row-major flat `n×n` host buffer in, transformed flat buffer out. The
    /// separable orchestration is flat host arithmetic between per-line GPU
    /// dispatches; the GPU work itself runs on Hephaestus buffers.
    pub fn execute_forward_2d(
        &self,
        plan: &DctDstWgpuPlan,
        input: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        let n = plan.len();
        if input.len() != n * n {
            return Err(WgpuError::ShapeMismatch {
                message: format!("2D input expected {n}x{n} = {} elements, got {}", n * n, input.len()),
            });
        }
        // Row pass: rows are contiguous `n`-element chunks.
        let mut tmp = vec![0.0_f32; n * n];
        for r in 0..n {
            let out = self.execute_forward(plan, &input[r * n..r * n + n])?;
            tmp[r * n..r * n + n].copy_from_slice(&out);
        }
        // Column pass: gather each strided column into a reused lane.
        let mut lane = vec![0.0_f32; n];
        let mut result = vec![0.0_f32; n * n];
        for c in 0..n {
            for r in 0..n {
                lane[r] = tmp[r * n + c];
            }
            let out = self.execute_forward(plan, &lane)?;
            for r in 0..n {
                result[r * n + c] = out[r];
            }
        }
        Ok(result)
    }

    /// Execute the unnormalized 2D separable forward transform from a Leto view.
    /// Leto appears only at this CPU↔GPU seam.
    pub fn execute_forward_2d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let n = plan.len();
        let flat: Vec<f32> = input.iter().copied().collect();
        let result = self.execute_forward_2d(plan, &flat)?;
        leto::Array::from_mnemosyne_vec([n, n], result).map_err(|_| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto DCT/DST 2D output".to_string(),
        })
    }

    /// Execute the unnormalized 3D separable forward real-to-real transform.
    ///
    /// Applies the 1D forward transform along axes 0, 1, and 2 in sequence.
    /// Requires a cubic `n × n × n` input where `n == plan.len()`.
    pub fn execute_forward_3d(
        &self,
        plan: &DctDstWgpuPlan,
        input: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        let n = plan.len();
        if input.len() != n * n * n {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "3D input expected {n}x{n}x{n} = {} elements, got {}",
                    n * n * n,
                    input.len()
                ),
            });
        }
        // Row-major flat index of (i, j, k) is (i*n + j)*n + k.
        let idx = |i: usize, j: usize, k: usize| (i * n + j) * n + k;
        let mut lane = vec![0.0_f32; n];
        // Axis-0 pass (strided over i).
        let mut tmp0 = vec![0.0_f32; n * n * n];
        for j in 0..n {
            for k in 0..n {
                for i in 0..n {
                    lane[i] = input[idx(i, j, k)];
                }
                let out = self.execute_forward(plan, &lane)?;
                for i in 0..n {
                    tmp0[idx(i, j, k)] = out[i];
                }
            }
        }
        // Axis-1 pass (strided over j).
        let mut tmp1 = vec![0.0_f32; n * n * n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    lane[j] = tmp0[idx(i, j, k)];
                }
                let out = self.execute_forward(plan, &lane)?;
                for j in 0..n {
                    tmp1[idx(i, j, k)] = out[j];
                }
            }
        }
        // Axis-2 pass (contiguous over k).
        let mut result = vec![0.0_f32; n * n * n];
        for i in 0..n {
            for j in 0..n {
                let base = idx(i, j, 0);
                let out = self.execute_forward(plan, &tmp1[base..base + n])?;
                result[base..base + n].copy_from_slice(&out);
            }
        }
        Ok(result)
    }

    /// Execute the unnormalized 3D separable forward transform from a Leto view.
    /// Leto appears only at this CPU↔GPU seam.
    pub fn execute_forward_3d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView3<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>> {
        let n = plan.len();
        let flat: Vec<f32> = input.iter().copied().collect();
        let result = self.execute_forward_3d(plan, &flat)?;
        leto::Array::from_mnemosyne_vec([n, n, n], result).map_err(|_| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto DCT/DST 3D output".to_string(),
        })
    }
}
