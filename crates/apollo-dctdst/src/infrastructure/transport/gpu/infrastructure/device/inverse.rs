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
        self.kernel.execute(&self.device, input, mode, scale)
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
        let represented = if let Some(slice_f32) = T::as_f32_slice(input) {
            Cow::Borrowed(slice_f32)
        } else {
            let vec: Vec<f32> = input.iter().map(|v| v.to_f64() as f32).collect();
            Cow::Owned(vec)
        };
        let computed = self.execute_inverse(plan, &represented)?;
        if let Some(slice_f32) = T::as_f32_slice_mut(output) {
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
    /// Row-major flat `n×n` host buffer in/out; flat host orchestration between
    /// per-line GPU (Hephaestus) dispatches.
    pub fn execute_inverse_2d(
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
        // Column pass: gather each strided column into a reused lane.
        let mut lane = vec![0.0_f32; n];
        let mut tmp = vec![0.0_f32; n * n];
        for c in 0..n {
            for r in 0..n {
                lane[r] = input[r * n + c];
            }
            let out = self.execute_inverse(plan, &lane)?;
            for r in 0..n {
                tmp[r * n + c] = out[r];
            }
        }
        // Row pass: rows are contiguous `n`-element chunks.
        let mut result = vec![0.0_f32; n * n];
        for r in 0..n {
            let out = self.execute_inverse(plan, &tmp[r * n..r * n + n])?;
            result[r * n..r * n + n].copy_from_slice(&out);
        }
        Ok(result)
    }

    /// Execute the normalized 2D separable inverse transform from a Leto view.
    /// Leto appears only at this CPU↔GPU seam.
    pub fn execute_inverse_2d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let n = plan.len();
        let flat: Vec<f32> = input.iter().copied().collect();
        let result = self.execute_inverse_2d(plan, &flat)?;
        leto::Array::from_mnemosyne_vec([n, n], result).map_err(|_| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto DCT/DST 2D output".to_string(),
        })
    }

    /// Execute the normalized 3D separable inverse real-to-real transform.
    ///
    /// Applies the 1D inverse transform along axes 2, 1, and 0 in sequence.
    /// Requires a cubic `n × n × n` input where `n == plan.len()`.
    pub fn execute_inverse_3d(
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
        // Axis-2 pass (contiguous over k).
        let mut tmp0 = vec![0.0_f32; n * n * n];
        for i in 0..n {
            for j in 0..n {
                let base = idx(i, j, 0);
                let out = self.execute_inverse(plan, &input[base..base + n])?;
                tmp0[base..base + n].copy_from_slice(&out);
            }
        }
        // Axis-1 pass (strided over j).
        let mut tmp1 = vec![0.0_f32; n * n * n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    lane[j] = tmp0[idx(i, j, k)];
                }
                let out = self.execute_inverse(plan, &lane)?;
                for j in 0..n {
                    tmp1[idx(i, j, k)] = out[j];
                }
            }
        }
        // Axis-0 pass (strided over i).
        let mut result = vec![0.0_f32; n * n * n];
        for j in 0..n {
            for k in 0..n {
                for i in 0..n {
                    lane[i] = tmp1[idx(i, j, k)];
                }
                let out = self.execute_inverse(plan, &lane)?;
                for i in 0..n {
                    result[idx(i, j, k)] = out[i];
                }
            }
        }
        Ok(result)
    }

    /// Execute the normalized 3D separable inverse transform from a Leto view.
    /// Leto appears only at this CPU↔GPU seam.
    pub fn execute_inverse_3d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView3<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>> {
        let n = plan.len();
        let flat: Vec<f32> = input.iter().copied().collect();
        let result = self.execute_inverse_3d(plan, &flat)?;
        leto::Array::from_mnemosyne_vec([n, n, n], result).map_err(|_| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto DCT/DST 3D output".to_string(),
        })
    }
}
