use crate::{RealTransformGpuStorage, RealTransformKind};
use apollo_fft::PrecisionProfile;

use crate::infrastructure::transport::gpu::application::plan::DctDstWgpuPlan;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::device::helpers::{
    leto_array1_from_slice, leto_view1_cow,
};
use crate::infrastructure::transport::gpu::infrastructure::device::DctDstWgpuBackend;
use crate::infrastructure::transport::gpu::infrastructure::kernel::{
    DctGpuKernel, DctMode, FiberLayout,
};

impl DctDstWgpuBackend {
    /// The GPU kernel mode and per-axis normalization scale for the plan's
    /// inverse transform (DCT-I = 1/(2(N-1)), DCT/DST-IV = 2/N, DST-I =
    /// 1/(2(N+1)), else 2/N; DCT-I requires length ≥ 2 else
    /// [`WgpuError::InvalidPlan`]).
    ///
    /// A separable rank-`d` inverse applies this scale once per axis; since the
    /// scale is a scalar that commutes with the linear transform, the batched
    /// path folds it into a single `scale.powi(d)`.
    pub(crate) fn inverse_mode_scale(plan: &DctDstWgpuPlan) -> WgpuResult<(DctMode, f32)> {
        Ok(match plan.kind() {
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
        })
    }

    /// Execute the normalized inverse of the configured 1-D real-to-real
    /// transform for a real-valued `f32` signal (DCT-I/II/III/IV, DST-I/II/III/IV).
    pub fn execute_inverse(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; plan.len()];
        self.execute_inverse_into(plan, input, &mut output)?;
        Ok(output)
    }

    /// Execute the normalized inverse into caller-owned `f32` storage without
    /// a host-side result allocation.
    pub fn execute_inverse_into(
        &self,
        plan: &DctDstWgpuPlan,
        input: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_plan_input(plan, input)?;
        Self::validate_output(plan, output)?;
        let (mode, scale) = Self::inverse_mode_scale(plan)?;
        DctGpuKernel::execute_into(&self.device, input, output, mode, scale)
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
    pub fn execute_inverse_typed_into<T: RealTransformGpuStorage>(
        &self,
        plan: &DctDstWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_dct_typed_precision::<T>(precision)?;
        Self::validate_typed_plan_input::<T>(plan, input, output)?;
        let (mode, scale) = Self::inverse_mode_scale(plan)?;
        self.execute_typed_into(input, output, mode, scale)
    }

    /// Execute typed inverse transform from a Leto 1D host view.
    pub fn execute_inverse_leto_typed<T: RealTransformGpuStorage + Default>(
        &self,
        plan: &DctDstWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = leto_view1_cow(input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_inverse_typed_into(
            plan,
            precision,
            &input,
            output
                .as_slice_mut()
                .expect("DCT/DST typed Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the normalized 2D separable inverse real-to-real transform.
    ///
    /// Row-major flat `n×n` host buffer in/out. Both inverse axis passes run
    /// on-device (one batched dispatch each); the per-axis normalization folds
    /// into a single `scale²` applied once before download.
    pub fn execute_inverse_2d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        let n = plan.len();
        let expected = Self::cubic_element_count(n, 2)?;
        if input.len() != expected {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "2D input expected {n}x{n} = {expected} elements, got {}",
                    input.len()
                ),
            });
        }
        let (mode, scale) = Self::inverse_mode_scale(plan)?;
        let passes = [
            (mode, FiberLayout::axis(n, 2, 1)?),
            (mode, FiberLayout::axis(n, 2, 0)?),
        ];
        let mut output = vec![0.0_f32; input.len()];
        DctGpuKernel::execute_separable_into(
            &self.device,
            input,
            &mut output,
            &passes,
            scale.powi(2),
        )?;
        Ok(output)
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
    /// Row-major flat `n³` host buffer in/out; three on-device batched passes,
    /// per-axis normalization folded into a single `scale³`.
    pub fn execute_inverse_3d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        let n = plan.len();
        let expected = Self::cubic_element_count(n, 3)?;
        if input.len() != expected {
            return Err(WgpuError::ShapeMismatch {
                message: format!(
                    "3D input expected {n}x{n}x{n} = {expected} elements, got {}",
                    input.len()
                ),
            });
        }
        let (mode, scale) = Self::inverse_mode_scale(plan)?;
        let passes = [
            (mode, FiberLayout::axis(n, 3, 2)?),
            (mode, FiberLayout::axis(n, 3, 1)?),
            (mode, FiberLayout::axis(n, 3, 0)?),
        ];
        let mut output = vec![0.0_f32; input.len()];
        DctGpuKernel::execute_separable_into(
            &self.device,
            input,
            &mut output,
            &passes,
            scale.powi(3),
        )?;
        Ok(output)
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
