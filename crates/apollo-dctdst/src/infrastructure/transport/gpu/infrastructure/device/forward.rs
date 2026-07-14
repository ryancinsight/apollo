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
    /// The GPU kernel mode for the plan's forward transform (DCT-I requires
    /// length ≥ 2, else [`WgpuError::InvalidPlan`]).
    pub(crate) fn forward_mode(plan: &DctDstWgpuPlan) -> WgpuResult<DctMode> {
        Ok(match plan.kind() {
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
        })
    }

    /// Execute the unnormalized configured 1-D real-to-real transform for a
    /// real-valued `f32` signal (DCT-I/II/III/IV, DST-I/II/III/IV).
    pub fn execute_forward(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; plan.len()];
        self.execute_forward_into(plan, input, &mut output)?;
        Ok(output)
    }

    /// Execute the unnormalized configured transform into caller-owned `f32`
    /// storage without a host-side result allocation.
    pub fn execute_forward_into(
        &self,
        plan: &DctDstWgpuPlan,
        input: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_plan_input(plan, input)?;
        Self::validate_output(plan, output)?;
        let mode = Self::forward_mode(plan)?;
        DctGpuKernel::execute_into(&self.device, input, output, mode, 1.0)
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
    pub fn execute_forward_typed_into<T: RealTransformGpuStorage>(
        &self,
        plan: &DctDstWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_dct_typed_precision::<T>(precision)?;
        Self::validate_typed_plan_input::<T>(plan, input, output)?;
        let mode = Self::forward_mode(plan)?;
        self.execute_typed_into(input, output, mode, 1.0)
    }

    /// Execute typed forward transform from a Leto 1D host view.
    pub fn execute_forward_leto_typed<T: RealTransformGpuStorage + Default>(
        &self,
        plan: &DctDstWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = leto_view1_cow(input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_forward_typed_into(
            plan,
            precision,
            &input,
            output
                .as_slice_mut()
                .expect("DCT/DST typed Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the unnormalized 2D separable forward real-to-real transform.
    ///
    /// Row-major flat `n×n` host buffer in/out. Both axis passes run on-device
    /// via one batched dispatch each (rows then columns), so the field is
    /// uploaded once and downloaded once — no per-line host round trips.
    /// Requires `input.len() == n²` where `n == plan.len()`.
    pub fn execute_forward_2d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
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
        let mode = Self::forward_mode(plan)?;
        let passes = [
            (mode, FiberLayout::axis(n, 2, 1)?),
            (mode, FiberLayout::axis(n, 2, 0)?),
        ];
        let mut output = vec![0.0_f32; input.len()];
        DctGpuKernel::execute_separable_into(&self.device, input, &mut output, &passes, 1.0)?;
        Ok(output)
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
    /// Row-major flat `n³` host buffer in/out. The three axis passes run
    /// on-device as one batched dispatch each; the field is uploaded once and
    /// downloaded once. Requires `input.len() == n³` where `n == plan.len()`.
    pub fn execute_forward_3d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
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
        let mode = Self::forward_mode(plan)?;
        let passes = [
            (mode, FiberLayout::axis(n, 3, 2)?),
            (mode, FiberLayout::axis(n, 3, 1)?),
            (mode, FiberLayout::axis(n, 3, 0)?),
        ];
        let mut output = vec![0.0_f32; input.len()];
        DctGpuKernel::execute_separable_into(&self.device, input, &mut output, &passes, 1.0)?;
        Ok(output)
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
