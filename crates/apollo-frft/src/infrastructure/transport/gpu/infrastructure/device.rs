//! Hephaestus device acquisition and FrFT execution boundary.

use apollo_fft::PrecisionProfile;
use eunomia::Complex32;
use hephaestus_wgpu::WgpuDevice;
use mnemosyne::scratch::ScratchPool;

use crate::infrastructure::transport::gpu::application::plan::{FrftWgpuPlan, UnitaryFrftWgpuPlan};
use crate::infrastructure::transport::gpu::domain::capabilities::WgpuCapabilities;
use crate::infrastructure::transport::gpu::domain::error::{WgpuError, WgpuResult};
use crate::infrastructure::transport::gpu::infrastructure::kernel::{FrftGpuKernel, FrftMode};
use crate::infrastructure::transport::gpu::infrastructure::unitary_kernel::UnitaryFrftGpuKernel;
use crate::FrftGpuStorage;

thread_local! {
    static GPU_INPUT_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
    static GPU_OUTPUT_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
}

/// Return whether a default Hephaestus WGPU device can be acquired.
#[must_use]
pub fn wgpu_available() -> bool {
    FrftWgpuBackend::try_default().is_ok()
}

/// Hephaestus WGPU backend for direct and unitary FrFT execution.
#[derive(Debug, Clone)]
pub struct FrftWgpuBackend {
    device: WgpuDevice,
}

impl FrftWgpuBackend {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self { device }
    }

    /// Acquire the default Hephaestus WGPU device.
    pub fn try_default() -> WgpuResult<Self> {
        Ok(Self::new(WgpuDevice::try_default("apollo-frft-wgpu")?))
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::implemented(true)
    }

    /// Return the acquired Hephaestus WGPU device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only direct FrFT descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize, order: f32) -> FrftWgpuPlan {
        FrftWgpuPlan::new(len, order)
    }

    /// Create a metadata-only unitary FrFT descriptor.
    #[must_use]
    pub const fn plan_unitary(&self, len: usize, order: f32) -> UnitaryFrftWgpuPlan {
        UnitaryFrftWgpuPlan::new(len, order)
    }

    /// Execute the forward direct FrFT.
    pub fn execute_forward(
        &self,
        plan: &FrftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        self.execute_allocating(plan, input, false)
    }

    /// Execute the inverse direct FrFT.
    pub fn execute_inverse(
        &self,
        plan: &FrftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        self.execute_allocating(plan, input, true)
    }

    /// Execute the forward direct FrFT into caller-owned storage.
    pub fn execute_forward_into(
        &self,
        plan: &FrftWgpuPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        self.execute_into(plan, input, output, false)
    }

    /// Execute the inverse direct FrFT into caller-owned storage.
    pub fn execute_inverse_into(
        &self,
        plan: &FrftWgpuPlan,
        input: &[Complex32],
        output: &mut [Complex32],
    ) -> WgpuResult<()> {
        self.execute_into(plan, input, output, true)
    }

    /// Execute forward from a Leto complex host view.
    pub fn execute_forward_leto(
        &self,
        plan: &FrftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        self.execute_forward(plan, &input).and_then(|output| {
            apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
                WgpuError::InvalidPlan {
                    message: "failed to allocate Mnemosyne-backed Leto FrFT output".to_owned(),
                }
            })
        })
    }

    /// Execute inverse from a Leto complex host view.
    pub fn execute_inverse_leto(
        &self,
        plan: &FrftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        self.execute_inverse(plan, &input).and_then(|output| {
            apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
                WgpuError::InvalidPlan {
                    message: "failed to allocate Mnemosyne-backed Leto FrFT output".to_owned(),
                }
            })
        })
    }

    /// Execute the forward Candan--Gr\u00fcnbaum unitary DFrFT.
    pub fn execute_unitary_forward(
        &self,
        plan: &UnitaryFrftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        self.execute_unitary_allocating(plan, input, false)
    }

    /// Execute the inverse Candan--Gr\u00fcnbaum unitary DFrFT.
    pub fn execute_unitary_inverse(
        &self,
        plan: &UnitaryFrftWgpuPlan,
        input: &[Complex32],
    ) -> WgpuResult<Vec<Complex32>> {
        self.execute_unitary_allocating(plan, input, true)
    }

    /// Execute forward unitary DFrFT from a Leto complex host view.
    pub fn execute_unitary_forward_leto(
        &self,
        plan: &UnitaryFrftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        self.execute_unitary_forward(plan, &input)
            .and_then(|output| {
                apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
                    WgpuError::InvalidPlan {
                        message: "failed to allocate Mnemosyne-backed Leto FrFT output".to_owned(),
                    }
                })
            })
    }

    /// Execute inverse unitary DFrFT from a Leto complex host view.
    pub fn execute_unitary_inverse_leto(
        &self,
        plan: &UnitaryFrftWgpuPlan,
        input: leto::ArrayView1<'_, Complex32>,
    ) -> WgpuResult<leto::Array<Complex32, leto::MnemosyneStorage<Complex32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        self.execute_unitary_inverse(plan, &input)
            .and_then(|output| {
                apollo_leto_interop::try_array1_from_slice(&output).ok_or_else(|| {
                    WgpuError::InvalidPlan {
                        message: "failed to allocate Mnemosyne-backed Leto FrFT output".to_owned(),
                    }
                })
            })
    }

    /// Execute direct forward FrFT with storage admitted by the concrete GPU contract.
    pub fn execute_forward_typed_into<T: FrftGpuStorage>(
        &self,
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        self.execute_typed_into(plan, precision, input, output, false)
    }

    /// Execute direct inverse FrFT with storage admitted by the concrete GPU contract.
    pub fn execute_inverse_typed_into<T: FrftGpuStorage>(
        &self,
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        self.execute_typed_into(plan, precision, input, output, true)
    }

    /// Execute typed forward direct FrFT from a Leto host view.
    pub fn execute_forward_leto_typed<T: FrftGpuStorage + Default>(
        &self,
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        self.execute_typed_leto(plan, precision, input, false)
    }

    /// Execute typed inverse direct FrFT from a Leto host view.
    pub fn execute_inverse_leto_typed<T: FrftGpuStorage + Default>(
        &self,
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        self.execute_typed_leto(plan, precision, input, true)
    }

    fn execute_allocating(
        &self,
        plan: &FrftWgpuPlan,
        input: &[Complex32],
        inverse: bool,
    ) -> WgpuResult<Vec<Complex32>> {
        let mut output = vec![Complex32::new(0.0, 0.0); plan.len()];
        self.execute_into(plan, input, &mut output, inverse)?;
        Ok(output)
    }

    fn execute_into(
        &self,
        plan: &FrftWgpuPlan,
        input: &[Complex32],
        output: &mut [Complex32],
        inverse: bool,
    ) -> WgpuResult<()> {
        Self::validate_direct(plan, input, output)?;
        let (mode, cot, csc, scale_re, scale_im) = mode_params(plan, inverse)?;
        FrftGpuKernel::execute_into(
            &self.device,
            input,
            output,
            mode,
            cot,
            csc,
            scale_re,
            scale_im,
        )
    }

    fn execute_unitary_allocating(
        &self,
        plan: &UnitaryFrftWgpuPlan,
        input: &[Complex32],
        inverse: bool,
    ) -> WgpuResult<Vec<Complex32>> {
        Self::validate_unitary(plan, input)?;
        let mut output = vec![Complex32::new(0.0, 0.0); plan.len()];
        let order = if inverse { -plan.order() } else { plan.order() };
        UnitaryFrftGpuKernel::execute_into(&self.device, input, &mut output, order)?;
        Ok(output)
    }

    fn execute_typed_into<T: FrftGpuStorage>(
        &self,
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
        inverse: bool,
    ) -> WgpuResult<()> {
        Self::validate_typed(plan, precision, input, output)?;
        if let (Some(input), Some(output)) = (T::as_c32_slice(input), T::as_c32_slice_mut(output)) {
            return self.execute_into(plan, input, output, inverse);
        }
        GPU_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |represented| {
                for (target, value) in represented.iter_mut().zip(input.iter().copied()) {
                    *target = value.to_gpu();
                }
                GPU_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output.len(), |computed| {
                        self.execute_into(plan, represented, computed, inverse)?;
                        for (target, value) in output.iter_mut().zip(computed.iter().copied()) {
                            *target = T::from_gpu(value);
                        }
                        Ok(())
                    })
                })
            })
        })
    }

    fn execute_typed_leto<T: FrftGpuStorage + Default>(
        &self,
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
        inverse: bool,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        let output_slice = output
            .as_slice_mut()
            .expect("FrFT Mnemosyne output must be contiguous");
        self.execute_typed_into(plan, precision, &input, output_slice, inverse)?;
        Ok(output)
    }

    fn validate_typed<T: FrftGpuStorage>(
        plan: &FrftWgpuPlan,
        precision: PrecisionProfile,
        input: &[T],
        output: &[T],
    ) -> WgpuResult<()> {
        if precision != T::PROFILE {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Self::validate_direct_lengths(plan, input.len(), output.len())
    }

    fn validate_direct(
        plan: &FrftWgpuPlan,
        input: &[Complex32],
        output: &[Complex32],
    ) -> WgpuResult<()> {
        Self::validate_direct_lengths(plan, input.len(), output.len())
    }

    fn validate_direct_lengths(
        plan: &FrftWgpuPlan,
        input_len: usize,
        output_len: usize,
    ) -> WgpuResult<()> {
        if plan.len() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "transform length must be greater than zero".to_owned(),
            });
        }
        if !plan.order().is_finite() {
            return Err(WgpuError::NonFiniteOrder);
        }
        if input_len != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: input_len,
            });
        }
        if output_len != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output_len,
            });
        }
        Ok(())
    }

    fn validate_unitary(plan: &UnitaryFrftWgpuPlan, input: &[Complex32]) -> WgpuResult<()> {
        if plan.len() == 0 {
            return Err(WgpuError::InvalidPlan {
                message: "transform length must be greater than zero".to_owned(),
            });
        }
        if !plan.order().is_finite() {
            return Err(WgpuError::NonFiniteOrder);
        }
        if input.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: input.len(),
            });
        }
        Ok(())
    }
}

/// Determine the exact mode and chirp parameters for a direct FrFT dispatch.
fn mode_params(plan: &FrftWgpuPlan, inverse: bool) -> WgpuResult<(FrftMode, f32, f32, f32, f32)> {
    let order = if inverse { -plan.order() } else { plan.order() };
    if !order.is_finite() {
        return Err(WgpuError::NonFiniteOrder);
    }
    let reduced = ((order % 4.0_f32) + 4.0_f32) % 4.0_f32;
    let rounded = reduced.round();
    if (reduced - rounded).abs() < 1.0e-5_f32 {
        let mode = if reduced < 0.5_f32 || reduced > 3.5_f32 {
            FrftMode::Identity
        } else if reduced < 1.5_f32 {
            FrftMode::CenteredDft
        } else if reduced < 2.5_f32 {
            FrftMode::Reversal
        } else {
            FrftMode::CenteredInverseDft
        };
        return Ok((mode, 0.0, 0.0, 1.0, 0.0));
    }

    let alpha = reduced * core::f32::consts::FRAC_PI_2;
    let sin_alpha = alpha.sin();
    let cot = alpha.cos() / sin_alpha;
    let csc = sin_alpha.recip();
    let z_norm = (1.0_f32 + cot * cot).sqrt();
    let z_arg = (-cot).atan2(1.0_f32);
    let scale_radius = z_norm.sqrt() / (plan.len() as f32).sqrt();
    let scale_angle = z_arg * 0.5_f32;
    Ok((
        FrftMode::Chirp,
        cot,
        csc,
        scale_radius * scale_angle.cos(),
        scale_radius * scale_angle.sin(),
    ))
}
