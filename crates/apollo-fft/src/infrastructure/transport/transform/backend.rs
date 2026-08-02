//! Shared WGPU transform backend orchestration (ADR 0037).
//!
//! One generic backend owns what previously drifted across nineteen
//! per-transform copies: plan/input/output validation, thread-local
//! scratch reuse for typed dispatch, `_into` caller-owned-storage
//! execution, and Mnemosyne-native Leto outputs. The transform supplies
//! only its kernel dispatch through [`GpuTransformExecutor`].

use hephaestus_wgpu::WgpuDevice;
use mnemosyne::scratch::ScratchPool;

use super::capabilities::WgpuCapabilities;
use super::error::{WgpuError, WgpuResult};
use super::plan::WgpuTransformPlan;
use super::storage::GpuStorage;
use crate::PrecisionProfile;

thread_local! {
    static GPU_INPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
    static GPU_OUTPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
}

/// Kernel dispatch supplied by a transform to the shared backend.
///
/// Implementors are zero-sized markers owning the transform's shader
/// sources, parameter structs, and pass sequence. The scaffold's contract
/// is equal input and output length (`plan.len()` both sides); the
/// `inverse` flag selects the adjoint/normalized-inverse pass sequence.
pub trait GpuTransformExecutor {
    /// Execute the transform into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns the provider failure or an invalid-plan rejection.
    fn execute_into(
        device: &WgpuDevice,
        input: &[f32],
        output: &mut [f32],
        inverse: bool,
    ) -> WgpuResult<()>;
}

/// WGPU backend descriptor shared by every adopting transform.
#[derive(Debug, Clone)]
pub struct WgpuTransformBackend<X> {
    device: WgpuDevice,
    transform: core::marker::PhantomData<X>,
}

impl<X: GpuTransformExecutor> WgpuTransformBackend<X> {
    /// Create a backend from an acquired Hephaestus WGPU device.
    #[must_use]
    pub const fn new(device: WgpuDevice) -> Self {
        Self {
            device,
            transform: core::marker::PhantomData,
        }
    }

    /// Return truthful current capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> WgpuCapabilities {
        WgpuCapabilities::implemented(true)
    }

    /// Return the acquired Hephaestus device implementation.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Create a metadata-only plan descriptor.
    #[must_use]
    pub const fn plan(&self, len: usize) -> WgpuTransformPlan<X> {
        WgpuTransformPlan::new(len)
    }

    /// Execute the unnormalized forward transform for a real-valued `f32`
    /// signal.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    pub fn execute_forward(
        &self,
        plan: &WgpuTransformPlan<X>,
        input: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; plan.len()];
        self.execute_forward_into(plan, input, &mut output)?;
        Ok(output)
    }

    /// Execute the unnormalized forward transform into caller-owned
    /// contiguous storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    pub fn execute_forward_into(
        &self,
        plan: &WgpuTransformPlan<X>,
        input: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_plan_input(plan, input)?;
        Self::validate_output(plan, output)?;
        X::execute_into(&self.device, input, output, false)
    }

    /// Execute the unnormalized forward transform from a Leto `f32` view.
    ///
    /// Contiguous Leto views borrow host storage directly; strided views
    /// copy once into logical order before dispatching to the slice path.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    pub fn execute_forward_leto(
        &self,
        plan: &WgpuTransformPlan<X>,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_forward_into(
            plan,
            &input,
            output
                .as_slice_mut()
                .expect("transform Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the unnormalized forward transform with caller-owned typed
    /// storage.
    ///
    /// WGPU arithmetic remains `f32`; mixed `f16` storage is promoted once
    /// to represented `f32` before dispatch and quantized at the output
    /// boundary.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    pub fn execute_forward_typed_into<T: GpuStorage>(
        &self,
        plan: &WgpuTransformPlan<X>,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_plan_input::<T>(plan, precision, input, output)?;
        self.execute_typed_into(input, output, false)
    }

    /// Execute the unnormalized forward transform from typed Leto storage.
    ///
    /// Precision-profile validation and host quantization match
    /// [`Self::execute_forward_typed_into`].
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    pub fn execute_forward_leto_typed<T: GpuStorage + Default>(
        &self,
        plan: &WgpuTransformPlan<X>,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_forward_typed_into(
            plan,
            precision,
            &input,
            output
                .as_slice_mut()
                .expect("typed transform Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the normalized inverse transform for a real-valued `f32`
    /// spectrum.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    pub fn execute_inverse(
        &self,
        plan: &WgpuTransformPlan<X>,
        input: &[f32],
    ) -> WgpuResult<Vec<f32>> {
        let mut output = vec![0.0_f32; plan.len()];
        self.execute_inverse_into(plan, input, &mut output)?;
        Ok(output)
    }

    /// Execute the normalized inverse transform into caller-owned
    /// contiguous storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    pub fn execute_inverse_into(
        &self,
        plan: &WgpuTransformPlan<X>,
        input: &[f32],
        output: &mut [f32],
    ) -> WgpuResult<()> {
        Self::validate_plan_input(plan, input)?;
        Self::validate_output(plan, output)?;
        X::execute_into(&self.device, input, output, true)
    }

    /// Execute the normalized inverse transform from a Leto `f32` view.
    ///
    /// Output storage is Mnemosyne-backed Leto host memory.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    pub fn execute_inverse_leto(
        &self,
        plan: &WgpuTransformPlan<X>,
        input: leto::ArrayView1<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<f32, leto::MnemosyneStorage<f32>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_inverse_into(
            plan,
            &input,
            output
                .as_slice_mut()
                .expect("inverse transform Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the normalized inverse transform with caller-owned typed
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    pub fn execute_inverse_typed_into<T: GpuStorage>(
        &self,
        plan: &WgpuTransformPlan<X>,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed_plan_input::<T>(plan, precision, input, output)?;
        self.execute_typed_into(input, output, true)
    }

    /// Execute the normalized inverse transform from typed Leto storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    pub fn execute_inverse_leto_typed<T: GpuStorage + Default>(
        &self,
        plan: &WgpuTransformPlan<X>,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.len()]);
        self.execute_inverse_typed_into(
            plan,
            precision,
            &input,
            output
                .as_slice_mut()
                .expect("typed inverse transform Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    fn validate_plan_input(plan: &WgpuTransformPlan<X>, input: &[f32]) -> WgpuResult<()> {
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

    fn validate_typed_plan_input<T: GpuStorage>(
        plan: &WgpuTransformPlan<X>,
        precision: PrecisionProfile,
        input: &[T],
        output: &[T],
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
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
        Ok(())
    }

    fn validate_output(plan: &WgpuTransformPlan<X>, output: &[f32]) -> WgpuResult<()> {
        if output.len() != plan.len() {
            return Err(WgpuError::LengthMismatch {
                expected: plan.len(),
                actual: output.len(),
            });
        }
        Ok(())
    }

    fn execute_typed_into<T: GpuStorage>(
        &self,
        input: &[T],
        output: &mut [T],
        inverse: bool,
    ) -> WgpuResult<()> {
        if let (Some(input), Some(output)) = (T::as_f32_slice(input), T::as_f32_slice_mut(output)) {
            return X::execute_into(&self.device, input, output, inverse);
        }
        GPU_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input.len(), |represented| {
                for (slot, value) in represented.iter_mut().zip(input.iter().copied()) {
                    *slot = value.to_gpu();
                }
                GPU_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output.len(), |computed| {
                        X::execute_into(&self.device, represented, computed, inverse)?;
                        for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
                            *slot = T::from_gpu(value);
                        }
                        Ok(())
                    })
                })
            })
        })
    }
}
