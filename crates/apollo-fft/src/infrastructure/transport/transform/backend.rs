//! Shared WGPU transform backend orchestration (ADR 0037).
//!
//! One generic backend owns what previously drifted across nineteen
//! per-transform copies: plan/input/output validation, thread-local
//! scratch reuse for typed dispatch, `_into` caller-owned-storage
//! execution, and Mnemosyne-native Leto outputs. The transform supplies
//! its kernel dispatch, plan payload, and element types through
//! [`GpuTransformExecutor`]: `Sample` is the time/space-domain element,
//! `Bin` the transform-domain element, so complex-valued and asymmetric
//! transforms (real in, complex out) state their contract in the types
//! instead of forcing everything through `f32`.

use hephaestus_wgpu::WgpuDevice;

use super::capabilities::WgpuCapabilities;
use super::error::{WgpuError, WgpuResult};
use super::plan::WgpuTransformPlan;
use super::storage::{GpuElement, GpuStorage};
use crate::PrecisionProfile;

/// Kernel dispatch, plan payload, and element types supplied by a
/// transform to the shared backend.
///
/// Implementors are zero-sized markers owning the transform's shader
/// sources, parameter structs, and pass sequences. The plan payload names
/// what the transform's descriptors carry — a bare length for
/// same-length 1D transforms, a richer structure where the transform
/// demands one. Forward maps `Sample` slices of `input_len` to `Bin`
/// slices of `output_len`; inverse maps back.
pub trait GpuTransformPlanner {
    /// Plan payload carried by this transform's descriptors.
    type Plan: Copy + core::fmt::Debug + PartialEq + Send + Sync + 'static;

    /// Logical input length demanded by a plan payload.
    fn input_len(plan: &Self::Plan) -> usize;

    /// Logical output length produced by a plan payload.
    ///
    /// Equal to the input length for same-length transforms; spectra- or
    /// grid-shaped transforms override it.
    fn output_len(plan: &Self::Plan) -> usize {
        Self::input_len(plan)
    }

    /// Validate transform-specific plan structure.
    ///
    /// The shared backend already rejects zero input length and
    /// input/output length mismatches; this hook carries constraints only
    /// the transform knows (power-of-two lengths, level bounds, …).
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan rejection naming the violated constraint.
    fn validate(plan: &Self::Plan) -> WgpuResult<()> {
        let _ = plan;
        Ok(())
    }
}

/// Kernel dispatch and element types for transforms whose whole
/// execution surface fits the shared slice contract.
///
/// Transforms whose operations take extra operands (a graph Fourier
/// basis, per-call signal bounds) implement only
/// [`GpuTransformPlanner`] and carry their surface as an extension
/// trait on the aliased backend.
pub trait GpuTransformExecutor: GpuTransformPlanner {
    /// Time/space-domain element accepted by the forward direction.
    type Sample: bytemuck::Pod + Default + core::fmt::Debug + Send + Sync + 'static;

    /// Transform-domain element produced by the forward direction.
    type Bin: bytemuck::Pod + Default + core::fmt::Debug + Send + Sync + 'static;

    /// Execute the unnormalized forward transform into caller-owned
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns the provider failure or an invalid-plan rejection.
    fn forward_into(
        device: &WgpuDevice,
        plan: &Self::Plan,
        input: &[Self::Sample],
        output: &mut [Self::Bin],
    ) -> WgpuResult<()>;

    /// Execute the normalized inverse transform into caller-owned
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns the provider failure or an invalid-plan rejection.
    fn inverse_into(
        device: &WgpuDevice,
        plan: &Self::Plan,
        input: &[Self::Bin],
        output: &mut [Self::Sample],
    ) -> WgpuResult<()>;
}

/// WGPU backend descriptor shared by every adopting transform.
#[derive(Debug, Clone)]
pub struct WgpuTransformBackend<X> {
    device: WgpuDevice,
    transform: core::marker::PhantomData<X>,
}

impl<X: GpuTransformPlanner> WgpuTransformBackend<X> {
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
    pub const fn plan(&self, payload: X::Plan) -> WgpuTransformPlan<X> {
        WgpuTransformPlan::new(payload)
    }

    pub(super) fn validate_plan(plan: &WgpuTransformPlan<X>) -> WgpuResult<()> {
        let len = plan.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        X::validate(plan.payload())
    }

    pub(super) fn require_len(
        role: &'static str,
        actual: usize,
        expected: usize,
    ) -> WgpuResult<()> {
        let _ = role;
        if actual != expected {
            return Err(WgpuError::LengthMismatch { expected, actual });
        }
        Ok(())
    }
}

impl<X: GpuTransformExecutor> WgpuTransformBackend<X> {
    /// Execute the unnormalized forward transform.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    pub fn execute_forward(
        &self,
        plan: &WgpuTransformPlan<X>,
        input: &[X::Sample],
    ) -> WgpuResult<Vec<X::Bin>> {
        let mut output = vec![X::Bin::default(); plan.output_len()];
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
        input: &[X::Sample],
        output: &mut [X::Bin],
    ) -> WgpuResult<()> {
        Self::validate_plan(plan)?;
        Self::require_len("forward input", input.len(), plan.len())?;
        Self::require_len("forward output", output.len(), plan.output_len())?;
        X::forward_into(&self.device, plan.payload(), input, output)
    }

    /// Execute the unnormalized forward transform from a Leto view.
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
        input: leto::ArrayView1<'_, X::Sample>,
    ) -> WgpuResult<leto::Array<X::Bin, leto::MnemosyneStorage<X::Bin>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<X::Bin, leto::MnemosyneStorage<X::Bin>, 1>::zeros_mnemosyne([
                plan.output_len()
            ]);
        self.execute_forward_into(
            plan,
            &input,
            output
                .as_slice_mut()
                .expect("transform Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }

    /// Execute the normalized inverse transform.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    pub fn execute_inverse(
        &self,
        plan: &WgpuTransformPlan<X>,
        input: &[X::Bin],
    ) -> WgpuResult<Vec<X::Sample>> {
        let mut output = vec![X::Sample::default(); plan.len()];
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
        input: &[X::Bin],
        output: &mut [X::Sample],
    ) -> WgpuResult<()> {
        Self::validate_plan(plan)?;
        Self::require_len("inverse input", input.len(), plan.output_len())?;
        Self::require_len("inverse output", output.len(), plan.len())?;
        X::inverse_into(&self.device, plan.payload(), input, output)
    }

    /// Execute the normalized inverse transform from a Leto view.
    ///
    /// Output storage is Mnemosyne-backed Leto host memory.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    pub fn execute_inverse_leto(
        &self,
        plan: &WgpuTransformPlan<X>,
        input: leto::ArrayView1<'_, X::Bin>,
    ) -> WgpuResult<leto::Array<X::Sample, leto::MnemosyneStorage<X::Sample>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<X::Sample, leto::MnemosyneStorage<X::Sample>, 1>::zeros_mnemosyne([
                plan.len()
            ]);
        self.execute_inverse_into(
            plan,
            &input,
            output
                .as_slice_mut()
                .expect("inverse transform Mnemosyne output must be contiguous"),
        )?;
        Ok(output)
    }
}

impl<E, X> WgpuTransformBackend<X>
where
    E: GpuElement,
    X: GpuTransformExecutor<Sample = E, Bin = E>,
{
    /// Execute the unnormalized forward transform with caller-owned typed
    /// storage.
    ///
    /// WGPU arithmetic stays in the transform's native element; reduced
    /// storage forms are promoted once to the represented element before
    /// dispatch and quantized at the output boundary. Available on
    /// symmetric transforms, whose forward and inverse directions share
    /// one element.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    pub fn execute_forward_typed_into<T: GpuStorage<E>>(
        &self,
        plan: &WgpuTransformPlan<X>,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed::<T>(plan, precision, input.len(), output.len())?;
        self.execute_typed_into(plan, input, output, false)
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
    pub fn execute_forward_leto_typed<T: GpuStorage<E> + Default>(
        &self,
        plan: &WgpuTransformPlan<X>,
        precision: PrecisionProfile,
        input: leto::ArrayView1<'_, T>,
    ) -> WgpuResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let input = apollo_leto_interop::view_cow(&input);
        let mut output =
            leto::Array::<T, leto::MnemosyneStorage<T>, 1>::zeros_mnemosyne([plan.output_len()]);
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

    /// Execute the normalized inverse transform with caller-owned typed
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    pub fn execute_inverse_typed_into<T: GpuStorage<E>>(
        &self,
        plan: &WgpuTransformPlan<X>,
        precision: PrecisionProfile,
        input: &[T],
        output: &mut [T],
    ) -> WgpuResult<()> {
        Self::validate_typed::<T>(plan, precision, input.len(), output.len())?;
        self.execute_typed_into(plan, input, output, true)
    }

    /// Execute the normalized inverse transform from typed Leto storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, precision-profile, or
    /// provider failure.
    pub fn execute_inverse_leto_typed<T: GpuStorage<E> + Default>(
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

    fn validate_typed<T: GpuStorage<E>>(
        plan: &WgpuTransformPlan<X>,
        precision: PrecisionProfile,
        input_len: usize,
        output_len: usize,
    ) -> WgpuResult<()> {
        let expected = T::PROFILE;
        if precision.storage != expected.storage || precision.compute != expected.compute {
            return Err(WgpuError::InvalidPrecisionProfile);
        }
        Self::validate_plan(plan)?;
        Self::require_len("typed input", input_len, plan.len())?;
        Self::require_len("typed output", output_len, plan.output_len())?;
        Ok(())
    }

    fn execute_typed_into<T: GpuStorage<E>>(
        &self,
        plan: &WgpuTransformPlan<X>,
        input: &[T],
        output: &mut [T],
        inverse: bool,
    ) -> WgpuResult<()> {
        let run = |represented: &[E], computed: &mut [E]| {
            if inverse {
                X::inverse_into(&self.device, plan.payload(), represented, computed)
            } else {
                X::forward_into(&self.device, plan.payload(), represented, computed)
            }
        };
        if let (Some(input), Some(output)) =
            (T::as_element_slice(input), T::as_element_slice_mut(output))
        {
            return run(input, output);
        }
        E::with_scratch(input.len(), output.len(), |represented, computed| {
            for (slot, value) in represented.iter_mut().zip(input.iter().copied()) {
                *slot = value.to_gpu();
            }
            run(represented, computed)?;
            for (slot, value) in output.iter_mut().zip(computed.iter().copied()) {
                *slot = T::from_gpu(value);
            }
            Ok(())
        })
    }
}
