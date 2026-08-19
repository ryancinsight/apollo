use super::DctDstPlan;
use crate::domain::contracts::error::{DctDstError, DctDstResult};
use apollo_fft::{f16, CpuStorage, PrecisionProfile};
use mnemosyne::scratch::ScratchPool;

thread_local! {
    static TYPED_INPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static TYPED_OUTPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

fn with_f64_workspaces<R>(n: usize, f: impl FnOnce(&mut [f64], &mut [f64]) -> R) -> R {
    TYPED_INPUT64_SCRATCH.with(|in_pool| {
        in_pool.with_scratch(n, |input64| {
            TYPED_OUTPUT64_SCRATCH
                .with(|out_pool| out_pool.with_scratch(n, |output64| f(input64, output64)))
        })
    })
}

impl DctDstPlan {
    /// Execute the forward transform for `f64`, `f32`, or mixed `f16` storage.
    ///
    /// Lower storage profiles reuse the crate's authoritative `f64` transform
    /// and quantize once into the caller-owned output slice. This avoids
    /// precision-specific algorithm forks and preserves the DCT/DST theorem
    /// surface.
    pub fn forward_typed_into<T: RealTransformStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        T::forward_into(self, signal, output, profile)
    }

    /// Execute the forward transform over a typed Leto real-valued 1D view.
    pub fn forward_leto_typed<T: RealTransformStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> DctDstResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        let mut output = vec![T::from_cpu(0.0); self.len()];
        T::forward_into(self, &signal, &mut output, profile)?;
        Ok(apollo_leto_interop::try_array1_from_slice(&output)
            .expect("DCT/DST output length must match Leto output shape"))
    }

    /// Execute the inverse transform for `f64`, `f32`, or mixed `f16` storage.
    pub fn inverse_typed_into<T: RealTransformStorage>(
        &self,
        signal: &[T],
        output: &mut [T],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        T::inverse_into(self, signal, output, profile)
    }

    /// Execute the inverse transform over a typed Leto real-valued 1D view.
    pub fn inverse_leto_typed<T: RealTransformStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> DctDstResult<leto::Array<T, leto::MnemosyneStorage<T>, 1>> {
        let signal = apollo_leto_interop::view_cow(&signal);
        let mut output = vec![T::from_cpu(0.0); self.len()];
        T::inverse_into(self, &signal, &mut output, profile)?;
        Ok(apollo_leto_interop::try_array1_from_slice(&output)
            .expect("DCT/DST output length must match Leto output shape"))
    }
}

/// Real storage accepted by typed DCT/DST paths.
pub trait RealTransformStorage: CpuStorage {
    /// Execute forward transform into caller-owned storage.
    fn forward_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(DctDstError::LengthMismatch);
        }
        with_f64_workspaces(plan.len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(signal.iter()) {
                *slot = value.to_cpu();
            }
            plan.forward_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_cpu(value);
            }
            Ok(())
        })
    }

    /// Execute inverse transform into caller-owned storage.
    fn inverse_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        if signal.len() != plan.len() || output.len() != plan.len() {
            return Err(DctDstError::LengthMismatch);
        }
        with_f64_workspaces(plan.len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(signal.iter()) {
                *slot = value.to_cpu();
            }
            plan.inverse_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = Self::from_cpu(value);
            }
            Ok(())
        })
    }
}

impl RealTransformStorage for f64 {
    fn forward_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.forward_into(signal, output)
    }

    fn inverse_into(
        plan: &DctDstPlan,
        signal: &[Self],
        output: &mut [Self],
        profile: PrecisionProfile,
    ) -> DctDstResult<()> {
        validate_profile(profile, Self::PROFILE)?;
        plan.inverse_into(signal, output)
    }
}

impl RealTransformStorage for f32 {}

impl RealTransformStorage for f16 {}

mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for apollo_fft::f16 {}
}

/// Storage whose compute profile is the concrete `f32` accelerator contract.
///
/// This sealed capability admits native `f32` storage and explicit mixed
/// `f16`/`f32` storage. High-accuracy `f64` storage is intentionally excluded:
/// accepting it would silently narrow the public typed API to the WGPU kernel's
/// concrete arithmetic.
///
/// ```compile_fail
/// use apollo_dctdst::RealTransformGpuStorage;
///
/// fn require_gpu_storage<T: RealTransformGpuStorage>() {}
/// require_gpu_storage::<f64>();
/// ```
pub trait RealTransformGpuStorage: RealTransformStorage + sealed::Sealed {
    /// Convert storage into the concrete `f32` accelerator contract.
    fn to_gpu(self) -> f32;

    /// Convert a concrete `f32` accelerator result back to storage.
    fn from_gpu(value: f32) -> Self;
}

impl RealTransformGpuStorage for f32 {
    fn to_gpu(self) -> f32 {
        self
    }

    fn from_gpu(value: f32) -> Self {
        value
    }
}

impl RealTransformGpuStorage for f16 {
    fn to_gpu(self) -> f32 {
        self.to_f32()
    }

    fn from_gpu(value: f32) -> Self {
        f16::from_f32(value)
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> DctDstResult<()> {
    if actual.matches_storage_and_compute(expected) {
        Ok(())
    } else {
        Err(DctDstError::PrecisionMismatch)
    }
}
