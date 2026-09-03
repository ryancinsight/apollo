//! Shared CPU-tier storage vocabulary for the transform crates.
//!
//! [`CpuElement`](crate::CpuElement) names the owner arithmetic element families (`f64`
//! for real transforms, [`Complex64`](eunomia::Complex64) for complex ones) and owns their
//! thread-local scratch pools; [`CpuStorage`](crate::CpuStorage) maps host storage types
//! onto an element (identity for the native type, widen/narrow for the
//! represented forms). The design mirrors the GPU-tier
//! `GpuElement`/`GpuStorage` pair one precision tier up: transform
//! crates keep their plan-coupled dispatch traits and derive the
//! conversion ladder from this vocabulary instead of re-implementing
//! it per crate.

use eunomia::{Complex32, Complex64, F16};
use mnemosyne::scratch::ScratchPool;

use crate::domain::metadata::precision::PrecisionProfile;
mod sealed {
    pub trait SealedElement {}

    impl SealedElement for f64 {}
    impl SealedElement for eunomia::Complex64 {}

    pub trait SealedStorage {}

    impl SealedStorage for f64 {}
    impl SealedStorage for f32 {}
    impl SealedStorage for eunomia::F16 {}
    impl SealedStorage for eunomia::Complex64 {}
    impl SealedStorage for eunomia::Complex32 {}
    impl SealedStorage for [eunomia::F16; 2] {}
}

thread_local! {
    static REAL64_INPUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static REAL64_OUTPUT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static COMPLEX64_INPUT_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    static COMPLEX64_OUTPUT_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Owner arithmetic element of a CPU transform contract.
///
/// Sealed to the element families the CPU plans compute in: `f64` and
/// [`Complex64`]. Each element owns thread-local input/output scratch
/// pools so reduced-precision typed dispatch stays allocation-free.
pub trait CpuElement: Copy + Send + Sync + 'static + sealed::SealedElement {
    /// Run `body` over reused input scratch of the given length.
    fn with_input_scratch<R>(len: usize, body: impl FnOnce(&mut [Self]) -> R) -> R;

    /// Run `body` over reused output scratch of the given length.
    fn with_output_scratch<R>(len: usize, body: impl FnOnce(&mut [Self]) -> R) -> R;

    /// Return the calling thread's (input, output) scratch capacities.
    ///
    /// Reuse tests observe that repeated dispatches of one size leave
    /// the capacities unchanged.
    fn scratch_capacities() -> (usize, usize);

    /// Run `body` over reused input and output scratch of the given
    /// lengths.
    fn with_scratch<R>(
        input_len: usize,
        output_len: usize,
        body: impl FnOnce(&mut [Self], &mut [Self]) -> R,
    ) -> R {
        Self::with_input_scratch(input_len, |input| {
            Self::with_output_scratch(output_len, |output| body(input, output))
        })
    }
}

impl CpuElement for f64 {
    fn scratch_capacities() -> (usize, usize) {
        REAL64_INPUT_SCRATCH.with(|input| {
            REAL64_OUTPUT_SCRATCH.with(|output| (input.capacity(), output.capacity()))
        })
    }

    fn with_input_scratch<R>(len: usize, body: impl FnOnce(&mut [Self]) -> R) -> R {
        REAL64_INPUT_SCRATCH.with(|pool| pool.with_scratch(len, body))
    }

    fn with_output_scratch<R>(len: usize, body: impl FnOnce(&mut [Self]) -> R) -> R {
        REAL64_OUTPUT_SCRATCH.with(|pool| pool.with_scratch(len, body))
    }
}

impl CpuElement for Complex64 {
    fn scratch_capacities() -> (usize, usize) {
        COMPLEX64_INPUT_SCRATCH.with(|input| {
            COMPLEX64_OUTPUT_SCRATCH.with(|output| (input.capacity(), output.capacity()))
        })
    }

    fn with_input_scratch<R>(len: usize, body: impl FnOnce(&mut [Self]) -> R) -> R {
        COMPLEX64_INPUT_SCRATCH.with(|pool| pool.with_scratch(len, body))
    }

    fn with_output_scratch<R>(len: usize, body: impl FnOnce(&mut [Self]) -> R) -> R {
        COMPLEX64_OUTPUT_SCRATCH.with(|pool| pool.with_scratch(len, body))
    }
}

/// Host storage admitted by an element family's CPU typed paths.
///
/// The default element parameter keeps the dominant real case
/// annotation-free: `T: CpuStorage` reads as storage over `f64`.
/// Transform crates bound their plan-coupled dispatch traits on this
/// vocabulary instead of re-declaring the conversion ladder.
pub trait CpuStorage<E: CpuElement = f64>:
    Copy + Send + Sync + 'static + sealed::SealedStorage
{
    /// Precision profile this storage declares.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the owner arithmetic element.
    fn to_cpu(self) -> E;

    /// Convert an owner arithmetic result back to storage.
    fn from_cpu(value: E) -> Self;
}

impl CpuStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_cpu(self) -> f64 {
        self
    }

    fn from_cpu(value: f64) -> Self {
        value
    }
}

impl CpuStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_cpu(self) -> f64 {
        f64::from(self)
    }

    fn from_cpu(value: f64) -> Self {
        value as f32
    }
}

impl CpuStorage for F16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_cpu(self) -> f64 {
        f64::from(self.to_f32())
    }

    fn from_cpu(value: f64) -> Self {
        F16::from_f32(value as f32)
    }
}

impl CpuStorage<Complex64> for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_cpu(self) -> Complex64 {
        self
    }

    fn from_cpu(value: Complex64) -> Self {
        value
    }
}

impl CpuStorage<Complex64> for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_cpu(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }

    fn from_cpu(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
}

impl CpuStorage<Complex64> for [F16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_cpu(self) -> Complex64 {
        Complex64::new(f64::from(self[0].to_f32()), f64::from(self[1].to_f32()))
    }

    fn from_cpu(value: Complex64) -> Self {
        [
            F16::from_f32(value.re as f32),
            F16::from_f32(value.im as f32),
        ]
    }
}
