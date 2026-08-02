//! Shared typed-storage contracts for GPU transforms (ADR 0037).
//!
//! [`GpuElement`] names the concrete accelerator element families (`f32`
//! for real transforms, [`Complex32`] for complex ones) and owns their
//! thread-local scratch pools; [`GpuStorage`] maps host storage types
//! onto an element (identity for the native type, promote/quantize for
//! reduced-precision forms).

use mnemosyne::scratch::ScratchPool;

use crate::{f16, Complex32, PrecisionProfile};

mod sealed {
    pub trait SealedElement {}

    impl SealedElement for f32 {}
    impl SealedElement for crate::Complex32 {}

    pub trait SealedStorage {}

    impl SealedStorage for f32 {}
    impl SealedStorage for crate::f16 {}
    impl SealedStorage for crate::Complex32 {}
    impl SealedStorage for [crate::f16; 2] {}
}

thread_local! {
    static REAL_INPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
    static REAL_OUTPUT_SCRATCH: ScratchPool<f32> = const { ScratchPool::new() };
    static COMPLEX_INPUT_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
    static COMPLEX_OUTPUT_SCRATCH: ScratchPool<Complex32> = const { ScratchPool::new() };
}

/// Concrete accelerator element of a GPU transform contract.
///
/// Sealed to the element families the WGSL kernels execute: `f32` and
/// [`Complex32`]. Each element owns thread-local input/output scratch
/// pools so reduced-precision typed dispatch stays allocation-free.
pub trait GpuElement:
    bytemuck::Pod + Default + core::fmt::Debug + Send + Sync + 'static + sealed::SealedElement
{
    /// Run `body` over reused input and output scratch of the given
    /// lengths.
    fn with_scratch<R>(
        input_len: usize,
        output_len: usize,
        body: impl FnOnce(&mut [Self], &mut [Self]) -> R,
    ) -> R;
}

impl GpuElement for f32 {
    fn with_scratch<R>(
        input_len: usize,
        output_len: usize,
        body: impl FnOnce(&mut [Self], &mut [Self]) -> R,
    ) -> R {
        REAL_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input_len, |input| {
                REAL_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output_len, |output| body(input, output))
                })
            })
        })
    }
}

impl GpuElement for Complex32 {
    fn with_scratch<R>(
        input_len: usize,
        output_len: usize,
        body: impl FnOnce(&mut [Self], &mut [Self]) -> R,
    ) -> R {
        COMPLEX_INPUT_SCRATCH.with(|input_pool| {
            input_pool.with_scratch(input_len, |input| {
                COMPLEX_OUTPUT_SCRATCH.with(|output_pool| {
                    output_pool.with_scratch(output_len, |output| body(input, output))
                })
            })
        })
    }
}

/// Host storage admitted by an element family's typed GPU paths.
///
/// The default element parameter keeps the dominant real case
/// annotation-free: `T: GpuStorage` reads as storage over `f32`. The
/// sealed set intentionally excludes `f64`-family storage so GPU typed
/// APIs cannot silently narrow high-accuracy storage.
///
/// ```compile_fail
/// use apollo_fft::GpuStorage;
///
/// fn require_gpu_storage<T: GpuStorage>() {}
/// require_gpu_storage::<f64>();
/// ```
pub trait GpuStorage<E: GpuElement = f32>:
    Copy + Send + Sync + 'static + sealed::SealedStorage
{
    /// Precision profile this storage declares.
    const PROFILE: PrecisionProfile;

    /// Convert storage into the concrete accelerator element.
    fn to_gpu(self) -> E;

    /// Convert a concrete accelerator element back to storage.
    fn from_gpu(value: E) -> Self;

    /// View slice as the accelerator element if layout is identical.
    #[inline]
    fn as_element_slice(slice: &[Self]) -> Option<&[E]> {
        let _ = slice;
        None
    }

    /// View mutable slice as the accelerator element if layout is
    /// identical.
    #[inline]
    fn as_element_slice_mut(slice: &mut [Self]) -> Option<&mut [E]> {
        let _ = slice;
        None
    }
}

impl GpuStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_gpu(self) -> f32 {
        self
    }

    fn from_gpu(value: f32) -> Self {
        value
    }

    #[inline]
    fn as_element_slice(slice: &[Self]) -> Option<&[f32]> {
        Some(slice)
    }

    #[inline]
    fn as_element_slice_mut(slice: &mut [Self]) -> Option<&mut [f32]> {
        Some(slice)
    }
}

impl GpuStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_gpu(self) -> f32 {
        self.to_f32()
    }

    fn from_gpu(value: f32) -> Self {
        f16::from_f32(value)
    }
}

impl GpuStorage<Complex32> for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_gpu(self) -> Complex32 {
        self
    }

    fn from_gpu(value: Complex32) -> Self {
        value
    }

    #[inline]
    fn as_element_slice(slice: &[Self]) -> Option<&[Complex32]> {
        Some(slice)
    }

    #[inline]
    fn as_element_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        Some(slice)
    }
}

impl GpuStorage<Complex32> for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_gpu(self) -> Complex32 {
        Complex32::new(self[0].to_f32(), self[1].to_f32())
    }

    fn from_gpu(value: Complex32) -> Self {
        [f16::from_f32(value.re), f16::from_f32(value.im)]
    }
}
