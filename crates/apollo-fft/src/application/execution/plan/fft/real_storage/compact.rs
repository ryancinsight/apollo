//! `F16` real-storage implementation: `Complex32` spectrum, `f32` plan scalar.
//!
//! Reduced-precision storage with widened execution: F16 inputs are promoted to
//! f32 at the storage boundary so the plan stays monomorphized on `f32`.

use super::RealFftData;
use eunomia::{Complex32, F16};

impl RealFftData for F16 {
    type PlanScalar = f32;

    /// Storage-boundary promotion `F16 → f32`; one widening per element at
    /// input, never inside plan arithmetic.
    #[inline]
    fn to_spectrum(self) -> Complex32 {
        Complex32::new(self.to_f32(), 0.0)
    }

    /// Output-boundary quantization `f32 → F16`; one narrowing per element.
    #[inline]
    fn from_spectrum(value: Complex32) -> Self {
        F16::from_f32(value.re)
    }
}
