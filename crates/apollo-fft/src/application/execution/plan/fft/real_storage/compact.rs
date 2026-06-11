//! `f16` real-storage implementation: `Complex32` spectrum, `f32` plan scalar.
//!
//! Reduced-precision storage with widened execution: f16 inputs are promoted to
//! f32 at the storage boundary so the plan stays monomorphized on `f32`.

use super::RealFftData;
use half::f16;
use num_complex::Complex32;

impl RealFftData for f16 {
    type PlanScalar = f32;

    /// Storage-boundary promotion `f16 → f32`; one widening per element at
    /// input, never inside plan arithmetic.
    #[inline]
    fn to_spectrum(self) -> Complex32 {
        Complex32::new(self.to_f32(), 0.0)
    }

    /// Output-boundary quantization `f32 → f16`; one narrowing per element.
    #[inline]
    fn from_spectrum(value: Complex32) -> Self {
        f16::from_f32(value.re)
    }
}
