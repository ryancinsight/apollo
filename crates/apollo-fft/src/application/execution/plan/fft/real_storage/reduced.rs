//! `f32` real-storage implementation: native `Complex32` spectrum, `f32` plan scalar.

use super::RealFftData;
use num_complex::Complex32;

impl RealFftData for f32 {
    type PlanScalar = f32;

    #[inline]
    fn to_spectrum(self) -> Complex32 {
        Complex32::new(self, 0.0)
    }

    #[inline]
    fn from_spectrum(value: Complex32) -> Self {
        value.re
    }
}
