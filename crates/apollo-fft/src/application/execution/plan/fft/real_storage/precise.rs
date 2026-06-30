//! `f64` real-storage implementation: native `Complex64` spectrum, `f64` plan scalar.

use super::RealFftData;
use eunomia::Complex64;

impl RealFftData for f64 {
    type PlanScalar = f64;

    #[inline]
    fn to_spectrum(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }

    #[inline]
    fn from_spectrum(value: Complex64) -> Self {
        value.re
    }
}
