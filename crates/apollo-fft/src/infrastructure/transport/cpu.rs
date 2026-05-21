//! CPU backend adapter.
//
//! The CPU backend is a thin capability surface over Apollo-owned FFT plan
//! implementations. It does not depend on external FFT engines in production.

use crate::application::execution::plan::fft::dimension_1d::FftPlan1D;
use crate::application::execution::plan::fft::dimension_2d::FftPlan2D;
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use crate::domain::contracts::backend::{BackendCapabilities, FftBackend};
use crate::domain::contracts::error::ApolloResult;
use crate::domain::metadata::precision::{BackendKind, Normalization, PrecisionProfile};
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};

/// CPU backend backed by Apollo-owned dense FFT implementations.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl FftBackend for CpuBackend {
    type Plan1D = FftPlan1D<f64>;
    type Plan2D = FftPlan2D<f64>;
    type Plan3D = FftPlan3D<f64>;

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Cpu,
            normalization: Normalization::FftwCompatible,
            supports_1d: true,
            supports_2d: true,
            supports_3d: true,
            supports_real_to_complex: true,
            supports_mixed_precision: true,
            default_precision_profile: PrecisionProfile::HIGH_ACCURACY_F64,
            supported_precision_profiles: vec![
                PrecisionProfile::HIGH_ACCURACY_F64,
                PrecisionProfile::LOW_PRECISION_F32,
                PrecisionProfile::MIXED_PRECISION_F16_F32,
            ],
        }
    }

    fn plan_1d(&self, shape: Shape1D) -> ApolloResult<Self::Plan1D> {
        Ok(FftPlan1D::<f64>::new(shape))
    }

    fn plan_2d(&self, shape: Shape2D) -> ApolloResult<Self::Plan2D> {
        Ok(FftPlan2D::<f64>::new(shape))
    }

    fn plan_3d(&self, shape: Shape3D) -> ApolloResult<Self::Plan3D> {
        Ok(FftPlan3D::<f64>::new(shape))
    }
}

#[cfg(test)]
mod tests {
    use super::CpuBackend;
    use crate::domain::contracts::backend::FftBackend;
    use crate::domain::metadata::precision::BackendKind;
    use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};
    use ndarray::{Array1, Array2, Array3};
    use num_complex::Complex64;

    #[test]
    fn default_produces_cpu_backend() {
        let backend = CpuBackend;
        assert_eq!(backend.backend_kind(), BackendKind::Cpu);
    }

    #[test]
    fn capabilities_has_any_capability() {
        let backend = CpuBackend;
        let caps = backend.capabilities();
        assert!(
            caps.has_any_capability(),
            "CpuBackend must advertise at least one capability"
        );
        assert_eq!(caps.kind, BackendKind::Cpu);
    }

    #[test]
    fn plan_1d_correct_roundtrip() {
        let backend = CpuBackend;
        let shape = Shape1D::new(8).expect("valid shape");
        let plan = backend.plan_1d(shape).expect("plan_1d succeeded");
        let input = Array1::from_iter((0..8usize).map(|i| (i as f64 * 0.7).sin()));
        let complex = input.mapv(|value| Complex64::new(value, 0.0));
        let spectrum = plan.forward_complex(&complex);
        let recovered = plan.inverse_complex(&spectrum).mapv(|value| value.re);
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-10, "1D roundtrip err={err:.2e}");
        }
    }

    #[test]
    fn plan_2d_correct_roundtrip() {
        let backend = CpuBackend;
        let shape = Shape2D::new(4, 4).expect("valid shape");
        let plan = backend.plan_2d(shape).expect("plan_2d succeeded");
        let input = Array2::from_shape_fn((4, 4), |(i, j)| (i as f64 * 0.3 + j as f64 * 0.5).sin());
        let mut spectrum = input.mapv(|value| Complex64::new(value, 0.0));
        plan.forward_complex_inplace(&mut spectrum);
        plan.inverse_complex_inplace(&mut spectrum);
        let recovered = spectrum.mapv(|value| value.re);
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-10, "2D roundtrip err={err:.2e}");
        }
    }

    #[test]
    fn plan_3d_correct_roundtrip() {
        let backend = CpuBackend;
        let shape = Shape3D::new(4, 4, 4).expect("valid shape");
        let plan = backend.plan_3d(shape).expect("plan_3d succeeded");
        let input = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| {
            (i as f64 * 0.3 + j as f64 * 0.2 + k as f64 * 0.5).sin()
        });
        let mut spectrum = input.mapv(|value| Complex64::new(value, 0.0));
        plan.forward_complex_inplace(&mut spectrum);
        plan.inverse_complex_inplace(&mut spectrum);
        let recovered = spectrum.mapv(|value| value.re);
        for (a, b) in input.iter().zip(recovered.iter()) {
            let err = (a - b).abs();
            assert!(err < 1e-10, "3D roundtrip err={err:.2e}");
        }
    }
}
