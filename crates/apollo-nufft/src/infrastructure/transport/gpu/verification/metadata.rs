use crate::{
    infrastructure::transport::gpu::{
        NufftWgpuCapabilities, NufftWgpuError, NufftWgpuPlan1D, NufftWgpuPlan3D,
    },
    UniformDomain1D, UniformGrid3D,
};
use apollo_fft::PrecisionProfile;

#[test]
fn capabilities_advertise_all_direct_execution() {
    let capabilities = NufftWgpuCapabilities::direct_all(true);
    assert!(capabilities.device_available);
    assert!(capabilities.supports_type1_1d);
    assert!(capabilities.supports_type2_1d);
    assert!(capabilities.supports_type1_3d);
    assert!(capabilities.supports_type2_3d);
    assert!(!capabilities.supports_fast_type1_1d);
    assert!(!capabilities.supports_fast_type2_1d);
    assert!(!capabilities.supports_mixed_precision);
    assert_eq!(
        capabilities.default_precision_profile,
        PrecisionProfile::LOW_PRECISION_F32
    );
}

#[test]
fn capabilities_advertise_all_fast_when_enabled() {
    let capabilities = NufftWgpuCapabilities::direct_all_fast_all(true);
    assert!(capabilities.device_available);
    assert!(capabilities.supports_type1_1d);
    assert!(capabilities.supports_type2_1d);
    assert!(capabilities.supports_type1_3d);
    assert!(capabilities.supports_type2_3d);
    assert!(capabilities.supports_fast_type1_1d);
    assert!(capabilities.supports_fast_type2_1d);
    assert!(capabilities.supports_fast_type1_3d);
    assert!(capabilities.supports_fast_type2_3d);
    assert!(capabilities.supports_mixed_precision);
    assert_eq!(
        capabilities.default_precision_profile,
        PrecisionProfile::LOW_PRECISION_F32
    );
}

#[test]
fn plan_1d_preserves_validated_metadata() {
    let domain = UniformDomain1D::new(16, 0.25).expect("domain");
    let plan = NufftWgpuPlan1D::new(domain, 2, 6);
    assert_eq!(plan.domain(), domain);
    assert_eq!(plan.oversampling(), 2);
    assert_eq!(plan.kernel_width(), 6);
}

#[test]
fn plan_3d_preserves_validated_metadata() {
    let grid = UniformGrid3D::new(4, 8, 16, 0.1, 0.2, 0.3).expect("grid");
    let plan = NufftWgpuPlan3D::new(grid, 2, 6);
    assert_eq!(plan.grid(), grid);
    assert_eq!(plan.oversampling(), 2);
    assert_eq!(plan.kernel_width(), 6);
}

#[test]
fn unsupported_execution_error_identifies_operation() {
    let error = NufftWgpuError::UnsupportedExecution {
        operation: "type2_1d",
    };
    assert_eq!(
        error.to_string(),
        "type2_1d is unsupported by the current apollo-nufft-wgpu capability set"
    );
}
