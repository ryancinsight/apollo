use crate::infrastructure::transport::gpu::SphericalPlan;
use crate::infrastructure::transport::gpu::{ShtWgpuPlan, WgpuCapabilities};

#[test]
fn capabilities_advertise_direct_complex_execution() {
    let capabilities = WgpuCapabilities::implemented(true);
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
    assert!(capabilities.supports_mixed_precision);
    assert_eq!(
        capabilities.default_precision_profile,
        apollo_fft::PrecisionProfile::LOW_PRECISION_F32
    );
}

#[test]
fn plan_preserves_grid_and_bandlimit() {
    let plan = ShtWgpuPlan::new(SphericalPlan::new(4, 5, 2));
    assert_eq!(plan.payload().latitudes(), 4);
    assert_eq!(plan.payload().longitudes(), 5);
    assert_eq!(plan.payload().max_degree(), 2);
    assert_eq!(plan.payload().sample_count(), 20);
    assert_eq!(plan.payload().mode_count(), 9);
    assert!(!plan.is_empty());
    assert!(ShtWgpuPlan::new(SphericalPlan::new(0, 5, 0)).is_empty());
    assert!(ShtWgpuPlan::new(SphericalPlan::new(4, 0, 0)).is_empty());
}
