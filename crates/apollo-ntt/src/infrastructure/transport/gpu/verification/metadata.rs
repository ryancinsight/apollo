use crate::infrastructure::transport::gpu::ResiduePlan;
use crate::infrastructure::transport::gpu::{
    NttWgpuBackend, NttWgpuPlan, WgpuCapabilities, WgpuError,
};
use crate::{DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};

#[test]
fn capabilities_reflect_full_kernel_surface() {
    let Ok(device) = hephaestus_wgpu::WgpuDevice::try_default("apollo-ntt-caps") else {
        return;
    };
    let capabilities = NttWgpuBackend::new(device).capabilities();
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
    assert!(!capabilities.supports_mixed_precision);
}

#[test]
fn capabilities_detected_without_device_clear_execution_flags() {
    let capabilities = WgpuCapabilities::detected(false);
    assert!(!capabilities.device_available);
    assert!(!capabilities.supports_forward);
    assert!(!capabilities.supports_inverse);
    assert!(!capabilities.supports_mixed_precision);
}

#[test]
fn plan_preserves_modular_configuration() {
    let plan = NttWgpuPlan::new(ResiduePlan::with_modulus(
        64,
        DEFAULT_MODULUS,
        DEFAULT_PRIMITIVE_ROOT,
    ));
    assert_eq!(plan.len(), 64);
    assert_eq!(plan.payload().modulus(), DEFAULT_MODULUS);
    assert_eq!(plan.payload().primitive_root(), DEFAULT_PRIMITIVE_ROOT);
    assert!(!NttWgpuPlan::new(ResiduePlan::with_modulus(
        64,
        DEFAULT_MODULUS,
        DEFAULT_PRIMITIVE_ROOT
    ))
    .is_empty());
    assert!(NttWgpuPlan::new(ResiduePlan::with_modulus(
        0,
        DEFAULT_MODULUS,
        DEFAULT_PRIMITIVE_ROOT
    ))
    .is_empty());
}

#[test]
fn unsupported_execution_error_identifies_operation() {
    let error = WgpuError::UnsupportedExecution {
        operation: "forward",
    };
    assert_eq!(
        error.to_string(),
        "forward is unsupported by the current WGPU capability set"
    );
}

#[test]
fn available_backend_reports_execution_capabilities() {
    let device = match hephaestus_wgpu::WgpuDevice::try_default("apollo-ntt-wgpu") {
        Ok(device) => device,
        Err(hephaestus_core::HephaestusError::AdapterUnavailable { .. }) => return,
        Err(error) => panic!("NTT GPU verification requires a working provider: {error}"),
    };
    let backend = NttWgpuBackend::new(device);
    let capabilities = backend.capabilities();
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
}

/// The shader's u32 modular add wraps for a modulus at or above 2^30, so
/// validation refuses those primes; 998244353 (below 2^30) is accepted.
#[test]
fn validation_refuses_moduli_the_shader_cannot_add() {
    assert!(ResiduePlan::with_modulus(2, 3_221_225_473, 5)
        .validate_field()
        .is_err_and(|error| error.to_string().contains("below 2^30")));
    assert!(ResiduePlan::with_modulus(2, 4_294_967_291, 2)
        .validate_field()
        .is_err());
    assert_eq!(
        ResiduePlan::with_modulus(2, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT)
            .validate_field()
            .expect("the default field is valid on the accelerator"),
        DEFAULT_MODULUS - 1
    );
}
