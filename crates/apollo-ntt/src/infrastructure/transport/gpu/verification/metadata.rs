use crate::infrastructure::transport::gpu::{
    NttWgpuBackend, NttWgpuPlan, WgpuCapabilities, WgpuError,
};
use crate::{DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT};

#[test]
fn capabilities_reflect_full_kernel_surface() {
    let capabilities = WgpuCapabilities::full(true);
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
    assert!(!capabilities.supports_mixed_precision);
    assert!(capabilities.supports_quantized_storage);
}

#[test]
fn capabilities_detected_without_device_clear_execution_flags() {
    let capabilities = WgpuCapabilities::detected(false);
    assert!(!capabilities.device_available);
    assert!(!capabilities.supports_forward);
    assert!(!capabilities.supports_inverse);
    assert!(!capabilities.supports_mixed_precision);
    assert!(!capabilities.supports_quantized_storage);
}

#[test]
fn plan_preserves_modular_configuration() {
    let plan = NttWgpuPlan::new(64, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT);
    assert_eq!(plan.len(), 64);
    assert_eq!(plan.modulus(), DEFAULT_MODULUS);
    assert_eq!(plan.primitive_root(), DEFAULT_PRIMITIVE_ROOT);
    assert!(!NttWgpuPlan::new(64, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT).is_empty());
    assert!(NttWgpuPlan::new(0, DEFAULT_MODULUS, DEFAULT_PRIMITIVE_ROOT).is_empty());
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
    let Ok(backend) = NttWgpuBackend::try_default() else {
        return;
    };
    let capabilities = backend.capabilities();
    assert!(capabilities.device_available);
    assert!(capabilities.supports_forward);
    assert!(capabilities.supports_inverse);
    assert!(capabilities.supports_quantized_storage);
}
