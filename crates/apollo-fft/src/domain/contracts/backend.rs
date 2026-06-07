//! Backend traits and capability descriptors.

use crate::domain::contracts::error::ApolloResult;
use crate::domain::metadata::precision::{BackendKind, Normalization, PrecisionProfile};
use crate::domain::metadata::shape::{Shape1D, Shape2D, Shape3D};

/// Capability descriptor advertised by a backend.
///
/// All fields are `Copy` and the profile list is a `&'static` reference, making
/// this struct zero-allocation on every `capabilities()` call. The reference
/// lifetime ties the struct to the compile-time constant profiles defined
/// by each backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// Backend family.
    pub kind: BackendKind,
    /// Normalization convention implemented by the backend.
    pub normalization: Normalization,
    /// Whether the backend can plan 1D transforms.
    pub supports_1d: bool,
    /// Whether the backend can plan 2D transforms.
    pub supports_2d: bool,
    /// Whether the backend can plan 3D transforms.
    pub supports_3d: bool,
    /// Whether the backend supports real-to-complex half-spectrum transforms.
    pub supports_real_to_complex: bool,
    /// Whether the backend implements at least one mixed-precision profile.
    pub supports_mixed_precision: bool,
    /// Default precision profile selected when a caller does not request one.
    pub default_precision_profile: PrecisionProfile,
    /// Precision profiles truthfully implemented by this backend.
    /// Uses `&'static` reference to avoid heap allocation on every call.
    pub supported_precision_profiles: &'static [PrecisionProfile],
}

/// Backend trait used by consumers that want backend selection via dependency inversion.
pub trait FftBackend {
    /// 1D plan type returned by the backend.
    type Plan1D;
    /// 2D plan type returned by the backend.
    type Plan2D;
    /// 3D plan type returned by the backend.
    type Plan3D;

    /// Identify the backend family.
    fn backend_kind(&self) -> BackendKind;

    /// Report backend capabilities.
    fn capabilities(&self) -> BackendCapabilities;

    /// Construct a 1D plan.
    fn plan_1d(&self, shape: Shape1D) -> ApolloResult<Self::Plan1D>;

    /// Construct a 2D plan.
    fn plan_2d(&self, shape: Shape2D) -> ApolloResult<Self::Plan2D>;

    /// Construct a 3D plan.
    fn plan_3d(&self, shape: Shape3D) -> ApolloResult<Self::Plan3D>;
}

impl BackendCapabilities {
    /// Returns true when at least one transform dimensionality is supported.
    #[must_use]
    pub const fn has_any_capability(&self) -> bool {
        self.supports_1d || self.supports_2d || self.supports_3d
    }

    /// CPU backend capabilities as a compile-time constant.
    pub const CPU: Self = Self {
        kind: BackendKind::Cpu,
        normalization: Normalization::FftwCompatible,
        supports_1d: true,
        supports_2d: true,
        supports_3d: true,
        supports_real_to_complex: true,
        supports_mixed_precision: true,
        default_precision_profile: PrecisionProfile::HIGH_ACCURACY_F64,
        supported_precision_profiles: &[
            PrecisionProfile::HIGH_ACCURACY_F64,
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        ],
    };

    /// WGPU backend capabilities as a compile-time constant.
    pub const WGPU: Self = Self {
        kind: BackendKind::Wgpu,
        normalization: Normalization::FftwCompatible,
        supports_1d: false,
        supports_2d: false,
        supports_3d: true,
        supports_real_to_complex: false,
        supports_mixed_precision: true,
        default_precision_profile: PrecisionProfile::LOW_PRECISION_F32,
        supported_precision_profiles: &[
            PrecisionProfile::LOW_PRECISION_F32,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        ],
    };
}
