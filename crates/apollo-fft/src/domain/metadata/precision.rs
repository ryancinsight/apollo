//! Shared precision and backend-identification types.

use serde::{Deserialize, Serialize};

/// Supported backend families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    /// CPU backend backed by Apollo-owned dense FFT implementations.
    Cpu,
    /// WGPU backend.
    Wgpu,
    /// CUDA backend through the typed Hephaestus provider boundary.
    Cuda,
}

/// Apollo's normalization convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Normalization {
    /// Forward unnormalized, inverse normalized by the transformed volume.
    FftwCompatible,
}

/// Storage precision used by an execution profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StoragePrecision {
    /// 16-bit floating point storage.
    F16,
    /// 32-bit floating point storage.
    F32,
    /// 64-bit floating point storage.
    F64,
}

/// Arithmetic precision used internally by an execution profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputePrecision {
    /// Mixed `F16` storage with `f32` compute.
    MixedF16F32,
    /// 32-bit floating point compute.
    F32,
    /// 64-bit floating point compute.
    F64,
}

/// High-level precision intent selected by a caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionMode {
    /// Highest-accuracy backend path.
    HighAccuracy,
    /// Explicit mixed-precision execution.
    MixedPrecision,
    /// Explicit low-precision execution.
    LowPrecision,
}

/// Precision contract advertised by a backend or plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrecisionProfile {
    /// Precision mode intent.
    pub mode: PrecisionMode,
    /// Input/output storage precision.
    pub storage: StoragePrecision,
    /// Internal arithmetic precision.
    pub compute: ComputePrecision,
}

impl PrecisionProfile {
    /// Whether storage and arithmetic precision match `expected`.
    ///
    /// This intentionally ignores the high-level mode intent: two plans may
    /// select different modes while executing with the same storage and compute
    /// representation.
    #[must_use]
    #[inline]
    pub fn matches_storage_and_compute(self, expected: Self) -> bool {
        self.storage == expected.storage && self.compute == expected.compute
    }

    /// Fail with `mismatch()` unless storage and arithmetic precision match `expected`.
    ///
    /// Mode intent is ignored, matching [`matches_storage_and_compute`](Self::matches_storage_and_compute).
    /// The error is generic so every transform crate calls this directly instead
    /// of cloning a per-crate validator.
    #[inline]
    pub fn ensure_matches<E>(self, expected: Self, mismatch: impl FnOnce() -> E) -> Result<(), E> {
        if self.matches_storage_and_compute(expected) {
            Ok(())
        } else {
            Err(mismatch())
        }
    }

    /// CPU-style high-accuracy profile.
    pub const HIGH_ACCURACY_F64: Self = Self {
        mode: PrecisionMode::HighAccuracy,
        storage: StoragePrecision::F64,
        compute: ComputePrecision::F64,
    };

    /// Explicit low-precision f32 profile.
    pub const LOW_PRECISION_F32: Self = Self {
        mode: PrecisionMode::LowPrecision,
        storage: StoragePrecision::F32,
        compute: ComputePrecision::F32,
    };

    /// Mixed-precision profile with `F16` storage and `f32` compute.
    pub const MIXED_PRECISION_F16_F32: Self = Self {
        mode: PrecisionMode::MixedPrecision,
        storage: StoragePrecision::F16,
        compute: ComputePrecision::MixedF16F32,
    };

    /// Low-precision-intent profile with `f32` storage and `f64` compute.
    ///
    /// This is the truthful descriptor for the CPU typed paths that widen
    /// `f32`/`Complex32` storage through the owner `f64`/`Complex64`
    /// arithmetic (`CpuStorage`): the mode records the caller's intent,
    /// the compute field the arithmetic actually executed. Native `f32`
    /// paths (FFT, FWHT, GPU) keep `LOW_PRECISION_F32` and reject this.
    pub const F32_STORAGE_F64_COMPUTE: Self = Self {
        mode: PrecisionMode::LowPrecision,
        storage: StoragePrecision::F32,
        compute: ComputePrecision::F64,
    };

    /// Mixed-precision-intent profile with `F16` storage and `f64` compute.
    ///
    /// The `F16`/`[F16; 2]` counterpart of `F32_STORAGE_F64_COMPUTE` for
    /// the same widening CPU typed paths. Native mixed paths keep
    /// `MIXED_PRECISION_F16_F32` and reject this.
    pub const F16_STORAGE_F64_COMPUTE: Self = Self {
        mode: PrecisionMode::MixedPrecision,
        storage: StoragePrecision::F16,
        compute: ComputePrecision::F64,
    };
}
