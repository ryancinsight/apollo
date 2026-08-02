//! Shared WGPU transform transport scaffold (ADR 0037).
//!
//! One generic execution layer replaces the per-transform copies of the
//! `infrastructure/transport/gpu/` scaffold across the Apollo transform
//! crates. A transform adopts it by implementing [`GpuTransformExecutor`]
//! on its zero-sized kernel marker and aliasing the plan and backend
//! types under its domain names; the transform keeps its kernel (shader
//! sources, parameter structs, pass sequence) and mathematical contract —
//! which is what actually varies.

mod backend;
mod capabilities;
mod error;
mod plan;
mod storage;

pub use backend::{GpuTransformExecutor, WgpuTransformBackend};
pub use capabilities::WgpuCapabilities;
pub use error::{WgpuError, WgpuResult};
pub use plan::WgpuTransformPlan;
pub use storage::{GpuElement, GpuStorage};
