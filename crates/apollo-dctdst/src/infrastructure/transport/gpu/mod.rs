#![warn(missing_docs)]
//! WGPU backend boundary for Apollo DCT/DST.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the DCT/DST kernel, its domain names,
//! and the separable multi-dimensional surface. The 1-D real-to-real
//! transforms instantiate the scaffold; the batched 2-D/3-D separable
//! passes extend it through [`SeparableExecution`], since their operands
//! are `n^d` fields rather than plan-length slices.

/// Infrastructure boundary for the DCT/DST kernel.
pub mod infrastructure;
#[cfg(test)]
pub(crate) mod verification;

use apollo_fft::WgpuTransformPlan as Plan;

use infrastructure::kernel::{forward_mode, inverse_mode_scale, FiberLayout};

pub use crate::RealTransformKind;
pub use apollo_fft::{WgpuCapabilities, WgpuError, WgpuResult};
pub use infrastructure::kernel::DctGpuKernel;
pub use leto::{Array2, Array3};

/// Plan payload for a real-to-real transform: logical length and the
/// transform family member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealTransformPlan {
    len: usize,
    kind: RealTransformKind,
}

impl RealTransformPlan {
    /// Create a real-transform plan payload.
    #[must_use]
    pub const fn new(len: usize, kind: RealTransformKind) -> Self {
        Self { len, kind }
    }

    /// Return the logical transform length carried by this payload.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return whether the payload carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Return the transform family member carried by this payload.
    #[must_use]
    pub const fn kind(self) -> RealTransformKind {
        self.kind
    }
}

/// Metadata-preserving WGPU plan descriptor.
pub type DctDstWgpuPlan = apollo_fft::WgpuTransformPlan<DctGpuKernel>;

/// WGPU backend descriptor.
pub type DctDstWgpuBackend = apollo_fft::WgpuTransformBackend<DctGpuKernel>;

/// Separable multi-dimensional surface of the DCT/DST backend.
///
/// Row-major flat `n^d` host buffers in and out. Every axis pass runs
/// on-device via one batched dispatch each, so the field is uploaded
/// once and downloaded once; the inverse folds its per-axis
/// normalization into a single `scale.powi(d)`.
pub trait SeparableExecution {
    /// Execute the unnormalized 2D separable forward transform.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or provider failure.
    fn execute_forward_2d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>>;

    /// Execute the unnormalized 2D separable forward transform from a
    /// Leto view. Leto appears only at this CPU-GPU seam.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or provider failure.
    fn execute_forward_2d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>>;

    /// Execute the unnormalized 3D separable forward transform.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or provider failure.
    fn execute_forward_3d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>>;

    /// Execute the unnormalized 3D separable forward transform from a
    /// Leto view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or provider failure.
    fn execute_forward_3d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView3<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>>;

    /// Execute the normalized 2D separable inverse transform.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or provider failure.
    fn execute_inverse_2d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>>;

    /// Execute the normalized 2D separable inverse transform from a
    /// Leto view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or provider failure.
    fn execute_inverse_2d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>>;

    /// Execute the normalized 3D separable inverse transform.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or provider failure.
    fn execute_inverse_3d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>>;

    /// Execute the normalized 3D separable inverse transform from a
    /// Leto view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, shape-mismatch, or provider failure.
    fn execute_inverse_3d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView3<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>>;
}

impl SeparableExecution for DctDstWgpuBackend {
    fn execute_forward_2d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        let passes_mode = separable_gate(plan, input.len(), 2)?;
        let n = plan.len();
        let passes = [
            (passes_mode, FiberLayout::axis(n, 2, 1)?),
            (passes_mode, FiberLayout::axis(n, 2, 0)?),
        ];
        let mut output = vec![0.0_f32; input.len()];
        DctGpuKernel::execute_separable_into(self.device(), input, &mut output, &passes, 1.0)?;
        Ok(output)
    }

    fn execute_forward_2d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let n = plan.len();
        let flat: Vec<f32> = input.iter().copied().collect();
        let result = self.execute_forward_2d(plan, &flat)?;
        leto::Array::from_mnemosyne_vec([n, n], result).map_err(|_| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto DCT/DST 2D output".to_string(),
        })
    }

    fn execute_forward_3d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        let passes_mode = separable_gate(plan, input.len(), 3)?;
        let n = plan.len();
        let passes = [
            (passes_mode, FiberLayout::axis(n, 3, 2)?),
            (passes_mode, FiberLayout::axis(n, 3, 1)?),
            (passes_mode, FiberLayout::axis(n, 3, 0)?),
        ];
        let mut output = vec![0.0_f32; input.len()];
        DctGpuKernel::execute_separable_into(self.device(), input, &mut output, &passes, 1.0)?;
        Ok(output)
    }

    fn execute_forward_3d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView3<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>> {
        let n = plan.len();
        let flat: Vec<f32> = input.iter().copied().collect();
        let result = self.execute_forward_3d(plan, &flat)?;
        leto::Array::from_mnemosyne_vec([n, n, n], result).map_err(|_| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto DCT/DST 3D output".to_string(),
        })
    }

    fn execute_inverse_2d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        separable_gate(plan, input.len(), 2)?;
        let n = plan.len();
        let (mode, scale) = inverse_mode_scale(plan.payload());
        let passes = [
            (mode, FiberLayout::axis(n, 2, 1)?),
            (mode, FiberLayout::axis(n, 2, 0)?),
        ];
        let mut output = vec![0.0_f32; input.len()];
        DctGpuKernel::execute_separable_into(
            self.device(),
            input,
            &mut output,
            &passes,
            scale.powi(2),
        )?;
        Ok(output)
    }

    fn execute_inverse_2d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView2<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 2>> {
        let n = plan.len();
        let flat: Vec<f32> = input.iter().copied().collect();
        let result = self.execute_inverse_2d(plan, &flat)?;
        leto::Array::from_mnemosyne_vec([n, n], result).map_err(|_| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto DCT/DST 2D output".to_string(),
        })
    }

    fn execute_inverse_3d(&self, plan: &DctDstWgpuPlan, input: &[f32]) -> WgpuResult<Vec<f32>> {
        separable_gate(plan, input.len(), 3)?;
        let n = plan.len();
        let (mode, scale) = inverse_mode_scale(plan.payload());
        let passes = [
            (mode, FiberLayout::axis(n, 3, 2)?),
            (mode, FiberLayout::axis(n, 3, 1)?),
            (mode, FiberLayout::axis(n, 3, 0)?),
        ];
        let mut output = vec![0.0_f32; input.len()];
        DctGpuKernel::execute_separable_into(
            self.device(),
            input,
            &mut output,
            &passes,
            scale.powi(3),
        )?;
        Ok(output)
    }

    fn execute_inverse_3d_leto(
        &self,
        plan: &DctDstWgpuPlan,
        input: leto::ArrayView3<'_, f32>,
    ) -> WgpuResult<leto::Array<f32, leto::MnemosyneStorage<f32>, 3>> {
        let n = plan.len();
        let flat: Vec<f32> = input.iter().copied().collect();
        let result = self.execute_inverse_3d(plan, &flat)?;
        leto::Array::from_mnemosyne_vec([n, n, n], result).map_err(|_| WgpuError::InvalidPlan {
            message: "failed to allocate Mnemosyne-backed Leto DCT/DST 3D output".to_string(),
        })
    }
}

/// Validate the plan and the flat `n^rank` field length, returning the
/// forward kernel mode.
fn separable_gate(
    plan: &Plan<DctGpuKernel>,
    input_len: usize,
    rank: u32,
) -> WgpuResult<infrastructure::kernel::DctMode> {
    plan.validate()?;
    let n = plan.len();
    let expected = cubic_element_count(n, rank)?;
    if input_len != expected {
        return Err(WgpuError::ShapeMismatch {
            message: format!(
                "{rank}D input expected {expected} elements for edge length {n}, got {input_len}"
            ),
        });
    }
    Ok(forward_mode(plan.payload()))
}

fn cubic_element_count(len: usize, rank: u32) -> WgpuResult<usize> {
    len.checked_pow(rank).ok_or_else(|| WgpuError::InvalidPlan {
        message: format!("{rank}D element count overflows usize for length {len}"),
    })
}
