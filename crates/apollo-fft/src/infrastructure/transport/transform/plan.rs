//! Shared WGPU plan descriptor, typed per transform (ADR 0037).

use super::backend::GpuTransformPlanner;
use super::error::{WgpuError, WgpuResult};

/// Metadata-preserving WGPU plan descriptor typed by its transform.
///
/// The type parameter is the transform's executor marker, so a plan built
/// for one transform cannot feed another transform's backend. The
/// descriptor carries the transform's plan payload
/// ([`GpuTransformPlanner::Plan`]) — a bare length for same-length 1D
/// transforms, a richer structure where the transform demands one.
pub struct WgpuTransformPlan<X: GpuTransformPlanner> {
    payload: X::Plan,
}

impl<X: GpuTransformPlanner> Clone for WgpuTransformPlan<X> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<X: GpuTransformPlanner> Copy for WgpuTransformPlan<X> {}

impl<X: GpuTransformPlanner> core::fmt::Debug for WgpuTransformPlan<X> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuTransformPlan")
            .field("payload", &self.payload)
            .finish()
    }
}

impl<X: GpuTransformPlanner> PartialEq for WgpuTransformPlan<X> {
    fn eq(&self, other: &Self) -> bool {
        self.payload == other.payload
    }
}

impl<X: GpuTransformPlanner> Eq for WgpuTransformPlan<X> {}

impl<X: GpuTransformPlanner> WgpuTransformPlan<X> {
    /// Create a WGPU plan descriptor from the transform's plan payload.
    #[must_use]
    pub const fn new(payload: X::Plan) -> Self {
        Self { payload }
    }

    /// Return the transform's plan payload.
    #[must_use]
    pub const fn payload(&self) -> &X::Plan {
        &self.payload
    }

    /// Return the logical input length demanded by this descriptor.
    #[must_use]
    pub fn len(&self) -> usize {
        X::input_len(&self.payload)
    }

    /// Return the logical output length produced by this descriptor.
    #[must_use]
    pub fn output_len(&self) -> usize {
        X::output_len(&self.payload)
    }

    /// Return whether the descriptor demands zero input.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validate the descriptor: nonzero input length plus the
    /// transform's structural rules. Extension surfaces run the same
    /// gate the generic backend applies before dispatch.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan rejection naming the violated constraint.
    pub fn validate(&self) -> WgpuResult<()> {
        let len = self.len();
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid length {len}: length must be greater than zero"),
            });
        }
        X::validate(&self.payload)
    }
}
