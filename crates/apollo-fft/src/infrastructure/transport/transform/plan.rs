//! Shared WGPU plan descriptor, typed per transform (ADR 0037).

use core::marker::PhantomData;

/// Metadata-preserving WGPU plan descriptor typed by its transform.
///
/// The type parameter is the transform's executor marker, so a plan built
/// for one transform cannot feed another transform's backend. The
/// descriptor carries the logical length only; equal input and output
/// length is the scaffold's contract (real-valued same-length transforms —
/// spectra-shaped transforms own their plan types).
pub struct WgpuTransformPlan<X> {
    len: usize,
    transform: PhantomData<X>,
}

impl<X> Clone for WgpuTransformPlan<X> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<X> Copy for WgpuTransformPlan<X> {}

impl<X> core::fmt::Debug for WgpuTransformPlan<X> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuTransformPlan")
            .field("len", &self.len)
            .finish()
    }
}

impl<X> PartialEq for WgpuTransformPlan<X> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
    }
}

impl<X> Eq for WgpuTransformPlan<X> {}

impl<X> WgpuTransformPlan<X> {
    /// Create a WGPU plan descriptor for a positive logical length.
    #[must_use]
    pub const fn new(len: usize) -> Self {
        Self {
            len,
            transform: PhantomData,
        }
    }

    /// Return the logical transform length carried by this descriptor.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return whether the descriptor carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}
