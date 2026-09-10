//! The spectrum a two-move transform leaves, in the order it leaves it.
//!
//! A forward that skips the move restoring C order hands back a volume whose
//! axes read `(z, x, y)`. Nothing about the buffer says so, so the order rides
//! in this type: it holds the caller's storage borrowed for as long as the
//! spectrum is rotated, reports the shape in that order, and is the only thing
//! the matching inverse accepts.

use eunomia::Complex;

use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

/// A 3-D spectrum whose axes are in `(z, x, y)` order.
///
/// Elementwise work — a k-space operator, a filter, a mask — is order-agnostic
/// as long as its own array is stored in this order, which is what
/// [`Self::as_mut_slice`] and [`Self::shape`] are for. Anything that reads the
/// volume by axis must transform it back first.
#[derive(Debug)]
pub struct RotatedSpectrum<'data, F: MixedRadixScalar> {
    data: &'data mut [Complex<F>],
    shape: [usize; 3],
}

impl<'data, F: MixedRadixScalar<Complex = Complex<F>>> RotatedSpectrum<'data, F> {
    /// Binds `data` as a rotated spectrum of `[nz, nx, ny]` elements.
    pub(super) fn new(data: &'data mut [Complex<F>], shape: [usize; 3]) -> Self {
        let [nx, ny, nz] = shape;
        debug_assert_eq!(
            data.len(),
            nx * ny * nz,
            "invariant: a rotated spectrum spans the whole volume"
        );
        Self {
            data,
            shape: [nz, nx, ny],
        }
    }

    /// The extents in the order the elements are stored: `[nz, nx, ny]`.
    #[must_use]
    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    /// The spectrum in storage order.
    #[must_use]
    pub fn as_slice(&self) -> &[Complex<F>] {
        self.data
    }

    /// The spectrum in storage order, for elementwise work.
    pub fn as_mut_slice(&mut self) -> &mut [Complex<F>] {
        self.data
    }

    /// Releases the borrow for the inverse that consumes this order.
    pub(super) fn into_parts(self) -> (&'data mut [Complex<F>], [usize; 3]) {
        let [nz, nx, ny] = self.shape;
        (self.data, [nx, ny, nz])
    }
}
