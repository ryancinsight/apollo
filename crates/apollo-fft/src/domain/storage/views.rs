//! Concrete mutable FFT storage views.

use super::{FftSample, FftStorage};
use eunomia::Complex;
use std::borrow::Cow;

/// Mutable planar structure-of-arrays FFT view.
///
/// The real and imaginary planes are independent contiguous slices. This is
/// the storage shape consumed by batched SIMD kernels because lane `c` across
/// all rows is one independent transform instance and no cross-lane shuffle is
/// required.
pub struct FftPlanarMut<'a, T: FftSample> {
    re: &'a mut [T],
    im: &'a mut [T],
    rows: usize,
    cols: usize,
}

impl<'a, T: FftSample> FftPlanarMut<'a, T> {
    /// Create a planar FFT view over row-major real and imaginary planes.
    ///
    /// # Panics
    ///
    /// Panics if either plane length is not `rows * cols`.
    #[must_use]
    pub fn new(re: &'a mut [T], im: &'a mut [T], rows: usize, cols: usize) -> Self {
        assert_eq!(re.len(), rows * cols, "real plane length mismatch");
        assert_eq!(im.len(), rows * cols, "imaginary plane length mismatch");
        Self { re, im, rows, cols }
    }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.rows);
        debug_assert!(col < self.cols);
        row * self.cols + col
    }
}

impl<T: FftSample> FftStorage<T> for FftPlanarMut<'_, T> {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    fn load_re(&self, row: usize, col: usize) -> T {
        self.re[self.index(row, col)]
    }

    #[inline]
    fn load_im(&self, row: usize, col: usize) -> T {
        self.im[self.index(row, col)]
    }

    #[inline]
    fn store(&mut self, row: usize, col: usize, re: T, im: T) {
        let idx = self.index(row, col);
        self.re[idx] = re;
        self.im[idx] = im;
    }
}

/// Mutable interleaved array-of-structures FFT view.
///
/// This view represents caller-owned AoS complex buffers. Application
/// orchestration converts to planar storage at the plan boundary when a SIMD
/// path requires SoA.
pub struct FftInterleavedMut<'a, T: FftSample> {
    data: &'a mut [Complex<T>],
    rows: usize,
    cols: usize,
}

impl<'a, T: FftSample> FftInterleavedMut<'a, T> {
    /// Create an interleaved FFT view over row-major complex samples.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    #[must_use]
    pub fn new(data: &'a mut [Complex<T>], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "interleaved length mismatch");
        Self { data, rows, cols }
    }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.rows);
        debug_assert!(col < self.cols);
        row * self.cols + col
    }
}

impl<T: FftSample> FftStorage<T> for FftInterleavedMut<'_, T> {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    fn load_re(&self, row: usize, col: usize) -> T {
        self.data[self.index(row, col)].re
    }

    #[inline]
    fn load_im(&self, row: usize, col: usize) -> T {
        self.data[self.index(row, col)].im
    }

    #[inline]
    fn store(&mut self, row: usize, col: usize, re: T, im: T) {
        let idx = self.index(row, col);
        self.data[idx] = Complex { re, im };
    }
}

/// Copy-on-write interleaved array-of-structures FFT storage.
///
/// This view borrows caller-owned complex buffers for read-only paths and
/// promotes to owned storage only when [`FftStorage::store`] mutates a sample.
pub struct FftInterleavedCow<'a, T: FftSample> {
    data: Cow<'a, [Complex<T>]>,
    rows: usize,
    cols: usize,
}

impl<'a, T: FftSample> FftInterleavedCow<'a, T> {
    /// Create a borrowed copy-on-write FFT view over row-major complex samples.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    #[must_use]
    pub fn borrowed(data: &'a [Complex<T>], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "interleaved length mismatch");
        Self {
            data: Cow::Borrowed(data),
            rows,
            cols,
        }
    }

    /// Create an owned copy-on-write FFT view over row-major complex samples.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    #[must_use]
    pub fn owned(data: Vec<Complex<T>>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "interleaved length mismatch");
        Self {
            data: Cow::Owned(data),
            rows,
            cols,
        }
    }

    /// Returns true when this view still borrows caller-owned storage.
    #[inline]
    #[must_use]
    pub fn is_borrowed(&self) -> bool {
        matches!(self.data, Cow::Borrowed(_))
    }

    /// Returns true when this view owns its storage.
    #[inline]
    #[must_use]
    pub fn is_owned(&self) -> bool {
        matches!(self.data, Cow::Owned(_))
    }

    /// Borrow the current interleaved complex storage.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Complex<T>] {
        self.data.as_ref()
    }

    /// Consume the view and return owned interleaved complex storage.
    #[inline]
    #[must_use]
    pub fn into_owned(self) -> Vec<Complex<T>> {
        self.data.into_owned()
    }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.rows);
        debug_assert!(col < self.cols);
        row * self.cols + col
    }
}

impl<T: FftSample> FftStorage<T> for FftInterleavedCow<'_, T> {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    fn load_re(&self, row: usize, col: usize) -> T {
        self.data[self.index(row, col)].re
    }

    #[inline]
    fn load_im(&self, row: usize, col: usize) -> T {
        self.data[self.index(row, col)].im
    }

    #[inline]
    fn store(&mut self, row: usize, col: usize, re: T, im: T) {
        let idx = self.index(row, col);
        self.data.to_mut()[idx] = Complex { re, im };
    }
}

#[cfg(test)]
mod tests {
    use super::{FftInterleavedCow, FftStorage};
    use eunomia::Complex64;

    #[test]
    fn interleaved_cow_borrows_without_copying_on_read() {
        let source = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        ];
        let view = FftInterleavedCow::borrowed(&source, 2, 2);

        assert!(view.is_borrowed());
        assert!(!view.is_owned());
        assert_eq!(view.as_slice().as_ptr(), source.as_ptr());
        assert_eq!(view.load_re(1, 0), 5.0);
        assert_eq!(view.load_im(1, 1), 8.0);
    }

    #[test]
    fn interleaved_cow_detaches_on_store() {
        let source = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        ];
        let mut view = FftInterleavedCow::borrowed(&source, 2, 2);

        view.store(0, 1, 30.0, 40.0);

        assert!(view.is_owned());
        assert!(!view.is_borrowed());
        assert_ne!(view.as_slice().as_ptr(), source.as_ptr());
        assert_eq!(source[1], Complex64::new(3.0, 4.0));
        assert_eq!(view.as_slice()[1], Complex64::new(30.0, 40.0));
    }
}
