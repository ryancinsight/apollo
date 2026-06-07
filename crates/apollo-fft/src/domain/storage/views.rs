//! Concrete mutable FFT storage views.

use super::{FftSample, FftStorage};
use num_complex::Complex;

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
