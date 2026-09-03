//! Storage-layout contract for FFT execution.
//!
//! The domain layer owns this trait because layout is part of Apollo's
//! mathematical data contract, not a CPU or GPU implementation detail.

/// Scalar precision admitted by FFT storage views.
///
/// The storage scalar is a domain contract because precision and
/// representation are observable by callers and by backend capability
/// descriptors. `ZERO` is required so infrastructure kernels can allocate
/// fixed-size stack tiles without binding to a concrete scalar type.
pub trait FftSample: Copy {
    /// Additive identity in this storage precision.
    const ZERO: Self;
}

impl FftSample for eunomia::F16 {
    const ZERO: Self = Self::ZERO;
}

impl FftSample for f32 {
    const ZERO: Self = 0.0;
}

impl FftSample for f64 {
    const ZERO: Self = 0.0;
}

/// Zero-cost mutable FFT storage view over a rectangular complex matrix.
///
/// `row` identifies the point inside one short transform and `col` identifies
/// the independent transform instance. Six-step execution uses this as the
/// matrix coordinate `(n1, n2)` in a conceptual `N1 x N2` factorization of a
/// one-dimensional signal.
///
/// # Theorem: row-major coordinate bijection
///
/// Let `A` be a flat complex array of length `rows * cols`. The map
/// `phi(row, col) = row * cols + col` is a bijection from
/// `{0..rows} x {0..cols}` to `{0..rows * cols}`. Injectivity follows because
/// `row_a * cols + col_a = row_b * cols + col_b` implies equal quotients and
/// remainders under Euclidean division by `cols`. Surjectivity follows because
/// every flat index `k < rows * cols` has inverse `(k / cols, k % cols)`.
///
/// Therefore algorithms expressed against `(row, col)` read and write the same
/// mathematical samples as flat storage without allocating or copying.
///
/// # Contract
///
/// Implementations must map every valid `(row, col)` pair to one storage
/// element. `load_*` and `store` must not allocate.
pub trait FftStorage<T: FftSample> {
    /// Number of matrix rows.
    fn rows(&self) -> usize;

    /// Number of matrix columns.
    fn cols(&self) -> usize;

    /// Load the real component at `(row, col)`.
    fn load_re(&self, row: usize, col: usize) -> T;

    /// Load the imaginary component at `(row, col)`.
    fn load_im(&self, row: usize, col: usize) -> T;

    /// Store the complex components at `(row, col)`.
    fn store(&mut self, row: usize, col: usize, re: T, im: T);
}
