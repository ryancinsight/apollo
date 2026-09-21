/// Scaling convention for a direct transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Normalization {
    /// Preserve Apollo's analytical DCT-III convention.
    Unnormalized,
    /// Scale the basis to an orthonormal matrix.
    Orthonormal,
}
