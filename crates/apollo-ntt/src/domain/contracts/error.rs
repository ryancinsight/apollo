//! NTT contracts and capabilities.

use thiserror::Error;

/// Errors produced by NTT plan creation or execution.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NttError {
    /// Length is zero.
    #[error("length must be > 0")]
    EmptyLength,
    /// Length is not a power of two.
    #[error("length must be a power of two")]
    NonPowerOfTwo,
    /// Input length does not match the plan.
    #[error("input length mismatch")]
    LengthMismatch,
    /// Modulus is less than 2.
    #[error("modulus must be at least 2")]
    InvalidModulus,
    /// Selected modulus does not support the requested transform length.
    #[error("transform length is not supported by the modulus")]
    UnsupportedLength,
    /// The modulus is composite, so the residues form no field and the
    /// inverse transform has no `n^{-1}` or root inverse to use.
    #[error("modulus {modulus} is not prime")]
    CompositeModulus {
        /// The rejected modulus.
        modulus: u64,
    },
    /// The supplied root does not yield a primitive `n`-th root of unity:
    /// `g^((q - 1) / n)` has an order below `n`, or `g ≡ 0 (mod q)`.
    #[error("root {primitive_root} gives no primitive {n}-th root of unity modulo {modulus}")]
    NotPrimitiveRoot {
        /// The transform length.
        n: usize,
        /// The modulus.
        modulus: u64,
        /// The rejected root.
        primitive_root: u64,
    },
}
