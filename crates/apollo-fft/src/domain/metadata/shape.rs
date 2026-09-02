//! Shared shape and grid descriptor types.
//!
//! Every descriptor validates at construction — a zero length is refused by
//! `new` — and that is the only way in: fields are private, the structs are
//! `#[non_exhaustive]`, and deserialization routes through the same check via
//! a `try_from` representation ([ADR 0044](../../../../docs/adr/0044-shape-descriptors-validate-at-construction.md)).

use crate::domain::contracts::error::{ApolloError, ApolloResult};
use serde::{Deserialize, Serialize};

/// Shape descriptor for 1D plans: a non-zero signal length.
///
/// ```compile_fail
/// // The length is validated by `Shape1D::new`; there is no literal path.
/// let shape = apollo_fft::Shape1D { n: 0 };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "Shape1DRepr", into = "Shape1DRepr")]
#[non_exhaustive]
pub struct Shape1D {
    n: usize,
}

impl Shape1D {
    /// Create a validated 1D shape descriptor.
    ///
    /// # Errors
    ///
    /// A validation error when `n == 0`.
    pub fn new(n: usize) -> ApolloResult<Self> {
        if n == 0 {
            return Err(ApolloError::validation("n", n.to_string(), "must be > 0"));
        }
        Ok(Self { n })
    }

    /// Length of the signal; never zero.
    #[must_use]
    #[inline]
    pub const fn n(self) -> usize {
        self.n
    }
}

/// Shape descriptor for 2D plans: non-zero lengths along x and y.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "Shape2DRepr", into = "Shape2DRepr")]
#[non_exhaustive]
pub struct Shape2D {
    nx: usize,
    ny: usize,
}

impl Shape2D {
    /// Create a validated 2D shape descriptor.
    ///
    /// # Errors
    ///
    /// A validation error naming the first zero axis.
    pub fn new(nx: usize, ny: usize) -> ApolloResult<Self> {
        if nx == 0 {
            return Err(ApolloError::validation("nx", nx.to_string(), "must be > 0"));
        }
        if ny == 0 {
            return Err(ApolloError::validation("ny", ny.to_string(), "must be > 0"));
        }
        Ok(Self { nx, ny })
    }

    /// X dimension; never zero.
    #[must_use]
    #[inline]
    pub const fn nx(self) -> usize {
        self.nx
    }

    /// Y dimension; never zero.
    #[must_use]
    #[inline]
    pub const fn ny(self) -> usize {
        self.ny
    }
}

/// Shape descriptor for 3D plans: non-zero lengths along x, y and z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "Shape3DRepr", into = "Shape3DRepr")]
#[non_exhaustive]
pub struct Shape3D {
    nx: usize,
    ny: usize,
    nz: usize,
}

impl Shape3D {
    /// Create a validated 3D shape descriptor.
    ///
    /// # Errors
    ///
    /// A validation error naming the first zero axis.
    pub fn new(nx: usize, ny: usize, nz: usize) -> ApolloResult<Self> {
        if nx == 0 {
            return Err(ApolloError::validation("nx", nx.to_string(), "must be > 0"));
        }
        if ny == 0 {
            return Err(ApolloError::validation("ny", ny.to_string(), "must be > 0"));
        }
        if nz == 0 {
            return Err(ApolloError::validation("nz", nz.to_string(), "must be > 0"));
        }
        Ok(Self { nx, ny, nz })
    }

    /// X dimension; never zero.
    #[must_use]
    #[inline]
    pub const fn nx(self) -> usize {
        self.nx
    }

    /// Y dimension; never zero.
    #[must_use]
    #[inline]
    pub const fn ny(self) -> usize {
        self.ny
    }

    /// Z dimension; never zero.
    #[must_use]
    #[inline]
    pub const fn nz(self) -> usize {
        self.nz
    }

    /// Return the total number of points.
    #[must_use]
    pub const fn volume(self) -> usize {
        self.nx * self.ny * self.nz
    }
}

/// Serialized form of [`Shape1D`]; deserialization re-validates through `new`.
#[derive(Serialize, Deserialize)]
struct Shape1DRepr {
    n: usize,
}

impl TryFrom<Shape1DRepr> for Shape1D {
    type Error = ApolloError;
    fn try_from(repr: Shape1DRepr) -> ApolloResult<Self> {
        Self::new(repr.n)
    }
}

impl From<Shape1D> for Shape1DRepr {
    fn from(shape: Shape1D) -> Self {
        Self { n: shape.n }
    }
}

/// Serialized form of [`Shape2D`]; deserialization re-validates through `new`.
#[derive(Serialize, Deserialize)]
struct Shape2DRepr {
    nx: usize,
    ny: usize,
}

impl TryFrom<Shape2DRepr> for Shape2D {
    type Error = ApolloError;
    fn try_from(repr: Shape2DRepr) -> ApolloResult<Self> {
        Self::new(repr.nx, repr.ny)
    }
}

impl From<Shape2D> for Shape2DRepr {
    fn from(shape: Shape2D) -> Self {
        Self {
            nx: shape.nx,
            ny: shape.ny,
        }
    }
}

/// Serialized form of [`Shape3D`]; deserialization re-validates through `new`.
#[derive(Serialize, Deserialize)]
struct Shape3DRepr {
    nx: usize,
    ny: usize,
    nz: usize,
}

impl TryFrom<Shape3DRepr> for Shape3D {
    type Error = ApolloError;
    fn try_from(repr: Shape3DRepr) -> ApolloResult<Self> {
        Self::new(repr.nx, repr.ny, repr.nz)
    }
}

impl From<Shape3D> for Shape3DRepr {
    fn from(shape: Shape3D) -> Self {
        Self {
            nx: shape.nx,
            ny: shape.ny,
            nz: shape.nz,
        }
    }
}

/// Half-spectrum descriptor for R2C 3D transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HalfSpectrum3D {
    /// Full real-domain shape.
    pub full: Shape3D,
    /// Number of independent complex bins along Z.
    pub nz_c: usize,
}

impl HalfSpectrum3D {
    /// Construct the half-spectrum descriptor implied by a real-domain shape.
    #[must_use]
    pub fn from_shape(full: Shape3D) -> Self {
        Self {
            full,
            nz_c: full.nz() / 2 + 1,
        }
    }
}
