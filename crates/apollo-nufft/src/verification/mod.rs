//! NUFFT verification-tree manifest.
//!
//! Each private leaf owns one mathematical law or storage contract. The
//! assertions, fixtures, and error bounds live with that law so the module
//! boundary does not duplicate any numerical implementation.
//!
//! ## Evidence hierarchy
//!
//! The direct identities and adjoint relationship are empirical floating-point
//! checks of the documented algebraic theorems. Kernel-width checks validate
//! the stated Fessler-Sutton error ordering for fixed fixtures; they are not a
//! machine-checked proof of the global bound.

mod adjoint;
mod direct_identity;
mod in_place_consistency;
mod kernel_width;
mod typed_storage_1d;
mod typed_storage_3d;
