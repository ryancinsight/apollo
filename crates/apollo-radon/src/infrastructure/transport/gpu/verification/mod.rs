//! Value-semantic Radon GPU verification manifest.
//!
//! Each private leaf owns one metadata, projection, adjoint-backprojection,
//! Leto-boundary, represented-storage, or filtered-backprojection contract.
//! ADR 0007 records the discrete-adjoint and filtered-backprojection theorems
//! whose finite-precision evidence remains in their owning leaves.

#[cfg(test)]
mod backprojection;
#[cfg(test)]
mod filtered;
#[cfg(test)]
mod forward;
#[cfg(test)]
mod leto;
#[cfg(test)]
mod metadata;
#[cfg(test)]
mod support;
#[cfg(test)]
mod typed;
