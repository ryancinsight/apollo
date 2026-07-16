//! Value-semantic Hilbert GPU verification manifest.
//!
//! Each private leaf owns one metadata, execution, inverse-projection,
//! host-boundary, represented-storage, or precision contract. ADR 0018 records
//! the double-Hilbert proof sketch whose finite-precision evidence remains in
//! the inverse leaf.

#[cfg(test)]
mod forward;
#[cfg(test)]
mod inverse;
#[cfg(test)]
mod leto;
#[cfg(test)]
mod metadata;
#[cfg(test)]
mod precision;
#[cfg(test)]
mod support;
#[cfg(test)]
mod typed;
