//! Value-semantic SFT GPU verification manifest.
//!
//! Each private leaf owns one metadata, execution, host-boundary, represented-
//! storage, or explicit-precision contract. ADR 0017 records the dense inverse
//! proof sketch whose finite-precision evidence remains in the inverse leaf.

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
