//! Value-semantic Wavelet GPU verification manifest.
//!
//! Each private leaf owns one metadata, forward, inverse-law, Leto-boundary,
//! or represented-storage contract. ADR 0016 records the Haar orthonormality
//! proof sketch whose finite-precision reconstruction and Parseval evidence
//! remains in the inverse leaf.

#[cfg(test)]
mod forward;
#[cfg(test)]
mod inverse;
#[cfg(test)]
mod leto;
#[cfg(test)]
mod metadata;
#[cfg(test)]
mod support;
#[cfg(test)]
mod typed;
