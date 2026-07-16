//! Value-semantic CZT GPU verification manifest.
//!
//! Each leaf owns one private contract family. ADR 0019 records the CZT
//! definition and the DFT specialization used by the inverse round-trip law.

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
mod rejection;
#[cfg(test)]
mod support;
#[cfg(test)]
mod typed;
