//! Value-semantic CZT GPU verification manifest.
//!
//! Each leaf owns one private contract family. ADR 0019 records the CZT
//! definition and the DFT specialization used by the inverse round-trip law.

#![cfg(test)]

mod forward;
mod inverse;
mod leto;
mod metadata;
mod precision;
mod rejection;
mod support;
mod typed;
