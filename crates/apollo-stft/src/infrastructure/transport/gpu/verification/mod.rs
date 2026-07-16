//! Value-semantic STFT GPU verification manifest.
//!
//! Each private leaf owns one independent metadata, differential,
//! reconstruction, host-boundary, or reusable-storage contract. The
//! weighted-overlap-add identity remains documented in ADR 0008 and is
//! exercised by the inverse leaf without duplicating test support.

#[cfg(test)]
mod forward;
#[cfg(test)]
mod inverse;
#[cfg(test)]
mod metadata;
#[cfg(test)]
mod reusable;
#[cfg(test)]
mod support;
#[cfg(test)]
mod typed;
