//! Value-semantic verification grouped by the NTT contract under test.
//!
//! Live-device leaves return only when the provider cannot acquire an adapter.
//! An acquired device always executes computed-residue assertions; no case is
//! ignored or replaced by a capability-only assertion.
//!
//! The finite-field contracts are `INTT(NTT(x)) = x` and
//! `INTT(NTT(a) ⊙ NTT(b)) = a ★ b`. The owning NTT plan and README contain the
//! proof sketch; generated exact-residue and direct-convolution properties
//! supply executable evidence.

#[cfg(test)]
mod exact;
#[cfg(test)]
mod metadata;
#[cfg(test)]
mod properties;
#[cfg(test)]
mod quantized;
#[cfg(test)]
mod reusable;
#[cfg(test)]
mod support;
