#![deny(missing_docs)]

//! Deterministic-lane heap allocation for Apollo measurement.
//!
//! Fresh processes slide heap addresses with ASLR, and the low address bits
//! decide 4 KiB-aliasing stalls and L1-set conflicts for an engine's access
//! pattern. The N=32 comparison cell measured +-70-100% cross-run median
//! swings on one binary that cluster exactly by buffer address lanes
//! (apollo#646), so plain `Vec` working buffers make every ratio a
//! per-process lottery. [`LaneBuffer`] allocates page-aligned and starts at
//! a caller-chosen in-page lane, pinning the low 12 address bits regardless
//! of the ASLR slide and making repeated runs of one binary comparable.
//!
//! This crate owns the `unsafe` allocation core; consumers (notably
//! `apollo-bench`, which forbids `unsafe_code`) use the safe [`LaneBuffer`]
//! API. One fixed lane per role measures one aliasing configuration:
//! absolute medians are lane-specific, but ratios at a fixed lane are stable
//! across runs and commits, which is what the comparison needs. Roles
//! sharing a measurement window must take distinct lanes (a multiple of 512
//! keeps L1 sets disjoint too); roles in disjoint windows may reuse lanes.

mod lane_buffer;

pub use lane_buffer::LaneBuffer;
