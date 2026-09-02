//! The Stockham stages as hermes lane kernels.
//!
//! [`stage_impl`], [`stage_pair_impl`], and [`stage_triple_impl`] are the
//! scalar recurrences (their docs derive them; the pair and triple forms fuse
//! two and three radix-2 passes). This module is
//! those recurrences vectorised: each group's `k` loop runs `A::LANE_COUNT / 2`
//! complex samples per register through [`ComplexReg`] arithmetic, and the
//! ragged tail (fewer samples than a register holds) runs the scalar
//! recurrence. One generic body per stage serves every lane width hermes
//! offers, so the per-ISA intrinsic copies of these stages are retired; the
//! caller picks the lane count its route was tuned for and falls back to the
//! scalar recurrence when no hardware backend serves it (ADR 0045).
//!
//! Rounding matches the retired AVX/FMA stages bit-for-bit: [`ComplexReg`]'s
//! multiply is the same dup/swap/`fmaddsub` sequence with the same operand
//! order, and adds and subtracts are lane-wise IEEE operations in both.
//!
//! [`stage_impl`]: super::super::stage::stage_impl
//! [`stage_pair_impl`]: super::stage::stage_pair_impl
//! [`ComplexReg`]: hermes_simd::ComplexReg

pub(crate) mod base;
pub(crate) mod pair;
#[cfg(test)]
mod tests;
pub(crate) mod triple;

pub(crate) use base::*;
pub(crate) use pair::*;
pub(crate) use triple::*;
