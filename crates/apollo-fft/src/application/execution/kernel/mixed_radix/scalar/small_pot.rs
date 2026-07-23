//! Unrolled small power-of-two codelets, split by SIMD lane density.

mod precise;
mod reduced;

pub(super) use precise::{small_pot_inplace_precise, small_pot_inplace_sized_precise};
pub(super) use reduced::{small_pot_inplace_reduced, small_pot_inplace_sized_reduced};
