//! Unrolled small power-of-two codelets, split by SIMD lane density.

mod n16;
mod n32;
mod precise;
mod reduced;

// Test-gated deliberately: correct against the direct-DFT oracle in both
// directions, and measured slower than the scalar codelet it would replace
// (`small_pot_arms`). It stays as the instrument's subject, so the comparison
// that declined it can be re-run rather than becoming a note nobody can check.
#[cfg(test)]
mod n8;

pub(super) use precise::{small_pot_inplace_precise, small_pot_inplace_sized_precise};
pub(super) use reduced::{small_pot_inplace_reduced, small_pot_inplace_sized_reduced};

// The probe entries the `small_pot_arms` instrument reaches; they exist only
// under `cfg(test)` and carry no production call site.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) use n16::vector_arm_unchecked as n16_vector_arm_unchecked;
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) use n8::vector_arm_unchecked as n8_vector_arm_unchecked;
