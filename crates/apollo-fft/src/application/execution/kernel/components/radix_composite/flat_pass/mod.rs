//! Flat Stockham passes over the dispatched register width, one generic
//! kernel per radix.
//!
//! The interleaved-complex arithmetic rides hermes' sub-lane primitives
//! (`dup_even`, `dup_odd`, `swap_adjacent`, `fmaddsub`): the same products
//! and the same fused structure as the AVX2 intrinsics these kernels
//! replace, so every backend `vectorize` dispatches (AVX2, AVX-512, NEON)
//! shares one body and one per-element operation order, and the scalar
//! backend declines to the scalar pass. A `Complex<T>` is two `T` reals in
//! order, so a register of `LANE_COUNT` reals holds `LANE_COUNT / 2`
//! complexes and a complex offset is twice a real one.

mod radix2;
mod radix3;
mod radix4;
mod radix5;
mod radix7;
mod register;

pub(super) use radix2::FlatPassR2;
pub(super) use radix3::FlatPassR3;
pub(super) use radix4::FlatPassR4;
pub(super) use radix5::FlatPassR5;
pub(super) use radix7::FlatPassR7;

use register::{
    apply_pointwise, cmul, load, scatter_spill, store, store_arms, MAX_COMPLEXES_PER_REGISTER,
};
