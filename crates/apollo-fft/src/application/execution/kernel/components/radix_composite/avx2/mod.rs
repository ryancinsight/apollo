//! AVX2+FMA flat Stockham passes, split into per-radix leaf modules.
//!
//! ## Module map
//!
//! | Module      | f64      | f32      |
//! |-------------|----------|----------|
//! | `helpers`   | cmul, rot_neg_i, rot_pos_i, apply_pointwise | cmul_f32, rot_neg_i_f32, rot_pos_i_f32, apply_pointwise_f32, cmul_f32_128 |
//! | `radix2`    | flat_pass_r2_f64 | flat_pass_r2_f32 |
//! | `radix3`    | flat_pass_r3_f64 | flat_pass_r3_f32 |
//! | `radix4`    | flat_pass_r4_f64 | flat_pass_r4_f32 |
//! | `radix5`    | flat_pass_r5_f64 | flat_pass_r5_f32 |
//! | `radix7`    | flat_pass_r7_f64 | flat_pass_r7_f32 |
//!
//! All `flat_pass_rN_*` functions are visible within `radix_composite` and
//! called from `cache.rs`.

mod helpers;
mod radix2;
mod radix3;
mod radix4;
mod radix5;
mod radix7;

// Re-export the 10 public entry points for cache.rs callers.
pub(super) use radix2::{flat_pass_r2_f32, flat_pass_r2_f64};
pub(super) use radix3::{flat_pass_r3_f32, flat_pass_r3_f64};
pub(super) use radix4::{flat_pass_r4_f32, flat_pass_r4_f64};
pub(super) use radix5::{flat_pass_r5_f32, flat_pass_r5_f64};
pub(super) use radix7::{flat_pass_r7_f32, flat_pass_r7_f64};
