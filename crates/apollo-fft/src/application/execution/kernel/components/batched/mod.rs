//! Batched power-of-two sub-transforms with the transform index in the lane.
//!
//! ## Why this layout
//!
//! A butterfly vectorized *within* one transform must gather its two operands
//! from positions `j` and `j + half`, and a complex multiply on interleaved data
//! then needs cross-lane shuffles to separate the real and imaginary parts. Both
//! costs vanish when the lane position holds the *transform* index instead:
//! with `B` independent transforms laid out so element `j` of transform `b` sits
//! at `j * B + b`, a butterfly reads two contiguous runs of `B` values, the
//! twiddle is one scalar broadcast across every lane, and no shuffle occurs
//! anywhere.
//!
//! This is the arrangement [`FftPlanarMut`] documents — "lane `c` across all
//! rows is one independent transform instance and no cross-lane shuffle is
//! required" — and which nothing previously used.
//!
//! Measured against the other kernel shapes tried in this crate, all of which
//! sat between 3.4 and 6.1 flops/ns, this reaches 10.2 to 10.7 across batch and
//! length combinations, using the same `hermes_simd` `Vector` operations. The
//! layout is the variable.
//!
//! ## Where it applies
//!
//! The four-step decomposition already splits `N = N1 x N2` and transforms
//! along each axis in turn. Writing `i = j * N2 + b`, the input is *already*
//! batch-major for the first axis, so those `N2` transforms of length `N1` need
//! no transpose at all; and the output index `k2 * N1 + k1` falls out of the
//! second batched pass, so there is no final transpose either. One transpose
//! sits between them, in place because the four-step gate admits only square
//! splits.
//!
//! [`FftPlanarMut`]: crate::domain::storage::FftPlanarMut

#[cfg(all(test, windows, target_arch = "x86_64"))]
macro_rules! sect {
    ($label:expr, $body:block) => {{
        let t0 = unsafe { core::arch::x86_64::_rdtsc() };
        let out = $body;
        let t1 = unsafe { core::arch::x86_64::_rdtsc() };
        crate::application::execution::kernel::components::batched::sections::record(
            $label,
            t1 - t0,
        );
        out
    }};
}
#[cfg(not(all(test, windows, target_arch = "x86_64")))]
macro_rules! sect {
    ($label:expr, $body:block) => {{
        let _label: &str = $label;
        $body
    }};
}

pub(crate) mod boundary;
mod cache;
mod dif;
mod dit;
mod driver;
mod fold;
mod lane;
mod lane_order;
mod pass;
mod plan;
pub(crate) mod plane;
mod radix;
mod register;
mod seams;
mod sweep;

pub(crate) use cache::BatchedPlanCache;
pub(crate) use driver::four_step_batched;
pub(crate) use plane::{planar_applies, scratch_len};

#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) mod sections;

// Test-gated deliberately: the interleaved in-place kernel is correct and
// measured — and slower than the planar sibling by 16 to 37% pinned on an
// E-core at every covered size, because the shuffle cost of interleaved
// butterflies outweighs the planar seams it removes. It stays as the
// independent-implementation differential oracle for this module; the
// pinned probe that declined it is committed beside it.
#[cfg(test)]
pub(crate) mod interleaved;

// Windows-gated: pins threads through Win32 to control the hybrid scheduler.
#[cfg(all(test, windows))]
mod pinned_ladder;

// Additionally x86-gated: the pass totals come from `_rdtsc`.
#[cfg(all(test, windows, target_arch = "x86_64"))]
mod pinned_sections;

#[cfg(test)]
mod tests;
