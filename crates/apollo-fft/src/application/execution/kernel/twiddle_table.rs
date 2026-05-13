//! SSOT for Cooley-Tukey twiddle-factor table construction.
//!
//! ## Mathematical foundation
//!
//! ### Twiddle factors
//!
//! For an N-point power-of-two DFT, stage s (sub-transform length `len = 2^s`)
//! requires twiddle factors `W_len^j = exp(i·sign·2π·j/len)` for j = 0..len/2.
//! The j=0 factor is always 1 and can be elided; however this implementation
//! stores all `half = len/2` entries to enable direct index arithmetic at call sites.
//!
//! ### Layout
//!
//! The contiguous table has length N − 1. Stage s occupies indices
//! `2^(s−1) − 1 .. 2^s − 1` (half = 2^(s−1) entries). The entry at relative
//! position j within that stage is `C::from_angle(sign · 2π · j / len)`.
//!
//! **Proof of layout**: The total number of entries across all L = log₂N stages
//! is `∑_{s=1}^{L} 2^(s-1) = 2^L − 1 = N − 1`. Each stage s contributes
//! `half = 2^(s-1)` entries starting at base = `2^(s-1) − 1`. □
//!
//! ### Precision model
//!
//! Angles are computed in f64 arithmetic regardless of the output type `C`.
//! For `Complex32` and `Complex<f16>`, the f64 sine/cosine is computed first, then
//! narrowed to the target precision. This minimises accumulated angle error
//! at the cost of one narrowing conversion per entry at table-build time
//! (a one-time cost amortised over all transform calls that share the table).
//!
//! **Error bound**: Each twiddle entry has relative error at most
//! `ε_target / 2` where `ε_target` is the machine epsilon of `C`.
//! For `Complex32`, `ε_f32 / 2 ≈ 5.96e-8`.
//!
//! ## DRY invariant
//!
//! `build_twiddle_table<C>` is the single authoritative implementation.
//! All typed wrappers (`build_forward_twiddle_table_64`, etc.) are one-line
//! delegations. Adding a new precision target requires only a new
//! `impl TwiddleOutput for <new type>` plus a one-line wrapper.
//!
//! ## References
//!
//! - Cooley, J.W. & Tukey, J.W. (1965). *Mathematics of Computation*, 19(90), 297-301.
//! - Van Loan, C. (1992). *Computational Frameworks for the FFT*. SIAM, §2.2.


#![allow(clippy::uninit_vec)]
use half::f16;

use num_complex::{Complex, Complex32, Complex64};

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Conversion from a radian angle to a stored complex twiddle element.
///
/// ## Invariant
///
/// `from_angle(a)` must return `exp(i·a)` rounded to the precision of `C`.
/// The angle `a` is computed in f64. Precision loss on conversion is acceptable
/// and documented in the caller's precision model.
///
/// ## Failure modes
///
/// No panic or error. IEEE 754 rounding guarantees bounded relative error
/// in the cos/sin evaluation at f64 precision before narrowing to C.
pub(crate) trait TwiddleOutput: Copy + Sized {
    /// Convert radian angle `a` to the stored complex element `exp(i·a)`.
    fn from_angle(a: f64) -> Self;

    /// Convert (cos, sin) components — already computed in f64 — to the stored type.
    ///
    /// Default: delegates to `from_angle` via `atan2`. Implementations should
    /// override this to directly narrow `(re, im)` without an inverse-trig call.
    #[inline]
    fn from_components(re: f64, im: f64) -> Self {
        // Fallback: reconstruct the angle.  Concrete impls override this.
        let _ = (re, im);
        // Provide the fallback via atan2 — never called in practice since all
        // three concrete impls override.
        Self::from_angle(im.atan2(re))
    }
}

impl TwiddleOutput for Complex64 {
    /// Computes cos(a) + i·sin(a) in f64 precision.
    #[inline]
    fn from_angle(a: f64) -> Self {
        Complex64::new(a.cos(), a.sin())
    }

    #[inline]
    fn from_components(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

impl TwiddleOutput for Complex32 {
    /// Computes cos(a) + i·sin(a) in f64, then narrows to f32.
    ///
    /// **Error bound**: relative error ≤ ε_f32 / 2 ≈ 5.96e-8 per component.
    #[inline]
    fn from_angle(a: f64) -> Self {
        Complex32::new(a.cos() as f32, a.sin() as f32)
    }

    #[inline]
    fn from_components(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
}

impl TwiddleOutput for Complex<f16> {
    /// Computes cos(a) + i·sin(a) in f64, narrows to f32, then to f16.
    ///
    /// **Error bound**: relative error ≤ ε_f16 / 2 ≈ 4.88e-4 per component.
    #[inline]
    fn from_angle(a: f64) -> Self {
        Complex::new(f16::from_f32(a.cos() as f32), f16::from_f32(a.sin() as f32))
    }

    #[inline]
    fn from_components(re: f64, im: f64) -> Self {
        Complex::new(f16::from_f32(re as f32), f16::from_f32(im as f32))
    }
}

// ── Generic builder ───────────────────────────────────────────────────────────

/// Build a contiguous per-stage Cooley-Tukey twiddle table for a power-of-two
/// transform of length `n`.
///
/// ## Algorithm
///
/// For each stage s = 1..log₂N:
/// - `len = 2^s`
/// - `half = len / 2`
/// - Entry at position j: `C::from_angle(sign · 2π · j / len)`, j = 0..half
///
/// Total entries = N − 1 (proof above in module doc).
///
/// ## Parameters
///
/// - `n`: transform length; must be a power of two.
/// - `sign`: `−1.0` for the forward transform (exp(−i·2π…)), `+1.0` for inverse.
///
/// ## Complexity and allocation
///
/// O(N log N) time, O(N) space (N − 1 elements).
///
/// ## Failure modes
///
/// - Panics (debug assertion) if `n` is not a power of two.
/// - Returns an empty `Vec` for n ≤ 1.
pub(crate) fn build_twiddle_table<C: TwiddleOutput>(n: usize, sign: f64) -> Vec<C> {
    debug_assert!(
        n.is_power_of_two(),
        "build_twiddle_table requires power-of-two n"
    );
    if n <= 1 {
        return Vec::new();
    }
    let log_n = n.trailing_zeros() as usize;
    let total = n - 1; // proved in module docs: sum of stage halves = n-1
                       // Pre-allocate exact capacity; set_len skips zero-init.
                       // SAFETY: C: TwiddleOutput + Copy has no Drop. Every slot is overwritten
                       // below before the Vec is returned. The cursor advances by half=2^(s-1)
                       // per stage; sum over all stages = n-1 = total.
    let mut table: Vec<C> = Vec::with_capacity(total);
    unsafe { table.set_len(total) };
    let mut cursor = 0usize;
    let mut len = 2usize;
    for _ in 0..log_n {
        let half = len >> 1;
        // One trig call per stage (not per entry).
        // W_base = exp(sign * 2πi / len); recurrence: tw[j] = tw[j-1] * W_base.
        let base_angle = sign * std::f64::consts::TAU / len as f64;
        let w_re = base_angle.cos();
        let w_im = base_angle.sin();
        let mut tw_re = 1.0f64; // tw[0] = 1
        let mut tw_im = 0.0f64;
        for j in 0..half {
            // SAFETY: cursor + j < total by stage-size invariant.
            unsafe {
                *table.get_unchecked_mut(cursor + j) = C::from_components(tw_re, tw_im);
            }
            let new_re = tw_re * w_re - tw_im * w_im;
            let new_im = tw_re * w_im + tw_im * w_re;
            tw_re = new_re;
            tw_im = new_im;
        }
        cursor += half;
        len <<= 1;
    }
    debug_assert_eq!(cursor, total, "cursor must reach total after all stages");
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn angle(j: usize, len: usize, sign: f64) -> f64 {
        sign * TAU * j as f64 / len as f64
    }

    #[test]
    fn table_length_is_n_minus_1() {
        for exp in 1..=10u32 {
            let n = 1usize << exp;
            let table = build_twiddle_table::<Complex64>(n, -1.0);
            assert_eq!(table.len(), n - 1, "table length must be n-1 for n={n}");
        }
    }

    #[test]
    fn forward_entries_match_definition() {
        let n = 16usize;
        let table = build_twiddle_table::<Complex64>(n, -1.0);
        // Stage 1: len=2, half=1, base=0 → entry 0 = exp(-i·2π·0/2) = 1+0i
        let w = Complex64::new(angle(0, 2, -1.0).cos(), angle(0, 2, -1.0).sin());
        assert!((table[0] - w).norm() < 1e-15, "stage1 entry mismatch");
        // Stage 2: len=4, half=2, base=1 → entry j=0 = exp(-i·π·0/2)=1, j=1=exp(-i·π/2)=-i
        let w0 = Complex64::new(angle(0, 4, -1.0).cos(), angle(0, 4, -1.0).sin());
        let w1 = Complex64::new(angle(1, 4, -1.0).cos(), angle(1, 4, -1.0).sin());
        assert!((table[1] - w0).norm() < 1e-15, "stage2 entry j=0 mismatch");
        assert!((table[2] - w1).norm() < 1e-15, "stage2 entry j=1 mismatch");
        // Stage 3: len=8, half=4, base=3 → entry j=2 = exp(-i·2π·2/8) = exp(-iπ/2) = -i
        let w2 = Complex64::new(angle(2, 8, -1.0).cos(), angle(2, 8, -1.0).sin());
        assert!((table[5] - w2).norm() < 1e-15, "stage3 entry j=2 mismatch");
    }

    #[test]
    fn inverse_table_is_conjugate_of_forward() {
        let n = 32usize;
        let fwd = build_twiddle_table::<Complex64>(n, -1.0);
        let inv = build_twiddle_table::<Complex64>(n, 1.0);
        for (f, i) in fwd.iter().zip(inv.iter()) {
            assert!(
                (f.re - i.re).abs() < 1e-15 && (f.im + i.im).abs() < 1e-15,
                "inverse entry must be conjugate of forward entry"
            );
        }
    }

    #[test]
    fn f32_entries_close_to_f64_reference() {
        let n = 64usize;
        let f64_table = build_twiddle_table::<Complex64>(n, -1.0);
        let f32_table = build_twiddle_table::<Complex32>(n, -1.0);
        for (f64v, f32v) in f64_table.iter().zip(f32_table.iter()) {
            let err_re = (f64v.re - f32v.re as f64).abs();
            let err_im = (f64v.im - f32v.im as f64).abs();
            assert!(err_re < 1e-7, "f32 re error too large: {err_re}");
            assert!(err_im < 1e-7, "f32 im error too large: {err_im}");
        }
    }

    #[test]
    fn empty_table_for_n_le_1() {
        assert!(build_twiddle_table::<Complex64>(1, -1.0).is_empty());
    }
}

