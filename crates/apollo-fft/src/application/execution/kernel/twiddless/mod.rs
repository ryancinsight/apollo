//! Twiddless FFT (Queiroz 2025) executed beside its butterfly isomorph.
//!
//! ## Algorithm as published
//!
//! Queiroz, "Fast compressed-domain N-point discrete Fourier transform: the
//! 'twiddless' FFT algorithm", arXiv:2505.23718v2 (2025), Algorithm 1, computes
//! an `N`-point DFT for `N = c · 2^k` through two rectangular-index-coefficient
//! compressions per level,
//!
//! ```text
//! x̂e[n] =  x[n] + x[n + N/2]                       n = 0 .. N/2     (line 9)
//! x̂o[n] = (x[n] − x[n + N/2]) · exp(−2πi·n / N)    n = 0 .. N/2     (line 10)
//! ```
//!
//! recursing on both halves and interleaving the results, `X[2c] = X̂e[c]` and
//! `X[2c + 1] = X̂o[c]` (lines 11 to 14). Recursion stops at `N ≤ 5`, where the
//! base case evaluates the DFT directly (lines 6 and 7). The paper states the
//! real operation count `T(N) = 2T(N/2) + 7N − 4 ≈ 7N·log₂N` against
//! `5N·log₂N` for radix-2 Cooley–Tukey and claims structural advantages: no
//! butterfly, a combination step that is only index reordering, and lengths
//! `c · 2^k` without zero padding.
//!
//! ## What the split is
//!
//! Lines 9 and 10 are the Gentleman–Sande decimation-in-frequency butterfly
//! (Gentleman and Sande, AFIPS 1966; Van Loan, *Computational Frameworks for
//! the FFT*, 1992, section 1.3). The modulation `exp(−2πi·n / N)` applied to
//! the odd half is the twiddle factor `W_N^n`; the paper counts it as `3N`
//! real operations, which is the twiddle multiplication under another name.
//! Writing the two compressed halves contiguously as `[x̂e | x̂o]` is the
//! in-place DIF stage's own layout, and recursing on both halves leaves the
//! `2^k` leaf blocks in bit-reversed residue order, so the index reordering of
//! lines 12 and 14 is the DIF bit-reversal gather. Apollo already runs this
//! stage set in production on the planar four-step's second axis
//! (`components::batched::dif`).
//!
//! The isomorphism is not argued here, it is executed. [`TwiddlessPlan`]
//! carries one arithmetic body and three [`Schedule`]s that differ only in
//! data movement. Every element's floating-point operation sequence is
//! identical across schedules, so their outputs agree bitwise (asserted by
//! test), and a measured difference between them is buffer traffic alone.
//!
//! - [`CompressedHalves`]: Algorithm 1 as published. Each level reads its
//!   node and writes both compressed halves into the other buffer, depth first.
//! - [`ButterflyRecursive`]: the same depth-first traversal with the
//!   butterfly applied in place, isolating the out-of-place write.
//! - [`ButterflyIterative`]: the classic breadth-first stage loop.
//!
//! ## Numerical contract
//!
//! Twiddles are evaluated directly through the crate's single twiddle
//! authority (`twiddle_table::twiddle_components`), never by a recurrence, so
//! the forward error obeys Higham, *Accuracy and Stability of Numerical
//! Algorithms*, 2nd ed., Theorem 24.2: with `t` halving levels and twiddle
//! relative error `μ`, `‖X̂ − X‖₂ ≤ tη / (1 − tη) · ‖X‖₂` where
//! `η = μ + γ₄(1 + μ)`. The tests instantiate that bound.
//!
//! ## Scope
//!
//! This module is a measurement instrument compiled under the same boundary
//! as `benchmark_kernels`: forward transform only, lengths whose odd part is
//! 1, 3 or 5 exactly as the published base case admits, and no production
//! route. ADR 0052 (`docs/adr/0052-twiddless-fft-evaluation.md`) records the
//! evaluation and its decision.

mod schedule;
#[cfg(test)]
mod tests;

use eunomia::{Complex, Complex32, Complex64, NumericElement};

use super::twiddle_table::{twiddle_components, TwiddleOutput};

pub use schedule::{ButterflyIterative, ButterflyRecursive, CompressedHalves, Schedule};

/// The published base case: recursion stops once a node is this short or shorter.
const BASE_CASE_MAX: usize = 5;

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Scalars the instrument instantiates: Apollo's CPU compute scalars.
///
/// Compact `F16` storage reaches the CPU kernels through the `f32` bridge, so
/// it has no separate instantiation here either.
pub trait TwiddlessScalar: NumericElement + private::Sealed {
    /// `cos + i·sin` narrowed from the `f64` evaluation, the same narrowing the
    /// production twiddle tables apply.
    fn unit(cos: f64, sin: f64) -> Complex<Self>;
}

impl TwiddlessScalar for f32 {
    #[inline]
    fn unit(cos: f64, sin: f64) -> Complex<Self> {
        <Complex32 as TwiddleOutput>::from_components(cos, sin)
    }
}

impl TwiddlessScalar for f64 {
    #[inline]
    fn unit(cos: f64, sin: f64) -> Complex<Self> {
        <Complex64 as TwiddleOutput>::from_components(cos, sin)
    }
}

/// A length Algorithm 1 does not admit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TwiddlessLengthError {
    /// The transform length is zero.
    #[error("twiddless FFT length must be non-zero")]
    Zero,
    /// Halving never reaches the published base case `N ≤ 5`.
    #[error(
        "twiddless FFT length {len} has odd part {odd_part}; Algorithm 1's base \
         case admits odd parts 1, 3 and 5 only"
    )]
    OddPart {
        /// The rejected length.
        len: usize,
        /// Its odd part, which exceeds the base case.
        odd_part: usize,
    },
}

/// Forward plan for one length: leaf size, level count and the full-period
/// twiddle table `W_N^j = exp(−2πi·j / N)` for `j < N`.
///
/// One table serves every level and the leaf: level `ℓ` reads `W_N^{n·2^ℓ}`
/// and a leaf of length `c` reads `W_N^{jm·(N/c)}`.
#[derive(Clone, Debug)]
pub struct TwiddlessPlan<F> {
    len: usize,
    leaf: usize,
    levels: u32,
    table: Box<[Complex<F>]>,
}

impl<F: TwiddlessScalar> TwiddlessPlan<F> {
    /// Builds the plan for `len`, or reports why Algorithm 1 rejects it.
    ///
    /// # Errors
    ///
    /// [`TwiddlessLengthError::Zero`] for `len == 0`;
    /// [`TwiddlessLengthError::OddPart`] when halving cannot reach `N ≤ 5`.
    pub fn new(len: usize) -> Result<Self, TwiddlessLengthError> {
        if len == 0 {
            return Err(TwiddlessLengthError::Zero);
        }
        let mut leaf = len;
        let mut levels = 0_u32;
        while leaf > BASE_CASE_MAX && leaf % 2 == 0 {
            leaf /= 2;
            levels += 1;
        }
        if leaf > BASE_CASE_MAX {
            return Err(TwiddlessLengthError::OddPart {
                len,
                odd_part: len >> len.trailing_zeros(),
            });
        }
        let table = (0..len)
            .map(|exponent| {
                let (sin, cos) = twiddle_components(-1.0, exponent, len);
                F::unit(cos, sin)
            })
            .collect();
        Ok(Self {
            len,
            leaf,
            levels,
            table,
        })
    }

    /// Transform length `N`.
    #[must_use]
    pub const fn transform_len(&self) -> usize {
        self.len
    }

    /// Base-case block length `c ∈ {1, 2, 3, 4, 5}`.
    #[must_use]
    pub const fn leaf(&self) -> usize {
        self.leaf
    }

    /// Number of halving levels `k` with `N = leaf · 2^k`.
    #[must_use]
    pub const fn levels(&self) -> u32 {
        self.levels
    }

    /// Scratch length every schedule requires: one buffer of `N` elements.
    #[must_use]
    pub const fn scratch_len(&self) -> usize {
        self.len
    }

    /// Forward DFT of `input` into `output` under schedule `S`.
    ///
    /// Output is in natural order and unnormalized, matching the crate's
    /// forward convention `X_k = Σ x_n · exp(−2πi·k·n / N)`.
    ///
    /// # Panics
    ///
    /// Panics if `input`, `output` or `scratch` differ from the plan length.
    #[track_caller]
    pub fn forward<S: Schedule>(
        &self,
        input: &[Complex<F>],
        output: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
    ) {
        assert_eq!(input.len(), self.len, "input length differs from plan");
        assert_eq!(output.len(), self.len, "output length differs from plan");
        assert_eq!(scratch.len(), self.len, "scratch length differs from plan");
        S::forward(self, input, output, scratch);
    }

    pub(super) fn table(&self) -> &[Complex<F>] {
        &self.table
    }
}
