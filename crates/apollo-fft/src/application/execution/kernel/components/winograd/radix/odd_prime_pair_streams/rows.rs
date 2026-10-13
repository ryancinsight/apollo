//! Step-one pair formation followed by a register transpose across streams.
//!
//! For stream s, four contiguous complex inputs at positions 1..=4 pair
//! with positions 10..=7 in reverse order. A lane permutation reverses
//! complex pairs without exchanging real and imaginary components.
//! Transposing the resulting 8 x 8 real-lane tile puts each pair component
//! in the existing stream-major scratch row. Pair five and x[0] stay scalar;
//! y[0] adds the five sums in the incumbent m-ascending order. Thus the row
//! path preserves the scalar prologue's expression tree and output bits.

use core::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_permutevar8x32_ps, _mm256_set1_ps,
    _mm256_setr_epi32, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps, _mm256_xor_ps,
};
use eunomia::Complex32;

use super::super::odd_prime_pair_interleave::transpose8;
use super::StreamsScratch;

/// Form one four-pair tile for streams 0..8. The const selects sum or
/// difference while keeping one authoritative address and permutation map.
/// Separate tiles bound live registers rather than retaining both tiles
/// across the transpose on an ISA with sixteen vector registers.
///
/// # Safety
///
/// AVX2 must be established and each stream's eleven step-one inputs must
/// be in bounds. Every load reads exactly four complex slots: 1..=4 or
/// 7..=10. Stream starts need neither alignment nor consecutive row spacing.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pair_rows<const DIFFERENCE: bool>(
    base: &[Complex32],
    starts: &[usize; 11],
) -> [__m256; 8] {
    // SAFETY: the caller proves the feature and bounds contracts above;
    // Complex32 has the existing two-float interleaved representation.
    unsafe {
        let reverse_pairs = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
        let slots = base.as_ptr().cast::<f32>();
        core::array::from_fn(|s| {
            let a = _mm256_loadu_ps(slots.add((starts[s] + 1) * 2));
            let b = _mm256_permutevar8x32_ps(
                _mm256_loadu_ps(slots.add((starts[s] + 7) * 2)),
                reverse_pairs,
            );
            if DIFFERENCE {
                _mm256_sub_ps(a, b)
            } else {
                _mm256_add_ps(a, b)
            }
        })
    }
}

impl StreamsScratch {
    /// Vector row prologue for the first eight streams, with the incumbent
    /// scalar prologue for streams 8..11. No input-buffer writes occur.
    ///
    /// # Safety
    ///
    /// AVX2+FMA must be established. All eleven step-one input slots of
    /// each stream must be in bounds and the stream extents disjoint.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn prologue_rows<const INVERSE: bool>(
        &mut self,
        base: &[Complex32],
        starts: &[usize; 11],
    ) {
        // SAFETY: the caller establishes features and input bounds. Tile
        // loads stay within each eleven-slot row; every eight-lane store
        // stays within a typed eleven-element scratch row. All reads
        // precede the driver's epilogue, preserving in-place alias safety.
        unsafe {
            let sums = transpose8(pair_rows::<false>(base, starts));
            for m in 0..4 {
                _mm256_storeu_ps(self.sums_re[m].as_mut_ptr(), sums[2 * m]);
                _mm256_storeu_ps(self.sums_im[m].as_mut_ptr(), sums[2 * m + 1]);
            }
            let negate = _mm256_set1_ps(-0.0);
            let flip_re = if INVERSE { negate } else { _mm256_setzero_ps() };
            let flip_im = if INVERSE { _mm256_setzero_ps() } else { negate };
            let diffs = transpose8(pair_rows::<true>(base, starts));
            for m in 0..4 {
                _mm256_storeu_ps(
                    self.idiffs_re[m].as_mut_ptr(),
                    _mm256_xor_ps(diffs[2 * m + 1], flip_re),
                );
                _mm256_storeu_ps(
                    self.idiffs_im[m].as_mut_ptr(),
                    _mm256_xor_ps(diffs[2 * m], flip_im),
                );
            }
            let sign = if INVERSE { 1.0f32 } else { -1.0f32 };
            for s in 0..8 {
                let start = starts[s];
                let x0 = *base.get_unchecked(start);
                let a = *base.get_unchecked(start + 5);
                let b = *base.get_unchecked(start + 6);
                self.x0_re[s] = x0.re;
                self.x0_im[s] = x0.im;
                self.sums_re[4][s] = a.re + b.re;
                self.sums_im[4][s] = a.im + b.im;
                self.idiffs_re[4][s] = -(a.im - b.im) * sign;
                self.idiffs_im[4][s] = (a.re - b.re) * sign;
            }
            let mut acc_re = _mm256_loadu_ps(self.x0_re.as_ptr());
            let mut acc_im = _mm256_loadu_ps(self.x0_im.as_ptr());
            for m in 0..5 {
                acc_re = _mm256_add_ps(acc_re, _mm256_loadu_ps(self.sums_re[m].as_ptr()));
                acc_im = _mm256_add_ps(acc_im, _mm256_loadu_ps(self.sums_im[m].as_ptr()));
            }
            _mm256_storeu_ps(self.y0_re.as_mut_ptr(), acc_re);
            _mm256_storeu_ps(self.y0_im.as_mut_ptr(), acc_im);
            self.prologue_scalar::<INVERSE, 8, 11>(base, starts, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Complex32, StreamsScratch};
    use proptest::prelude::{any, Strategy};

    fn bits(scratch: &StreamsScratch) -> Vec<u32> {
        scratch
            .sums_re
            .iter()
            .flatten()
            .chain(scratch.sums_im.iter().flatten())
            .chain(scratch.idiffs_re.iter().flatten())
            .chain(scratch.idiffs_im.iter().flatten())
            .chain(&scratch.x0_re)
            .chain(&scratch.x0_im)
            .chain(&scratch.y0_re)
            .chain(&scratch.y0_im)
            .map(|value| value.to_bits())
            .collect()
    }

    fn compare<const INVERSE: bool>(
        base: &[Complex32],
        starts: &[usize; 11],
    ) -> (Vec<u32>, Vec<u32>) {
        let mut scalar = StreamsScratch::zeroed();
        let mut rows = StreamsScratch::zeroed();
        // SAFETY: the test checks AVX2+FMA before entering this helper;
        // its starts permute eleven disjoint eleven-element input rows.
        unsafe {
            scalar.prologue_scalar::<INVERSE, 0, 11>(base, starts, 1);
            rows.prologue_rows::<INVERSE>(base, starts);
        }
        (bits(&rows), bits(&scalar))
    }

    #[test]
    fn chunked_row_prologue_matches_scalar_scratch() {
        assert!(std::arch::is_x86_feature_detected!("avx2"));
        assert!(std::arch::is_x86_feature_detected!("fma"));
        let starts = core::array::from_fn(|s| ((s * 7) % 11) * 11);
        // All finite binary32 components, including subnormals and signed
        // zeros. Integer shrinking reduces exponent, significand, and sign
        // bits while retaining the finite-input domain. Every scratch value
        // is compared, before the k dots can conceal a lane-map defect.
        let finite = any::<u32>()
            .prop_filter("finite component", |word| word & 0x7f80_0000 != 0x7f80_0000)
            .prop_map(f32::from_bits);
        let inputs = proptest::collection::vec((finite.clone(), finite), 121);
        proptest::test_runner::TestRunner::default()
            .run(&inputs, |components| {
                let base: Vec<_> = components
                    .into_iter()
                    .map(|(re, im)| Complex32::new(re, im))
                    .collect();
                let (actual, expected) = compare::<false>(&base, &starts);
                proptest::prop_assert_eq!(actual, expected);
                let (actual, expected) = compare::<true>(&base, &starts);
                proptest::prop_assert_eq!(actual, expected);
                Ok(())
            })
            .expect("row and scalar prologues agree for finite components");
    }
}
