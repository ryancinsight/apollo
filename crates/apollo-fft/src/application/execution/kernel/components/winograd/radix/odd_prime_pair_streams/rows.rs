//! Step-one pair formation in four-stream register tiles.
//!
//! One AVX2 register contains four interleaved complex pairs. Four streams
//! therefore transpose into real/imaginary half-registers without the
//! eight-row tile's cross-128-bit exchanges. Pair positions stay (1,10),
//! (2,9), (3,8), (4,7), (5,6); y[0] keeps the scalar m-ascending expression
//! tree. Two four-stream tiles fill the existing eight-stream vector region;
//! pair five and streams 8..11 keep the incumbent scalar arithmetic.

use core::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps, _mm256_xor_ps,
    _mm_storeu_ps,
};
use eunomia::Complex32;

use super::super::odd_prime_pair_interleave::{permute_qwords, transpose_pairs};
use super::StreamsScratch;

/// Form four pairs in four streams. The consts select the stream tile and
/// sum/difference while retaining one address and permutation map.
///
/// # Safety
///
/// AVX2 must be established; each stream's eleven step-one inputs must be
/// in bounds. S0 is 0 or 4. Every load reads exactly positions 1..=4 or
/// 7..=10, without requiring row alignment or consecutive stream starts.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pair_rows<const S0: usize, const DIFFERENCE: bool>(
    base: &[Complex32],
    starts: &[usize; 11],
) -> [__m256; 4] {
    const { assert!(S0 == 0 || S0 == 4) };
    // SAFETY: feature and input bounds follow the caller's contract. A
    // 64-bit granule is exactly one Complex32; reversal preserves its
    // real/imaginary order without a precision conversion.
    unsafe {
        let slots = base.as_ptr().cast::<f32>();
        core::array::from_fn(|s| {
            let start = starts[S0 + s];
            let a = _mm256_loadu_ps(slots.add((start + 1) * 2));
            let b = permute_qwords::<0x1B>(_mm256_loadu_ps(slots.add((start + 7) * 2)));
            if DIFFERENCE {
                _mm256_sub_ps(a, b)
            } else {
                _mm256_add_ps(a, b)
            }
        })
    }
}

impl StreamsScratch {
    /// Fill one sum or difference tile, transposing pair components across
    /// four streams. Low halves hold pairs 0/1; high halves hold pairs 2/3.
    ///
    /// # Safety
    ///
    /// AVX2+FMA and all step-one input bounds must be established; S0 is
    /// 0 or 4, so every four-lane store stays within its eleven-lane row.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn write_pairs<const S0: usize, const INVERSE: bool, const DIFFERENCE: bool>(
        &mut self,
        base: &[Complex32],
        starts: &[usize; 11],
    ) {
        // SAFETY: the caller establishes features and input bounds; S0's
        // compile-time assertion bounds both input indices and scratch stores.
        unsafe {
            let pairs = transpose_pairs(pair_rows::<S0, DIFFERENCE>(base, starts));
            let negate = _mm256_set1_ps(-0.0);
            let flip_re = if INVERSE { negate } else { _mm256_setzero_ps() };
            let flip_im = if INVERSE { _mm256_setzero_ps() } else { negate };
            let (dest_re, dest_im) = if DIFFERENCE {
                (&mut self.idiffs_re, &mut self.idiffs_im)
            } else {
                (&mut self.sums_re, &mut self.sums_im)
            };
            for m in 0..2 {
                let re = pairs[2 * m];
                let im = pairs[2 * m + 1];
                let (re, im) = if DIFFERENCE {
                    (_mm256_xor_ps(im, flip_re), _mm256_xor_ps(re, flip_im))
                } else {
                    (re, im)
                };
                _mm_storeu_ps(dest_re[m].as_mut_ptr().add(S0), _mm256_castps256_ps128(re));
                _mm_storeu_ps(dest_im[m].as_mut_ptr().add(S0), _mm256_castps256_ps128(im));
                _mm_storeu_ps(
                    dest_re[m + 2].as_mut_ptr().add(S0),
                    _mm256_extractf128_ps::<1>(re),
                );
                _mm_storeu_ps(
                    dest_im[m + 2].as_mut_ptr().add(S0),
                    _mm256_extractf128_ps::<1>(im),
                );
            }
        }
    }

    /// Vector row prologue for eight streams, with the scalar tail intact.
    ///
    /// # Safety
    ///
    /// AVX2+FMA must be established. Every stream's eleven step-one slots
    /// must be in bounds and its extent disjoint from the other streams.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn prologue_rows<const INVERSE: bool>(
        &mut self,
        base: &[Complex32],
        starts: &[usize; 11],
    ) {
        // SAFETY: input bounds and features follow the caller's contract;
        // every store names an initialized scratch field. All input reads
        // complete before the driver's epilogue writes the input buffer.
        unsafe {
            self.write_pairs::<0, INVERSE, false>(base, starts);
            self.write_pairs::<0, INVERSE, true>(base, starts);
            self.write_pairs::<4, INVERSE, false>(base, starts);
            self.write_pairs::<4, INVERSE, true>(base, starts);
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
