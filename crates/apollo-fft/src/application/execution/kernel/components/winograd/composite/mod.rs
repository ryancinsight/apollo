mod medium;
pub(crate) mod power;
mod radix_four_eight;
mod small;

/// The split variants of the generated composite codelets (column/row phases
/// in `#[inline(never)]` helpers). The fused bodies delegate to them when the
/// scalar's `prefers_split_codelet` gate selects the split for that length;
/// the equivalence test below asserts the split computes the identical
/// transform before any timing is read.
pub(crate) mod split {
    pub(crate) use super::medium::{dft144_cols, dft144_rows};
    pub(crate) use super::small::{dft50_cols, dft50_rows};
}

#[cfg(test)]
mod split_equivalence_tests {
    use super::split::{dft144_cols, dft144_rows, dft50_cols, dft50_rows};
    use super::{dft144_impl, dft50_impl};
    use eunomia::{Complex, Complex32, Complex64};
    fn source_f64(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|i| {
                let x = i as f64;
                Complex::new((0.017 * x).sin(), (0.031 * x).cos())
            })
            .collect()
    }

    fn source_f32(n: usize) -> Vec<Complex32> {
        (0..n)
            .map(|i| {
                let x = i as f64;
                Complex::new((0.017 * x).sin() as f32, (0.031 * x).cos() as f32)
            })
            .collect()
    }

    /// The split variant must compute the bit-identical transform of the
    /// fused body: it is generated from the same blocks in the same order,
    /// so any difference is a generator bug that would invalidate the
    /// measured routing decision built on it.
    #[test]
    fn split_variant_is_bit_identical_to_fused_body() {
        // f64, n=50: fused entry (gate off) vs the split helpers driven by hand.
        let mut fused_out = source_f64(50).try_into().unwrap();
        dft50_impl::<f64, false>(&mut fused_out);
        let mut split_out = source_f64(50).try_into().unwrap();
        let mut scratch = [Complex64::new(0.0, 0.0); 50];
        dft50_rows::<f64, false>(&split_out, &mut scratch);
        dft50_cols::<f64, false>(&mut scratch, &mut split_out);
        assert_eq!(
            fused_out, split_out,
            "the split variant of dft50 diverges from the fused body (f64)"
        );

        // f32, n=50.
        let mut fused_out = source_f32(50).try_into().unwrap();
        dft50_impl::<f32, false>(&mut fused_out);
        let mut split_out = source_f32(50).try_into().unwrap();
        let mut scratch = [Complex32::new(0.0, 0.0); 50];
        dft50_rows::<f32, false>(&split_out, &mut scratch);
        dft50_cols::<f32, false>(&mut scratch, &mut split_out);
        assert_eq!(
            fused_out, split_out,
            "the split variant of dft50 diverges from the fused body (f32)"
        );

        // f64 and f32, n=144 (CT 12×12).
        let mut fused_out = source_f64(144).try_into().unwrap();
        dft144_impl::<f64, false>(&mut fused_out);
        let mut split_out = source_f64(144).try_into().unwrap();
        let mut scratch = [Complex64::new(0.0, 0.0); 144];
        dft144_cols::<f64, false>(&split_out, &mut scratch);
        dft144_rows::<f64, false>(&mut scratch, &mut split_out);
        assert_eq!(
            fused_out, split_out,
            "the split variant of dft144 diverges from the fused body (f64)"
        );

        let mut fused_out = source_f32(144).try_into().unwrap();
        dft144_impl::<f32, false>(&mut fused_out);
        let mut split_out = source_f32(144).try_into().unwrap();
        let mut scratch = [Complex32::new(0.0, 0.0); 144];
        dft144_cols::<f32, false>(&split_out, &mut scratch);
        dft144_rows::<f32, false>(&mut scratch, &mut split_out);
        assert_eq!(
            fused_out, split_out,
            "the split variant of dft144 diverges from the fused body (f32)"
        );
    }
}

pub(crate) use medium::{
    dft108_impl, dft112_impl, dft120_impl, dft121_impl, dft126_impl, dft144_impl, dft154_impl,
    dft168_impl, dft180_impl, dft189_impl, dft222_impl, dft242_impl, dft246_impl, dft259_impl,
    dft275_impl, dft280_impl, dft296_impl, dft363_impl, dft400_impl, dft484_impl, dft72_impl,
    dft96_impl, dft99_impl,
};
pub(crate) use power::{dft128_impl, dft16_impl, dft32_impl, dft64_impl};
pub(crate) use radix_four_eight::{
    try_dft16_hardware, try_dft32_hardware, try_dft32_rows_hardware,
};
pub(crate) use small::{
    dft10_impl,
    dft12_impl,
    dft14_impl,
    dft18_impl,
    // Phase 3: coprime composites
    dft20_impl,
    dft21_impl,
    dft22_impl,
    dft24_impl,
    dft25_impl,
    // Phase 2: 2×prime twiddle-free WGT codelets (primes 13–23, N ≤ 46)
    dft26_impl,
    dft27_impl,
    dft28_impl,
    dft30_impl,
    dft33_impl,
    dft34_impl,
    dft35_impl,
    dft36_impl,
    dft38_impl,
    dft39_impl,
    dft40_impl,
    dft42_impl,
    dft44_impl,
    dft45_impl,
    dft46_impl,
    dft48_impl,
    // N 49–63: coprime WGT + 49=7² Cooley-Tukey
    dft49_impl,
    dft50_impl,
    dft51_impl,
    dft52_impl,
    dft54_impl,
    dft55_impl,
    dft56_impl,
    dft58_impl,
    dft60_impl,
    dft62_impl,
    dft63_impl,
    // Original composites
    dft6_impl,
    dft81_impl,
    dft9_impl,
};
