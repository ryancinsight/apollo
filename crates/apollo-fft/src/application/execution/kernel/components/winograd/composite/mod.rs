mod large;
mod medium;
pub(crate) mod power;
mod small;

pub(crate) use large::{
    dft100_impl, dft102_impl, dft104_impl, dft106_impl, dft108_impl, dft112_impl, dft120_impl,
    dft126_impl, dft144_impl, dft148_impl, dft160_impl, dft166_impl, dft168_impl, dft172_impl,
    dft176_impl, dft180_impl, dft192_impl, dft200_impl, dft66_impl, dft70_impl, dft72_impl,
    dft78_impl, dft80_impl, dft82_impl, dft88_impl, dft96_impl,
};
pub(crate) use medium::{
    dft121_impl, dft154_impl, dft189_impl, dft222_impl, dft242_impl, dft246_impl, dft259_impl,
    dft275_impl, dft280_impl, dft296_impl, dft363_impl, dft400_impl, dft484_impl, dft99_impl,
};
pub(crate) use power::{dft128_impl, dft16_impl, dft32_impl, dft64_impl};
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
