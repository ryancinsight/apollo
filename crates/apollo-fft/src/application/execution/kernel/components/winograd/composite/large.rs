//! Fixed-array DFT codelets for large composite lengths (64 ≤ N ≤ 200).
//!
//! These codelets use `#[inline(never)]` to prevent code bloat and debug-mode
//! stack overflow. Each size is factored as a Good-Thomas PFA pair (n1, n2)
//! with gcd(n1, n2) = 1, calling `ShortDft<n1>` and `ShortDft<n2>` directly.
//! This eliminates the strided scatter-gather in `pfa_fft_natural_inplace`.

// 64 ≤ N ≤ 200: never-inline; scratch 1024–3200 bytes for f64.
apollo_fft_macros::generate_winograd_composites! {
    inline_attr: never,
    gt_pairs: [
        // N = 66 = 2 × 33: gcd(2,33)=1 — needed as ShortDft<66> for static Rader P=67
        (2, 33),  // dft66_impl
        // N = 70 = 2 × 35: gcd(2,35)=1 — needed as ShortDft<70> for static Rader P=71
        (2, 35),  // dft70_impl
        // N = 72 = 8 × 9: gcd(8,9)=1
        (8, 9),   // dft72_impl
        // N = 78 = 2 × 39: gcd(2,39)=1 — needed as ShortDft<78> for static Rader P=79
        (2, 39),  // dft78_impl
        // N = 80 = 16 × 5: gcd(16,5)=1
        (5, 16),  // dft80_impl
        // N = 82 = 2 × 41: gcd(2,41)=1 — needed as ShortDft<82> for static Rader P=83
        (2, 41),  // dft82_impl
        // N = 88 = 8 × 11: gcd(8,11)=1 — needed as ShortDft<88> for static Rader P=89
        (8, 11),  // dft88_impl
        // N = 96 = 32 × 3: gcd(32,3)=1
        (3, 32),  // dft96_impl
        // N = 100 = 4 × 25: gcd(4,25)=1 — needed as ShortDft<100> for static Rader P=101
        (4, 25),  // dft100_impl
        // N = 102 = 2 × 51: gcd(2,51)=1 — needed as ShortDft<102> for static Rader P=103
        (2, 51),  // dft102_impl
        // N = 104 = 8 × 13: gcd(8,13)=1
        (8, 13),  // dft104_impl
        // N = 106 = 2 × 53: gcd(2,53)=1 — needed as ShortDft<106> for static Rader P=107
        (2, 53),  // dft106_impl
        // N = 108 = 4 × 27: gcd(4,27)=1
        (4, 27),  // dft108_impl
        // N = 112 = 16 × 7: gcd(16,7)=1
        (7, 16),  // dft112_impl
        // N = 120 = 8 × 15: gcd(8,15)=1
        (8, 15),  // dft120_impl
        // N = 126 = 2 × 63: gcd(2,63)=1 — needed as ShortDft<126> for static Rader P=127
        (2, 63),  // dft126_impl
        // N = 144 = 16 × 9: gcd(16,9)=1
        (9, 16),  // dft144_impl
        // N = 148 = 4 × 37: gcd(4,37)=1 — needed as ShortDft<148> for static Rader P=149
        (4, 37),  // dft148_impl
        // N = 160 = 32 × 5: gcd(32,5)=1
        (5, 32),  // dft160_impl
        // N = 166 = 2 × 83: gcd(2,83)=1 — needed as ShortDft<166> for static Rader P=167
        // ShortDft<83> is backed by rader_fft_83 (rader_prime_impl).
        (2, 83),  // dft166_impl
        // N = 168 = 8 × 21: gcd(8,21)=1
        (8, 21),  // dft168_impl
        // N = 172 = 4 × 43: gcd(4,43)=1 — needed as ShortDft<172> for static Rader P=173
        (4, 43),  // dft172_impl
        // N = 176 = 16 × 11: gcd(16,11)=1
        (11, 16), // dft176_impl
        // N = 180 = 9 × 20: gcd(9,20)=1
        (9, 20),  // dft180_impl
        // N = 192 = 64 × 3: gcd(64,3)=1
        (3, 64),  // dft192_impl
        // N = 200 = 8 × 25: gcd(8,25)=1
        (8, 25),  // dft200_impl
    ],
    ct_pairs: [],
    pp_pairs: [],
}
