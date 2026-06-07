//! Inline fixed-array DFT codelets for small composite lengths.
//!
//! Two macro invocations:
//! 1. `#[inline]` for N ≤ 36 — safe for all stack depths including
//!    debug-mode test harnesses.
//! 2. `#[inline]` for 37 ≤ N ≤ 48 — avoids forced inlining that overflows the
//!    debug stack when codelets are deeply nested.

// N ≤ 36 → forced inlining is safe; scratch ≤ 576 bytes per f64.
apollo_fft_macros::generate_winograd_composites! {
    gt_pairs: [
        // Original pairs
        (2, 3),   // dft6_impl
        (2, 5),   // dft10_impl
        (4, 3),   // dft12_impl
        (2, 7),   // dft14_impl
        (9, 2),   // dft18_impl
        (2, 11),  // dft22_impl
        (9, 4),   // dft36_impl
        // Phase 3: N ≤ 36
        (5, 4),   // dft20_impl
        (7, 3),   // dft21_impl
        (7, 4),   // dft28_impl
        (5, 6),   // dft30_impl
        (11, 3),  // dft33_impl
        (7, 5),   // dft35_impl
    ],
    ct_pairs: [
        (3, 9),   // dft27_impl: 3³ Cooley-Tukey
        (6, 4),   // dft24_impl: 6×4 Cooley-Tukey
        (3, 27),  // dft81_impl: 3⁴ Cooley-Tukey
        (5, 5),   // dft25_impl: 5² Cooley-Tukey
    ],
    pp_pairs: [
        (3, 2),   // dft9_impl:  3² prime-power Winograd-Rader
    ],
}

// 37 ≤ N ≤ 63 → hint-inline only; scratch 592–1008 bytes for f64.
apollo_fft_macros::generate_winograd_composites! {
    inline_attr: hint,
    gt_pairs: [
        // 37 ≤ N ≤ 48 (coprime)
        (13, 3),  // dft39_impl
        (5, 8),   // dft40_impl
        (7, 6),   // dft42_impl
        (11, 4),  // dft44_impl
        (9, 5),   // dft45_impl
        (3, 16),  // dft48_impl
        // 2×prime ≤ 46
        (2, 13),  // dft26_impl
        (2, 17),  // dft34_impl
        (2, 19),  // dft38_impl
        (2, 23),  // dft46_impl
        // 49 ≤ N ≤ 63 (coprime)
        (2, 25),  // dft50_impl
        (17, 3),  // dft51_impl
        (13, 4),  // dft52_impl
        (2, 27),  // dft54_impl
        (11, 5),  // dft55_impl
        (7, 8),   // dft56_impl
        (2, 29),  // dft58_impl
        (5, 12),  // dft60_impl
        (2, 31),  // dft62_impl
        (7, 9),   // dft63_impl
    ],
    ct_pairs: [
        (7, 7),   // dft49_impl: 7² Cooley-Tukey
    ],
    pp_pairs: [],
}
