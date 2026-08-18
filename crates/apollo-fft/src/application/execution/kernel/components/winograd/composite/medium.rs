//! Fixed-array DFT codelets for selected medium composite lengths (N < 64 or not in large.rs).
//!
//! Sizes 72, 96, 108, 112, 120, 126, 144, 168, and 180 are covered by
//! `large.rs` with `#[inline(never)]`; they are not repeated here.

apollo_fft_macros::generate_winograd_composites! {
    inline_attr: hint,
    gt_pairs: [
        (9, 11),  // dft99_impl
        (11, 14), // dft154_impl
        (2, 121), // dft242_impl
        (11, 25), // dft275_impl
        (8, 35),  // dft280_impl
        (3, 121), // dft363_impl
        (16, 25), // dft400_impl
        (6, 37),  // dft222_impl
        (6, 41),  // dft246_impl
        (7, 37),  // dft259_impl
        (8, 37),  // dft296_impl
    ],
    ct_pairs: [
        (11, 11), // dft121_impl
        (21, 9),  // dft189_impl
        (22, 22), // dft484_impl
    ],
    pp_pairs: [],
}
