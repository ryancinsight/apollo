//! Fixed-array DFT codelets for selected medium composite lengths.

apollo_fft_macros::generate_winograd_composites! {
    inline_attr: hint,
    gt_pairs: [
        (8, 9), // dft72_impl
        (3, 32), // dft96_impl
        (9, 11), // dft99_impl
        (4, 27), // dft108_impl
        (16, 7), // dft112_impl
        (15, 8), // dft120_impl
        (2, 63), // dft126_impl
        (11, 14), // dft154_impl
        (20, 9), // dft180_impl
        (2, 121), // dft242_impl
        (11, 25), // dft275_impl
        (8, 35), // dft280_impl
        (3, 121), // dft363_impl
        (16, 25), // dft400_impl
    ],
    ct_pairs: [
        (11, 11), // dft121_impl
        (12, 12), // dft144_impl
        (14, 12), // dft168_impl
        (21, 9), // dft189_impl
        (22, 22), // dft484_impl
    ],
    pp_pairs: [],
}
