//! Provider-neutral radix-two execution stages.

use super::kernel::FftParams;

/// Value parameters for one typed radix-two execution plan.
///
/// Accelerator providers own parameter transfer and ordered dispatch. Apollo
/// retains only the recurrence values derived from the FFT length.
pub(crate) struct RadixStages {
    pub(crate) bit_reverse: FftParams,
    pub(crate) butterflies: Box<[FftParams]>,
    pub(crate) inverse_scale: Option<FftParams>,
    pub(crate) fft_len: u32,
    pub(crate) batch_count: u32,
    pub(crate) radix_four: bool,
}

impl RadixStages {
    pub(crate) fn empty() -> Self {
        Self {
            bit_reverse: FftParams {
                n: 0,
                stage: 0,
                inverse: 0,
                batch_count: 0,
            },
            butterflies: Box::default(),
            inverse_scale: None,
            fft_len: 0,
            batch_count: 0,
            radix_four: false,
        }
    }

    pub(crate) fn radix_two(fft_len: u32, batch_count: u32, inverse: bool) -> Self {
        let inverse_flag = u32::from(inverse);
        let butterflies = (0..fft_len.trailing_zeros())
            .map(|stage| FftParams {
                n: fft_len,
                stage,
                inverse: inverse_flag,
                batch_count,
            })
            .collect();
        Self {
            bit_reverse: FftParams {
                n: fft_len,
                stage: 0,
                inverse: inverse_flag,
                batch_count,
            },
            butterflies,
            inverse_scale: inverse.then_some(FftParams {
                n: fft_len,
                stage: 0,
                inverse: 1,
                batch_count,
            }),
            fft_len,
            batch_count,
            radix_four: false,
        }
    }

    pub(crate) fn radix_four(fft_len: u32, batch_count: u32, inverse: bool) -> Self {
        let inverse_flag = u32::from(inverse);
        let butterflies = (0..(fft_len.trailing_zeros() / 2))
            .map(|stage| FftParams {
                n: fft_len,
                stage,
                inverse: inverse_flag,
                batch_count,
            })
            .collect();
        Self {
            bit_reverse: FftParams {
                n: fft_len,
                stage: 0,
                inverse: inverse_flag,
                batch_count,
            },
            butterflies,
            inverse_scale: inverse.then_some(FftParams {
                n: fft_len,
                stage: 0,
                inverse: 1,
                batch_count,
            }),
            fft_len,
            batch_count,
            radix_four: true,
        }
    }
}
