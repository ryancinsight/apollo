use crate::domain::contracts::math::bit_reverse_permute;

const PAR_THRESHOLD: usize = 1024;

/// Computes the NTT using a cached transposed twiddle vector splitting threaded array limits.
pub fn ntt_kernel(data: &mut [u64], twiddles: &[u64], modulus: u64) {
    let n = data.len();
    bit_reverse_permute(data);

    let mut offset = 0;
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let layer_twiddles = &twiddles[offset..offset + half];

        if len >= PAR_THRESHOLD {
            moirai::for_each_chunk_mut_with::<moirai::Adaptive, _, _>(data, len, |chunk| {
                hermes_simd::ntt_butterfly_stage_u64(chunk, len, layer_twiddles, modulus)
                    .expect("NTT stage chunk shape is validated by plan construction");
            });
        } else {
            hermes_simd::ntt_butterfly_stage_u64(data, len, layer_twiddles, modulus)
                .expect("NTT stage shape is validated by plan construction");
        }

        offset += half;
        len *= 2;
    }
}
