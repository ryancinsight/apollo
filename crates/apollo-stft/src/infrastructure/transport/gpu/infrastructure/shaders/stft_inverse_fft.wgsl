// Short-Time Fourier Transform inverse kernel — FFT-accelerated.
//
// Replaces the O(N²) per-frame direct IDFT with a batched Cooley-Tukey Radix-2 DIT IFFT,
// reducing per-frame complexity from O(N²) to O(N log N).
//
// Dispatch sequence (all passes encoded in one CommandEncoder):
//   Pass 1 (stft_deinterleave):   interleaved f32 spectrum → split re/im scratch buffers.
//   Pass 2 (stft_bitrev):         bit-reversal permutation on split re/im scratch.
//   Pass 3 (stft_butterfly):      one Radix-2 DIT butterfly stage; dispatched log₂(N) times,
//                                 each with a distinct params bind group carrying the stage index.
//   Pass 4 (stft_scale_and_window): scale by 1/N, apply Hann synthesis window → frame_data.
//
// After Pass 4 the host encodes an OLA pass using stft_inverse.wgsl::stft_inverse_ola
// unchanged (3-binding layout, StftParams uniform).
//
// Bind group layout:
//   group 0 — data buffers (shared by all four passes in this file):
//     binding 0: read-only storage  — interleaved complex spectrum f32 [2·frame_count·frame_len]
//     binding 1: read_write storage — re scratch                      f32 [frame_count·frame_len]
//     binding 2: read_write storage — im scratch                      f32 [frame_count·frame_len]
//     binding 3: read_write storage — frame_data output               f32 [frame_count·frame_len]
//   group 1 — per-stage parameters (one distinct bind group per dispatch):
//     binding 0: uniform — FftStageParams (16 bytes)
//
// Formal bases:
//   Cooley-Tukey Radix-2 DIT FFT: Cooley & Tukey (1965); Brigham (1988) §8.3.
//   IDFT twiddle: x[j] = (1/N)·Σ_k X[k]·exp(+2πi·k·j/N); twiddle W_N^{−k} = exp(+2πi·k/N).
//   WOLA synthesis window: Allen & Rabiner (1977) Theorem 1.

// Per-stage parameters for the FFT inverse passes.
// Sized to 16 bytes (4×u32) to satisfy WGPU uniform alignment.
struct FftStageParams {
    frame_count: u32,  // number of STFT frames (batch dimension)
    frame_len:   u32,  // FFT length per frame; must be a power of two
    stage:       u32,  // butterfly stage index: 0 = least-significant half, log₂(N)−1 = MSB half
    _pad:        u32,  // alignment padding
}

// Group 0: data buffers shared across all four FFT inverse passes.
@group(0) @binding(0) var<storage, read>       spectrum:   array<f32>;
@group(0) @binding(1) var<storage, read_write> re_scratch: array<f32>;
@group(0) @binding(2) var<storage, read_write> im_scratch: array<f32>;
@group(0) @binding(3) var<storage, read_write> frame_data: array<f32>;

// Group 1: per-stage parameters.
@group(1) @binding(0) var<uniform> fft_params: FftStageParams;

const TAU: f32 = 6.28318530717958647692;

// Hann synthesis window: w[j] = 0.5 − 0.5·cos(TAU·j / (frame_len − 1)).
// Returns 1.0 for the degenerate single-sample case (frame_len ≤ 1).
fn hann_window(j: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(TAU * f32(j) / f32(frame_len - 1u));
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1: stft_deinterleave
//
// Converts the interleaved complex spectrum layout to split re/im scratch buffers.
//
// Input layout:  spectrum[2·(m·N + k)]     = Re{X[m, k]}
//                spectrum[2·(m·N + k) + 1] = Im{X[m, k]}
// Output layout: re_scratch[m·N + k]        = Re{X[m, k]}
//                im_scratch[m·N + k]         = Im{X[m, k]}
//
// Total invocations: frame_count · frame_len.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_deinterleave(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = fft_params.frame_count * fft_params.frame_len;
    if idx >= total {
        return;
    }
    re_scratch[idx] = spectrum[2u * idx];
    im_scratch[idx] = spectrum[2u * idx + 1u];
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: stft_bitrev
//
// Applies the Cooley-Tukey bit-reversal permutation in-place on the split re/im
// scratch buffers, independently per frame (batched).
//
// Thread gid.x covers the joint (frame_m, local_i) space: gid.x = frame_m·N + local_i.
// Only the upper-half pairs (local_j > local_i) are swapped to avoid double-swapping.
//
// Total invocations: frame_count · frame_len.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_bitrev(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i:       u32 = gid.x;
    let total:   u32 = fft_params.frame_count * fft_params.frame_len;
    if i >= total {
        return;
    }

    let frame_m: u32 = i / fft_params.frame_len;
    let local_i: u32 = i % fft_params.frame_len;

    // Compute log₂(frame_len) — the bit-width for reversal.
    var log2n:     u32 = 0u;
    var tmp:       u32 = fft_params.frame_len >> 1u;
    loop {
        if tmp == 0u { break; }
        log2n   += 1u;
        tmp     >>= 1u;
    }

    // Reverse the log2n low-order bits of local_i.
    var value:    u32 = local_i;
    var reversed: u32 = 0u;
    var bits_rem: u32 = log2n;
    loop {
        if bits_rem == 0u { break; }
        reversed  = (reversed << 1u) | (value & 1u);
        value    >>= 1u;
        bits_rem  -= 1u;
    }

    let local_j: u32 = reversed;
    if local_j > local_i {
        let base:  u32 = frame_m * fft_params.frame_len;
        let i_idx: u32 = base + local_i;
        let j_idx: u32 = base + local_j;
        let re_i:  f32 = re_scratch[i_idx];
        let im_i:  f32 = im_scratch[i_idx];
        re_scratch[i_idx] = re_scratch[j_idx];
        im_scratch[i_idx] = im_scratch[j_idx];
        re_scratch[j_idx] = re_i;
        im_scratch[j_idx] = im_i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3: stft_butterfly
//
// One Radix-2 DIT butterfly stage of the IDFT, batched over all frames.
//
// IDFT twiddle: W_N^{−k} = exp(+2πi · local_idx / group_size) — conjugate of the DFT twiddle.
//
// Thread gid.x covers the joint (frame_m, half-element) space:
//   gid.x ∈ [0, frame_count · frame_len / 2).
//
// stage (from fft_params.stage): 0 = first pass (smallest butterflies), log₂(N)−1 = last pass.
//
// Total invocations per butterfly pass: frame_count · frame_len / 2.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id:     u32 = gid.x;
    let half_n: u32 = fft_params.frame_len >> 1u;
    let total:  u32 = fft_params.frame_count * half_n;
    if id >= total {
        return;
    }

    let frame_m:  u32 = id / half_n;
    let local_id: u32 = id % half_n;

    let h:          u32 = 1u << fft_params.stage;
    let group_size: u32 = h << 1u;
    let group_idx:  u32 = local_id / h;
    let local_idx:  u32 = local_id % h;

    let base: u32 = frame_m * fft_params.frame_len;
    let even: u32 = base + group_idx * group_size + local_idx;
    let odd:  u32 = even + h;

    // IDFT conjugate twiddle: exp(+2πi · local_idx / group_size).
    let angle: f32 = TAU * f32(local_idx) / f32(group_size);
    let w_re:  f32 = cos(angle);
    let w_im:  f32 = sin(angle);

    let e_re: f32 = re_scratch[even];
    let e_im: f32 = im_scratch[even];
    let o_re: f32 = re_scratch[odd];
    let o_im: f32 = im_scratch[odd];

    // wo = w · o
    let wo_re: f32 = w_re * o_re - w_im * o_im;
    let wo_im: f32 = w_re * o_im + w_im * o_re;

    re_scratch[even] = e_re + wo_re;
    im_scratch[even] = e_im + wo_im;
    re_scratch[odd]  = e_re - wo_re;
    im_scratch[odd]  = e_im - wo_im;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 4: stft_scale_and_window
//
// Finalizes the IDFT output by:
//   1. Scaling by 1/frame_len (Cooley-Tukey unnormalised IDFT produces N·x[j]).
//   2. Applying the Hann synthesis window: w[j] = 0.5 − 0.5·cos(TAU·j/(N−1)).
// Writes the result to frame_data for subsequent OLA reconstruction.
//
// gid.x = frame_m · frame_len + j,  j = local sample index in [0, frame_len).
// Total invocations: frame_count · frame_len.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_scale_and_window(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = fft_params.frame_count * fft_params.frame_len;
    if idx >= total {
        return;
    }
    let j: u32 = idx % fft_params.frame_len;
    let w: f32 = hann_window(j, fft_params.frame_len);
    frame_data[idx] = (re_scratch[idx] / f32(fft_params.frame_len)) * w;
}
