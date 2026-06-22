// Short-Time Fourier Transform forward kernel — FFT-accelerated (O(N log N) per frame).
//
// Replaces the O(N²) per-frame direct DFT with a batched Cooley-Tukey Radix-2 DIT FFT,
// reducing per-frame complexity from O(N²) to O(N log N).
//
// Dispatch sequence (all passes encoded in one CommandEncoder;
// implicit per-pass memory barriers guarantee write visibility between passes):
//   Pass 1 (stft_fwd_pack_window): apply Hann analysis window to centred overlapping frames,
//                                   pack real windowed samples into split re/im scratch.
//   Pass 2 (stft_fwd_bitrev):      Cooley-Tukey bit-reversal permutation (in-place, batched).
//   Pass 3 (stft_fwd_butterfly):   one Radix-2 DIT butterfly stage per dispatch;
//                                   dispatched log₂(N) times with distinct params bind groups.
//   Pass 4 (stft_fwd_interleave):  pack split re/im scratch → interleaved ComplexValue output.
//
// Bind group layout:
//   group 0 — data buffers (shared by all four passes):
//     binding 0: read-only storage  — signal_data   (f32 samples)         [signal_len]
//     binding 1: read_write storage — re_scratch     (f32)                 [frame_count · frame_len]
//     binding 2: read_write storage — im_scratch     (f32)                 [frame_count · frame_len]
//     binding 3: read_write storage — output_data    (ComplexValue pairs)  [frame_count · frame_len]
//   group 1 — per-stage parameters (one distinct bind group per butterfly dispatch):
//     binding 0: uniform — FwdFftParams (16 bytes; 4 × u32)
//
// Formal bases:
//   Cooley & Tukey (1965) "An algorithm for the machine calculation of complex Fourier series".
//   DFT definition: X[k] = Σ_{n=0}^{N-1} x[n] · exp(−2πi · k · n / N).
//   DFT twiddle: W_N^k = exp(−2πi · k / N)  — conjugate of IDFT twiddle in stft_inverse_fft.wgsl.
//   Hann analysis window: w[n] = 0.5 − 0.5 · cos(2π · n / (N − 1)).

const TAU: f32 = 6.28318530717958647692;

struct ComplexValue {
    re: f32,
    im: f32,
}

// Per-stage parameters for the forward FFT passes.
// hop_len is consumed by stft_fwd_pack_window only; stage is consumed by stft_fwd_butterfly only.
// Sized to 16 bytes (4 × u32) to satisfy WGPU uniform alignment.
// The BGL min_binding_size is 16 bytes — identical to the inverse FftStageParams layout.
struct FwdFftParams {
    frame_count: u32,   // number of STFT frames (batch dimension)
    frame_len:   u32,   // FFT length per frame; must be a power of two (Radix-2 requirement)
    hop_len:     u32,   // analysis hop size in samples (frame centre spacing)
    stage:       u32,   // butterfly stage index: 0 = least-significant half, log₂(N)−1 = MSB half
}

// Group 0: data buffers, shared by all four forward FFT passes.
@group(0) @binding(0) var<storage, read>       signal_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> re_scratch:  array<f32>;
@group(0) @binding(2) var<storage, read_write> im_scratch:  array<f32>;
@group(0) @binding(3) var<storage, read_write> output_data: array<ComplexValue>;

// Group 1: per-stage parameters.
@group(1) @binding(0) var<uniform> fwd_params: FwdFftParams;

// Hann analysis window: w[n] = 0.5 − 0.5 · cos(TAU · n / (frame_len − 1)).
// Returns 1.0 for the degenerate single-sample case.
fn hann_analysis(n: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(TAU * f32(n) / f32(frame_len - 1u));
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1: stft_fwd_pack_window
//
// For each (frame_m, n) pair (gid.x = frame_m · frame_len + n):
//   center   = frame_m · hop_len
//   sig_idx  = center − frame_len/2 + n      (centred analysis window, same as stft.wgsl)
//   re_scratch[gid.x] = hann_analysis(n, frame_len) · signal_data[sig_idx]   (zero if out-of-range)
//   im_scratch[gid.x] = 0.0
//
// Total invocations: frame_count · frame_len.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_fwd_pack_window(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = fwd_params.frame_count * fwd_params.frame_len;
    if idx >= total {
        return;
    }

    let frame_m: u32 = idx / fwd_params.frame_len;
    let n:       u32 = idx % fwd_params.frame_len;
    let fl:      u32 = fwd_params.frame_len;

    let center:  i32 = i32(frame_m) * i32(fwd_params.hop_len);
    let half:    i32 = i32(fl) / 2;
    let sig_idx: i32 = center - half + i32(n);

    var sample: f32 = 0.0;
    if sig_idx >= 0 && u32(sig_idx) < arrayLength(&signal_data) {
        sample = signal_data[u32(sig_idx)];
    }

    re_scratch[idx] = hann_analysis(n, fl) * sample;
    im_scratch[idx] = 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: stft_fwd_bitrev
//
// Applies the Cooley-Tukey bit-reversal permutation in-place on split re/im scratch,
// independently per frame (batched).
//
// Only pairs with reversed > original are swapped to avoid double-swapping.
// Total invocations: frame_count · frame_len.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_fwd_bitrev(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i:     u32 = gid.x;
    let total: u32 = fwd_params.frame_count * fwd_params.frame_len;
    if i >= total {
        return;
    }

    let frame_m: u32 = i / fwd_params.frame_len;
    let local_i: u32 = i % fwd_params.frame_len;

    // Compute log₂(frame_len).
    var log2n: u32 = 0u;
    var tmp:   u32 = fwd_params.frame_len >> 1u;
    loop {
        if tmp == 0u { break; }
        log2n += 1u;
        tmp   >>= 1u;
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
        let base:  u32 = frame_m * fwd_params.frame_len;
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
// Pass 3: stft_fwd_butterfly
//
// One Radix-2 DIT butterfly stage of the DFT, batched over all frames.
//
// DFT twiddle: W_N^{local_idx · N/group_size} = exp(−2πi · local_idx / group_size)
//   — this is the NEGATED angle relative to the IDFT twiddle in stft_inverse_fft.wgsl.
//
// Thread gid.x ∈ [0, frame_count · frame_len / 2):
//   frame_m   = gid.x / (frame_len / 2)
//   local_id  = gid.x % (frame_len / 2)
//
// stage (from fwd_params.stage): 0 = first pass (size-2 butterflies), log₂(N)−1 = last pass.
// Total invocations per butterfly pass: frame_count · frame_len / 2.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_fwd_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id:     u32 = gid.x;
    let half_n: u32 = fwd_params.frame_len >> 1u;
    let total:  u32 = fwd_params.frame_count * half_n;
    if id >= total {
        return;
    }

    let frame_m:  u32 = id / half_n;
    let local_id: u32 = id % half_n;

    let h:          u32 = 1u << fwd_params.stage;
    let group_size: u32 = h << 1u;
    let group_idx:  u32 = local_id / h;
    let local_idx:  u32 = local_id % h;

    let base: u32 = frame_m * fwd_params.frame_len;
    let even: u32 = base + group_idx * group_size + local_idx;
    let odd:  u32 = even + h;

    // DFT twiddle: exp(−2πi · local_idx / group_size)  [negative angle = DFT, not IDFT].
    let angle: f32 = -TAU * f32(local_idx) / f32(group_size);
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
// Pass 4: stft_fwd_interleave
//
// Packs the split re/im scratch into the interleaved ComplexValue output array.
//
// Output layout: output_data[m·N + k].re = Re{X[m, k]}
//                output_data[m·N + k].im = Im{X[m, k]}
//
// Total invocations: frame_count · frame_len.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn stft_fwd_interleave(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = fwd_params.frame_count * fwd_params.frame_len;
    if idx >= total {
        return;
    }
    output_data[idx] = ComplexValue(re_scratch[idx], im_scratch[idx]);
}
