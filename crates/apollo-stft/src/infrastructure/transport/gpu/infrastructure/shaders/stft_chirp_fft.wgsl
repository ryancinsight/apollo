// Radix-2 DIT FFT/IFFT sub-shader for the STFT Bluestein/Chirp-Z path.
//
// Operates on the chirp working buffers (group 0 = chirp_data_bgl):
//   binding 0: read_write — chirp_re  f32[frame_count · M]
//   binding 1: read_write — chirp_im  f32[frame_count · M]
//   binding 2: read       — h_fft_re  (unused by this shader; present to match BGL)
//   binding 3: read       — h_fft_im  (unused by this shader; present to match BGL)
//
// Group 1: per-stage parameters.
//   [0] fft_len:     M (padded Radix-2 length)
//   [1] stage:       butterfly stage index (0 = LSB half, log₂M−1 = MSB half)
//   [2] inverse_flag: 0 = forward, 1 = inverse
//   [3] batch_count: frame_count
//
// Formal basis: Cooley & Tukey (1965).

const TAU: f32 = 6.28318530717958647692;

struct ChirpFftParams {
    fft_len:      u32,   // M
    stage:        u32,   // butterfly stage index
    inverse_flag: u32,   // 0 = forward, 1 = inverse
    batch_count:  u32,   // frame_count
}

@group(0) @binding(0) var<storage, read_write> chirp_re: array<f32>;
@group(0) @binding(1) var<storage, read_write> chirp_im: array<f32>;
@group(0) @binding(2) var<storage, read>       _h_fft_re: array<f32>;  // unused
@group(0) @binding(3) var<storage, read>       _h_fft_im: array<f32>;  // unused

@group(1) @binding(0) var<uniform> p: ChirpFftParams;

// ─────────────────────────────────────────────────────────────────────────────
// chirp_fft_bitrev — in-place bit-reversal permutation (batched over frame_count).
//
// For each (batch_m, j) with j in [0, M):
//   rev = bit_reverse(j, log₂M)
//   if j < rev: swap chirp[batch_m·M + j] ↔ chirp[batch_m·M + rev]
//
// Total invocations: batch_count · M.
// ─────────────────────────────────────────────────────────────────────────────
fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v: u32 = x;
    var r: u32 = 0u;
    for (var i: u32 = 0u; i < bits; i = i + 1u) {
        r = (r << 1u) | (v & 1u);
        v >>= 1u;
    }
    return r;
}

@compute @workgroup_size(256, 1, 1)
fn chirp_fft_bitrev(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = p.batch_count * p.fft_len;
    if idx >= total {
        return;
    }
    let batch_m: u32 = idx / p.fft_len;
    let j:       u32 = idx % p.fft_len;
    let bits:    u32 = countTrailingZeros(p.fft_len);
    let rev:     u32 = bit_reverse(j, bits);
    if j < rev {
        let base: u32 = batch_m * p.fft_len;
        let a_re: f32 = chirp_re[base + j];
        let a_im: f32 = chirp_im[base + j];
        chirp_re[base + j]   = chirp_re[base + rev];
        chirp_im[base + j]   = chirp_im[base + rev];
        chirp_re[base + rev] = a_re;
        chirp_im[base + rev] = a_im;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// chirp_fft_butterfly_fwd — one Radix-2 DIT butterfly stage (forward, DFT twiddle).
//
// DFT twiddle: W_M^k = exp(−2πi·k/M).
// Total invocations: batch_count · M/2.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn chirp_fft_butterfly_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let half:  u32 = p.batch_count * (p.fft_len / 2u);
    if idx >= half {
        return;
    }
    let batch_m: u32 = idx / (p.fft_len / 2u);
    let i:       u32 = idx % (p.fft_len / 2u);
    let m:       u32 = 1u << (p.stage + 1u);
    let j:       u32 = i % (m / 2u);
    let k:       u32 = (i / (m / 2u)) * m + j;
    let base:    u32 = batch_m * p.fft_len;

    let arg:     f32 = -TAU * f32(j) / f32(m);
    let tw_re:   f32 = cos(arg);
    let tw_im:   f32 = sin(arg);

    let a_re: f32 = chirp_re[base + k];
    let a_im: f32 = chirp_im[base + k];
    let b_re: f32 = chirp_re[base + k + m / 2u];
    let b_im: f32 = chirp_im[base + k + m / 2u];

    let t_re: f32 = tw_re * b_re - tw_im * b_im;
    let t_im: f32 = tw_re * b_im + tw_im * b_re;

    chirp_re[base + k]           = a_re + t_re;
    chirp_im[base + k]           = a_im + t_im;
    chirp_re[base + k + m / 2u]  = a_re - t_re;
    chirp_im[base + k + m / 2u]  = a_im - t_im;
}

// ─────────────────────────────────────────────────────────────────────────────
// chirp_fft_butterfly_inv — one Radix-2 DIT butterfly stage (inverse, IDFT twiddle).
//
// IDFT twiddle: W_M^{-k} = exp(+2πi·k/M).
// Total invocations: batch_count · M/2.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn chirp_fft_butterfly_inv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let half:  u32 = p.batch_count * (p.fft_len / 2u);
    if idx >= half {
        return;
    }
    let batch_m: u32 = idx / (p.fft_len / 2u);
    let i:       u32 = idx % (p.fft_len / 2u);
    let m:       u32 = 1u << (p.stage + 1u);
    let j:       u32 = i % (m / 2u);
    let k:       u32 = (i / (m / 2u)) * m + j;
    let base:    u32 = batch_m * p.fft_len;

    // Conjugate twiddle: exp(+2πi·k/M).
    let arg:     f32 = TAU * f32(j) / f32(m);
    let tw_re:   f32 = cos(arg);
    let tw_im:   f32 = sin(arg);

    let a_re: f32 = chirp_re[base + k];
    let a_im: f32 = chirp_im[base + k];
    let b_re: f32 = chirp_re[base + k + m / 2u];
    let b_im: f32 = chirp_im[base + k + m / 2u];

    let t_re: f32 = tw_re * b_re - tw_im * b_im;
    let t_im: f32 = tw_re * b_im + tw_im * b_re;

    chirp_re[base + k]           = a_re + t_re;
    chirp_im[base + k]           = a_im + t_im;
    chirp_re[base + k + m / 2u]  = a_re - t_re;
    chirp_im[base + k + m / 2u]  = a_im - t_im;
}

// ─────────────────────────────────────────────────────────────────────────────
// chirp_fft_scale — apply 1/M normalisation after the inverse sub-FFT.
//
// Total invocations: batch_count · M.
// ─────────────────────────────────────────────────────────────────────────────
@compute @workgroup_size(256, 1, 1)
fn chirp_fft_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = p.batch_count * p.fft_len;
    if idx >= total {
        return;
    }
    let scale: f32 = 1.0 / f32(p.fft_len);
    chirp_re[idx] *= scale;
    chirp_im[idx] *= scale;
}
