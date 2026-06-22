// Short-Time Fourier Transform inverse OLA reconstruction kernel.
//
// `stft_inverse_ola`: per output sample n:
//   y[n] = Σ_m frame_data[m·N + (n − start_m)] / Σ_m w[n − start_m]²
// where start_m = m·hop_len − N/2, terms outside [0, N) are skipped.
//
// frame_data is produced by stft_inverse_fft.wgsl (Closure XI+).
// Formal basis: WOLA identity (Allen–Rabiner 1977, Theorem 1).
//
// Binding layout:
//   @binding(0): read-only storage  — frame_data f32 values
//   @binding(1): read_write storage — output signal f32 values
//   @binding(2): uniform            — StftParams (16 bytes; layout matches Rust StftParams)

// StftParams must be 16 bytes to satisfy WGPU uniform alignment.
// Field order matches the Rust #[repr(C)] StftParams struct byte-for-byte.
struct StftParams {
    signal_len:  u32,   // byte offset 0
    frame_len:   u32,   // byte offset 4
    hop_len:     u32,   // byte offset 8
    frame_count: u32,   // byte offset 12
}

@group(0) @binding(0) var<storage, read>       input_data:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform>             params:      StftParams;

const TAU: f32 = 6.28318530717958647692;

// Hann synthesis window: w[j] = 0.5 − 0.5·cos(TAU·j / (N−1)).
// Returns 1.0 for N ≤ 1 (degenerate single-sample frame).
fn hann(j: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(TAU * f32(j) / f32(frame_len - 1u));
}

// Pass: weighted overlap-add reconstruction.
//
// Each invocation handles one output sample n ∈ [0, signal_len).
// input_data holds frame_data produced by stft_inverse_fft.wgsl.
@compute @workgroup_size(64, 1, 1)
fn stft_inverse_ola(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n: u32 = gid.x;
    if n >= params.signal_len {
        return;
    }

    let fl:   u32 = params.frame_len;
    let half: i32 = i32(fl / 2u);

    var overlap: f32 = 0.0;
    var weight:  f32 = 0.0;

    for (var m: u32 = 0u; m < params.frame_count; m = m + 1u) {
        // start_m = m·hop_len − floor(frame_len / 2)  (signed)
        let start_m: i32 = i32(m) * i32(params.hop_len) - half;
        let local_n: i32 = i32(n) - start_m;
        if local_n >= 0 && u32(local_n) < fl {
            let j: u32 = u32(local_n);
            let w: f32 = hann(j, fl);
            overlap = overlap + input_data[m * fl + j];
            weight  = weight  + w * w;
        }
    }

    // select(false_val, true_val, cond): guard against zero denominator.
    // When weight = 0, select returns 0.0 regardless of overlap / weight.
    output_data[n] = select(0.0, overlap / weight, weight > 0.0);
}
