// STFT inverse split/interleaved conversion and synthesis windowing.
// Hephaestus performs the normalized inverse FFT between these entry points.

const TAU: f32 = 6.28318530717958647692;

struct StftParams {
    signal_len: u32,
    frame_len: u32,
    hop_len: u32,
    frame_count: u32,
}

@group(0) @binding(0) var<storage, read> spectrum: array<f32>;
@group(0) @binding(1) var<storage, read_write> real_scratch: array<f32>;
@group(0) @binding(2) var<storage, read_write> imaginary_scratch: array<f32>;
@group(0) @binding(3) var<storage, read_write> frame_data: array<f32>;
@group(1) @binding(0) var<uniform> params: StftParams;

fn hann(index: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(TAU * f32(index) / f32(frame_len - 1u));
}

@compute @workgroup_size(256, 1, 1)
fn stft_deinterleave(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = params.frame_count * params.frame_len;
    if index >= total {
        return;
    }
    real_scratch[index] = spectrum[2u * index];
    imaginary_scratch[index] = spectrum[2u * index + 1u];
}

@compute @workgroup_size(256, 1, 1)
fn stft_synthesis_window(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = params.frame_count * params.frame_len;
    if index >= total {
        return;
    }
    let local = index % params.frame_len;
    frame_data[index] = real_scratch[index] * hann(local, params.frame_len);
}
