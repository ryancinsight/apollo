// STFT analysis framing and split/interleaved conversion.
// Dense Fourier arithmetic between these entry points is owned by Hephaestus.

const TAU: f32 = 6.28318530717958647692;

struct ComplexValue {
    re: f32,
    im: f32,
}

struct StftParams {
    signal_len: u32,
    frame_len: u32,
    hop_len: u32,
    frame_count: u32,
}

@group(0) @binding(0) var<storage, read> signal_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> real_scratch: array<f32>;
@group(0) @binding(2) var<storage, read_write> imaginary_scratch: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_data: array<ComplexValue>;
@group(1) @binding(0) var<uniform> params: StftParams;

fn hann(index: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u {
        return 1.0;
    }
    return 0.5 - 0.5 * cos(TAU * f32(index) / f32(frame_len - 1u));
}

@compute @workgroup_size(256, 1, 1)
fn stft_pack_window(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = params.frame_count * params.frame_len;
    if index >= total {
        return;
    }

    let frame = index / params.frame_len;
    let local = index % params.frame_len;
    let center = i32(frame) * i32(params.hop_len);
    let signal_index = center - i32(params.frame_len / 2u) + i32(local);
    var sample = 0.0;
    if signal_index >= 0 && u32(signal_index) < params.signal_len {
        sample = signal_data[u32(signal_index)];
    }
    real_scratch[index] = hann(local, params.frame_len) * sample;
    imaginary_scratch[index] = 0.0;
}

@compute @workgroup_size(256, 1, 1)
fn stft_interleave(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let total = params.frame_count * params.frame_len;
    if index >= total {
        return;
    }
    output_data[index] = ComplexValue(real_scratch[index], imaginary_scratch[index]);
}
