//! Direct-bins SDFT GPU kernel (forward and inverse).
//!
//! Forward: X[b] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*b*n/N) for b = 0..K.
//! Inverse: x[n] = (1/K) * sum_{b=0}^{K-1} X[b] * exp(+2*pi*i*b*n/N) for n = 0..N.

struct SdftParams {
    window_len: u32,
    bin_count: u32,
    padding0: u32,
    padding1: u32,
}

struct ComplexValue {
    re: f32,
    im: f32,
}

@group(0) @binding(0) var<storage, read> window_data: array<f32>;

@group(0) @binding(1) var<storage, read_write> output_data: array<ComplexValue>;

@group(0) @binding(2) var<uniform> params: SdftParams;

const TAU: f32 = 6.283185307179586476925;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}

@compute @workgroup_size(64, 1, 1)
fn sdft_direct_bins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= params.bin_count { return; }
    let n = params.window_len;
    var acc = vec2<f32>(0.0, 0.0);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let x = vec2<f32>(window_data[i], 0.0);
        let angle = -TAU * f32(b) * f32(i) / f32(n);
        let twiddle = vec2<f32>(cos(angle), sin(angle));
        acc = acc + cmul(x, twiddle);
    }
    output_data[b].re = acc.x;
    output_data[b].im = acc.y;
}

/// Inverse SDFT: reconstruct real signal x[n] from K complex bins.
///
/// x[n] = (1/K) * sum_{b=0}^{K-1} X[b] * exp(+2*pi*i*b*n/K)
///
/// Input: complex bins in `window_data` as interleaved f32 pairs
/// (window_data[2*b] = Re(X[b]), window_data[2*b+1] = Im(X[b])).
/// Output: real signal in `output_data[n].re` (imaginary part is zero).
@compute @workgroup_size(64, 1, 1)
fn sdft_inverse_bins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if n >= params.window_len { return; }
    let k = params.bin_count;
    var acc = vec2<f32>(0.0, 0.0);
    for (var b: u32 = 0u; b < k; b = b + 1u) {
        let x = vec2<f32>(window_data[b * 2u], window_data[b * 2u + 1u]);
        let angle = TAU * f32(b) * f32(n) / f32(k);
        let twiddle = vec2<f32>(cos(angle), sin(angle));
        acc = acc + cmul(x, twiddle);
    }
    let scale = 1.0 / f32(k);
    output_data[n].re = acc.x * scale;
    output_data[n].im = acc.y * scale;
}
