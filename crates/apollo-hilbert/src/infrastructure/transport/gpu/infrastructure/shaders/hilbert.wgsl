struct ComplexValue {
    re: f32,
    im: f32,
}

struct HilbertParams {
    len: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0)
var<storage, read> input_data: array<ComplexValue>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<ComplexValue>;

@group(0) @binding(2)
var<uniform> params: HilbertParams;

const TAU: f32 = 6.28318530717958647692;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

@compute @workgroup_size(64, 1, 1)
fn hilbert_forward_dft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.len {
        return;
    }

    let factor = -TAU / f32(params.len);
    var acc = vec2<f32>(0.0, 0.0);
    for (var n: u32 = 0u; n < params.len; n = n + 1u) {
        let angle = factor * f32(k * n);
        let twiddle = vec2<f32>(cos(angle), sin(angle));
        let sample = vec2<f32>(input_data[n].re, 0.0);
        acc = acc + cmul(sample, twiddle);
    }
    output_data[k].re = acc.x;
    output_data[k].im = acc.y;
}

@compute @workgroup_size(64, 1, 1)
fn hilbert_apply_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.len {
        return;
    }

    let positive_end = (params.len + 1u) / 2u;
    var scale = 0.0;
    if k == 0u || ((params.len & 1u) == 0u && k == params.len / 2u) {
        scale = 1.0;
    } else if k < positive_end {
        scale = 2.0;
    }
    output_data[k].re = output_data[k].re * scale;
    output_data[k].im = output_data[k].im * scale;
}

@compute @workgroup_size(64, 1, 1)
fn hilbert_inverse_dft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if n >= params.len {
        return;
    }
    let factor = TAU / f32(params.len);
    let scale = 1.0 / f32(params.len);
    var acc = vec2<f32>(0.0, 0.0);
    for (var k: u32 = 0u; k < params.len; k = k + 1u) {
        let angle = factor * f32(k * n);
        let twiddle = vec2<f32>(cos(angle), sin(angle));
        let coefficient = vec2<f32>(input_data[k].re, input_data[k].im);
        acc = acc + cmul(coefficient, twiddle);
    }
    output_data[n].re = acc.x * scale;
    output_data[n].im = acc.y * scale;
}

/// Inverse Hilbert mask: recover the original DFT spectrum X[k] from the
/// DFT of the quadrature component Q[k].
///
/// The Hilbert transform filter in the frequency domain is
/// H[k] = -j * sgn(k), so Q[k] = -j * sgn(k) * X[k].
/// Therefore X[k] = Q[k] / H[k] = Q[k] * j / sgn(k).
///
/// For positive frequencies: X[k] = j * Q[k] = (-Q[k].im, Q[k].re)
/// For negative frequencies: X[k] = -j * Q[k] = (Q[k].im, -Q[k].re)
/// DC and Nyquist: Q[k] = 0 (Hilbert of constant is zero), X[k] is
/// unrecoverable from the quadrature alone. We set X[0] and X[N/2] to zero;
/// the recovered signal will have zero mean (the DC offset is lost).
///
/// Reads from input_data (the DFT of the quadrature input) and writes to
/// output_data (the recovered original spectrum).
@compute @workgroup_size(64, 1, 1)
fn hilbert_inverse_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.len {
        return;
    }
    let N = params.len;
    let positive_end = (N + 1u) / 2u;

    if k == 0u {
        // DC: Hilbert of constant is zero. X[0] is unrecoverable.
        output_data[k].re = 0.0;
        output_data[k].im = 0.0;
    } else if (N & 1u) == 0u && k == N / 2u {
        // Nyquist (even N): same as DC, lost in Hilbert transform.
        output_data[k].re = 0.0;
        output_data[k].im = 0.0;
    } else if k < positive_end {
        // Positive frequency: Q[k] = -j * X[k], so X[k] = j * Q[k] = (-Q.im, Q.re).
        output_data[k].re = -input_data[k].im;
        output_data[k].im = input_data[k].re;
    } else {
        // Negative frequency: Q[k] = j * X[k], so X[k] = -j * Q[k] = (Q.im, -Q.re).
        output_data[k].re = input_data[k].im;
        output_data[k].im = -input_data[k].re;
    }
}
