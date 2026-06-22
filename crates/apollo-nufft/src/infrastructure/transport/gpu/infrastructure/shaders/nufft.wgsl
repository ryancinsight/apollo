struct Position3 {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

struct Complex32 {
    re: f32,
    im: f32,
}

struct NufftParams {
    n0: u32,
    n1: u32,
    n2: u32,
    sample_count: u32,
    l0: f32,
    l1: f32,
    l2: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<storage, read> positions: array<Position3>;

@group(0) @binding(1)
var<storage, read> values: array<Complex32>;

@group(0) @binding(2)
var<storage, read_write> output_values: array<Complex32>;

@group(0) @binding(3)
var<uniform> params: NufftParams;

const TAU: f32 = 6.283185307179586;

fn signed_index(index: u32, len: u32) -> i32 {
    if index <= len / 2u {
        return i32(index);
    }
    return i32(index) - i32(len);
}

fn complex_mul_exp(value: Complex32, angle: f32) -> Complex32 {
    let c = cos(angle);
    let s = sin(angle);
    return Complex32(value.re * c - value.im * s, value.re * s + value.im * c);
}

@compute @workgroup_size(64, 1, 1)
fn nufft_type1_1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.n0 {
        return;
    }

    let k_signed = f32(signed_index(k, params.n0));
    var sum = Complex32(0.0, 0.0);
    for (var sample: u32 = 0u; sample < params.sample_count; sample = sample + 1u) {
        let angle = -TAU * k_signed * positions[sample].x / params.l0;
        let term = complex_mul_exp(values[sample], angle);
        sum = Complex32(sum.re + term.re, sum.im + term.im);
    }
    output_values[k] = sum;
}

@compute @workgroup_size(64, 1, 1)
fn nufft_type2_1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample = gid.x;
    if sample >= params.sample_count {
        return;
    }

    let x = positions[sample].x;
    var sum = Complex32(0.0, 0.0);
    for (var k: u32 = 0u; k < params.n0; k = k + 1u) {
        let k_signed = f32(signed_index(k, params.n0));
        let angle = TAU * k_signed * x / params.l0;
        let term = complex_mul_exp(values[k], angle);
        sum = Complex32(sum.re + term.re, sum.im + term.im);
    }
    output_values[sample] = sum;
}

@compute @workgroup_size(64, 1, 1)
fn nufft_type1_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x;
    let total = params.n0 * params.n1 * params.n2;
    if linear >= total {
        return;
    }

    let kz = linear % params.n2;
    let ky = (linear / params.n2) % params.n1;
    let kx = linear / (params.n1 * params.n2);
    let kx_signed = f32(signed_index(kx, params.n0));
    let ky_signed = f32(signed_index(ky, params.n1));
    let kz_signed = f32(signed_index(kz, params.n2));

    var sum = Complex32(0.0, 0.0);
    for (var sample: u32 = 0u; sample < params.sample_count; sample = sample + 1u) {
        let point = positions[sample];
        let phase = (kx_signed * point.x / params.l0)
            + (ky_signed * point.y / params.l1)
            + (kz_signed * point.z / params.l2);
        let angle = -TAU * phase;
        let term = complex_mul_exp(values[sample], angle);
        sum = Complex32(sum.re + term.re, sum.im + term.im);
    }
    output_values[linear] = sum;
}

@compute @workgroup_size(64, 1, 1)
fn nufft_type2_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample = gid.x;
    if sample >= params.sample_count {
        return;
    }

    let point = positions[sample];
    var sum = Complex32(0.0, 0.0);
    for (var kx: u32 = 0u; kx < params.n0; kx = kx + 1u) {
        let kx_signed = f32(signed_index(kx, params.n0));
        for (var ky: u32 = 0u; ky < params.n1; ky = ky + 1u) {
            let ky_signed = f32(signed_index(ky, params.n1));
            for (var kz: u32 = 0u; kz < params.n2; kz = kz + 1u) {
                let kz_signed = f32(signed_index(kz, params.n2));
                let linear = (kx * params.n1 + ky) * params.n2 + kz;
                let phase = (kx_signed * point.x / params.l0)
                    + (ky_signed * point.y / params.l1)
                    + (kz_signed * point.z / params.l2);
                let angle = TAU * phase;
                let term = complex_mul_exp(values[linear], angle);
                sum = Complex32(sum.re + term.re, sum.im + term.im);
            }
        }
    }
    output_values[sample] = sum;
}
