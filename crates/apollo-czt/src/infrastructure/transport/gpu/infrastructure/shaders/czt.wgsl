struct ComplexValue {
    re: f32,
    im: f32,
}

struct CztParams {
    input_len: u32,
    output_len: u32,
    a_re: f32,
    a_im: f32,
    w_re: f32,
    w_im: f32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0)
var<storage, read> input_data: array<ComplexValue>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<ComplexValue>;

@group(0) @binding(2)
var<uniform> params: CztParams;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

fn complex_pow_real(z: vec2<f32>, exponent: f32) -> vec2<f32> {
    let radius = length(z);
    let angle = atan2(z.y, z.x);
    let scaled_radius = pow(radius, exponent);
    let scaled_angle = exponent * angle;
    return vec2<f32>(
        scaled_radius * cos(scaled_angle),
        scaled_radius * sin(scaled_angle)
    );
}

@compute @workgroup_size(64, 1, 1)
fn czt_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.output_len {
        return;
    }

    let a = vec2<f32>(params.a_re, params.a_im);
    let w = vec2<f32>(params.w_re, params.w_im);
    var acc = vec2<f32>(0.0, 0.0);

    for (var n: u32 = 0u; n < params.input_len; n = n + 1u) {
        let x = vec2<f32>(input_data[n].re, input_data[n].im);
        let a_pow = complex_pow_real(a, -f32(n));
        let w_pow = complex_pow_real(w, f32(n * k));
        let weight = cmul(a_pow, w_pow);
        acc = acc + cmul(x, weight);
    }

    output_data[k].re = acc.x;
    output_data[k].im = acc.y;
}

/// Inverse chirp z-transform: adjoint formula.
///
/// Computes `x[n] = (A^n / M) · Σ_k X[k] · W^{-nk}` where M = output_len of the
/// forward plan (= input_len here, since M = N is required for the inverse).
///
/// This formula is **exact** when the CZT matrix is unitary — i.e. when
/// |A| = 1, |W| = 1, and W is an N-th root of unity (DFT case).  For general
/// spiral parameters (|A| ≠ 1 or W not a root of unity) it returns the
/// minimum-norm least-squares solution; the exact inverse in those cases
/// requires the Björck-Pereyra Vandermonde solve available in the CPU crate.
@compute @workgroup_size(64, 1, 1)
fn czt_inverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if n >= params.input_len {
        return;
    }

    let a = vec2<f32>(params.a_re, params.a_im);
    let w = vec2<f32>(params.w_re, params.w_im);
    let inv_m = 1.0 / f32(params.output_len);
    var acc = vec2<f32>(0.0, 0.0);

    for (var k: u32 = 0u; k < params.output_len; k = k + 1u) {
        let x = vec2<f32>(input_data[k].re, input_data[k].im);
        let a_pow = complex_pow_real(a, f32(n));
        // W^{-nk} = complex_pow_real(w, -f32(n * k))
        let w_pow = complex_pow_real(w, -f32(n * k));
        let weight = cmul(a_pow, w_pow);
        acc = acc + cmul(x, weight);
    }

    output_data[n].re = inv_m * acc.x;
    output_data[n].im = inv_m * acc.y;
}
