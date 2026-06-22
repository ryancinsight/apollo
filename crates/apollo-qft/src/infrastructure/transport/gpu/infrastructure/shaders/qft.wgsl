struct Complex32 {
    re: f32,
    im: f32,
}

struct QftParams {
    len: u32,
    mode: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0)
var<storage, read> input_values: array<Complex32>;

@group(0) @binding(1)
var<storage, read_write> output_values: array<Complex32>;

@group(0) @binding(2)
var<uniform> params: QftParams;

fn complex_mul(a: Complex32, b: Complex32) -> Complex32 {
    return Complex32(
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    );
}

@compute @workgroup_size(64, 1, 1)
fn qft_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.len {
        return;
    }

    let sign = select(1.0, -1.0, params.mode == 1u);
    let scale = inverseSqrt(f32(params.len));
    var sum = Complex32(0.0, 0.0);

    for (var col: u32 = 0u; col < params.len; col = col + 1u) {
        let angle = sign * 6.283185307179586 * f32((row * col) % params.len) / f32(params.len);
        let twiddle = Complex32(cos(angle), sin(angle));
        let term = complex_mul(input_values[col], twiddle);
        sum = Complex32(sum.re + term.re, sum.im + term.im);
    }

    output_values[row] = Complex32(sum.re * scale, sum.im * scale);
}
