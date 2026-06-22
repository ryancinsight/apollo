struct Complex32 {
    re: f32,
    im: f32,
}

struct ShtParams {
    output_count: u32,
    reduction_count: u32,
    _padding0: u32,
    _padding1: u32,
}

struct GridSample {
    cos_theta: f32,
    phi: f32,
    weight: f32,
    _padding0: f32,
}

struct BasisParams {
    mode_count: u32,
    sample_count: u32,
    max_degree: u32,
    weighted: u32,
    conjugate: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0)
var<storage, read> input_values: array<Complex32>;

@group(0) @binding(1)
var<storage, read> basis_values: array<Complex32>;

@group(0) @binding(2)
var<storage, read_write> output_values: array<Complex32>;

@group(0) @binding(3)
var<uniform> params: ShtParams;

@group(0) @binding(4)
var<storage, read> grid_values: array<GridSample>;

@group(0) @binding(5)
var<storage, read_write> generated_basis: array<Complex32>;

@group(0) @binding(6)
var<uniform> basis_params: BasisParams;

fn complex_mul(a: Complex32, b: Complex32) -> Complex32 {
    return Complex32(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

fn complex_conj(a: Complex32) -> Complex32 {
    return Complex32(a.re, -a.im);
}

fn mode_degree(mode: u32) -> u32 {
    var degree = 0u;
    loop {
        let next = degree + 1u;
        if (next * next > mode) {
            break;
        }
        degree = next;
    }
    return degree;
}

fn mode_order(mode: u32, degree: u32) -> i32 {
    return i32(mode - degree * degree) - i32(degree);
}

fn associated_legendre(degree: u32, order: u32, x: f32) -> f32 {
    var one_minus_x2 = max(0.0, 1.0 - x * x);
    var p_mm = 1.0;
    if (order > 0u) {
        let sqrt_term = sqrt(one_minus_x2);
        var factor = 1.0;
        for (var k = 1u; k <= order; k = k + 1u) {
            p_mm = p_mm * (-factor * sqrt_term);
            factor = factor + 2.0;
        }
    }
    if (degree == order) {
        return p_mm;
    }

    var p_mmp1 = x * f32(2u * order + 1u) * p_mm;
    if (degree == order + 1u) {
        return p_mmp1;
    }

    var p_lm_minus_two = p_mm;
    var p_lm_minus_one = p_mmp1;
    for (var ell = order + 2u; ell <= degree; ell = ell + 1u) {
        let numerator =
            f32(2u * ell - 1u) * x * p_lm_minus_one - f32(ell + order - 1u) * p_lm_minus_two;
        let p_lm = numerator / f32(ell - order);
        p_lm_minus_two = p_lm_minus_one;
        p_lm_minus_one = p_lm;
    }
    return p_lm_minus_one;
}

fn normalization_constant(degree: u32, order: u32) -> f32 {
    var product = 1.0;
    let numerator = degree - order;
    let denominator = degree + order;
    if (numerator < denominator) {
        for (var value = numerator + 1u; value <= denominator; value = value + 1u) {
            product = product * f32(value);
        }
    }
    let ratio = 1.0 / product;
    return sqrt((f32(2u * degree + 1u) / (4.0 * 3.14159265358979323846)) * ratio);
}

fn spherical_harmonic(degree: u32, order: i32, sample: GridSample) -> Complex32 {
    let abs_order = u32(abs(order));
    let p = associated_legendre(degree, abs_order, sample.cos_theta);
    let norm = normalization_constant(degree, abs_order);
    let angle = f32(abs_order) * sample.phi;
    var y = Complex32(cos(angle) * norm * p, sin(angle) * norm * p);
    if (order < 0) {
        y = complex_conj(y);
        if (abs_order % 2u == 1u) {
            y = Complex32(-y.re, -y.im);
        }
    }
    return y;
}

@compute @workgroup_size(64, 1, 1)
fn sht_basis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let total = basis_params.mode_count * basis_params.sample_count;
    if (index >= total) {
        return;
    }

    var mode = 0u;
    var sample_index = 0u;
    if (basis_params.weighted == 1u) {
        mode = index / basis_params.sample_count;
        sample_index = index - mode * basis_params.sample_count;
    } else {
        sample_index = index / basis_params.mode_count;
        mode = index - sample_index * basis_params.mode_count;
    }

    let degree = mode_degree(mode);
    let order = mode_order(mode, degree);
    let sample = grid_values[sample_index];
    var value = spherical_harmonic(degree, order, sample);
    if (basis_params.conjugate == 1u) {
        value = complex_conj(value);
    }
    if (basis_params.weighted == 1u) {
        value = Complex32(value.re * sample.weight, value.im * sample.weight);
    }
    generated_basis[index] = value;
}

fn matrix_sum(row: u32) -> Complex32 {
    var sum = Complex32(0.0, 0.0);
    for (var col: u32 = 0u; col < params.reduction_count; col = col + 1u) {
        let basis = basis_values[row * params.reduction_count + col];
        let term = complex_mul(input_values[col], basis);
        sum = Complex32(sum.re + term.re, sum.im + term.im);
    }
    return sum;
}

@compute @workgroup_size(64, 1, 1)
fn sht_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.output_count {
        return;
    }
    output_values[row] = matrix_sum(row);
}

@compute @workgroup_size(64, 1, 1)
fn sht_inverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.output_count {
        return;
    }
    output_values[row] = matrix_sum(row);
}
