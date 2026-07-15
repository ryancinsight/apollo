struct Complex32 {
    re: f32,
    im: f32,
}

struct ShtParams {
    output_count: u32,
    reduction_count: u32,
    padding0: u32,
    padding1: u32,
}

struct GridSample {
    cos_theta: f32,
    phi: f32,
    weight: f32,
    padding0: f32,
}

struct BasisParams {
    mode_count: u32,
    sample_count: u32,
    max_degree: u32,
    weighted: u32,
    conjugate: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}

fn complex_mul(a: Complex32, b: Complex32) -> Complex32 {
    return Complex32(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

fn complex_conj(value: Complex32) -> Complex32 {
    return Complex32(value.re, -value.im);
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
    let one_minus_x2 = max(0.0, 1.0 - x * x);
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
    var value = Complex32(cos(angle) * norm * p, sin(angle) * norm * p);
    if (order < 0) {
        value = complex_conj(value);
        if (abs_order % 2u == 1u) {
            value = Complex32(-value.re, -value.im);
        }
    }
    return value;
}
