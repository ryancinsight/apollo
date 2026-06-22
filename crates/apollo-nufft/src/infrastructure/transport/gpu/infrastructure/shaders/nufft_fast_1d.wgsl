// Normalization convention:
// - Type-1 path: forward FFT on oversampled grid, then extract/deconvolve.
//   The forward FFT is unnormalized (no 1/N factor).
// - Type-2 path: load/deconvolve onto grid, then inverse FFT.
//   The inverse FFT applied by GpuFft3d::encode_inverse_split divides by m
//   (the oversampled length) per axis. The CPU path compensates by multiplying
//   by m after the IFFT. On the GPU, the host pre-scales deconv values by m
//   so the normalized IFFT output equals the unnormalized IDFT expected by
//   the KB interpolation kernel. See execute_fast_type2_1d in kernel.rs.

struct Complex32 {
    re: f32,
    im: f32,
}

struct FastNufftParams {
    n: u32,
    m: u32,
    sample_count: u32,
    kernel_width: u32,
    length: f32,
    beta: f32,
    i0_beta: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<storage, read> positions: array<Complex32>;

@group(0) @binding(1)
var<storage, read> values: array<Complex32>;

@group(0) @binding(2)
var<storage, read_write> grid_re: array<f32>;

@group(0) @binding(3)
var<storage, read_write> grid_im: array<f32>;

@group(0) @binding(4)
var<storage, read> deconv: array<Complex32>;

@group(0) @binding(5)
var<storage, read_write> output_values: array<Complex32>;

@group(0) @binding(6)
var<storage, read> coefficients: array<Complex32>;

@group(0) @binding(7)
var<uniform> params: FastNufftParams;

fn signed_index(index: u32, len: u32) -> i32 {
    if index <= len / 2u {
        return i32(index);
    }
    return i32(index) - i32(len);
}

fn i0(value: f32) -> f32 {
    let scaled = 0.25 * value * value;
    var sum = 1.0;
    var term = 1.0;
    for (var k: u32 = 1u; k <= 64u; k = k + 1u) {
        let kf = f32(k);
        term = term * scaled / (kf * kf);
        sum = sum + term;
        if term <= 1.1920929e-7 * sum {
            break;
        }
    }
    return sum;
}

fn kb_kernel(delta: f32) -> f32 {
    let width = f32(params.kernel_width);
    let u2 = (delta / width) * (delta / width);
    if u2 >= 1.0 {
        return 0.0;
    }
    return i0(params.beta * sqrt(1.0 - u2)) / params.i0_beta;
}

fn x_mod(sample: u32) -> f32 {
    let x = positions[sample].re;
    return x - floor(x / params.length) * params.length;
}

fn periodic_delta(index: f32, t: f32) -> f32 {
    let m = f32(params.m);
    let raw = index - t;
    return raw - round(raw / m) * m;
}

fn signed_mode_index(index: u32, len: u32, oversampled_len: u32) -> u32 {
    let signed_value = signed_index(index, len);
    return u32(((signed_value % i32(oversampled_len)) + i32(oversampled_len)) % i32(oversampled_len));
}

@compute @workgroup_size(64, 1, 1)
fn fast_type1_spread_1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid = gid.x;
    if grid >= params.m {
        return;
    }

    var re = 0.0;
    var im = 0.0;
    for (var sample: u32 = 0u; sample < params.sample_count; sample = sample + 1u) {
        let t = f32(params.m) * x_mod(sample) / params.length;
        let weight = kb_kernel(periodic_delta(f32(grid), t));
        if weight != 0.0 {
            re = re + values[sample].re * weight;
            im = im + values[sample].im * weight;
        }
    }
    grid_re[grid] = re;
    grid_im[grid] = im;
}

@compute @workgroup_size(64, 1, 1)
fn fast_type1_extract_1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.n {
        return;
    }
    let m_idx = signed_mode_index(k, params.n, params.m);
    let scale = deconv[k].re;
    output_values[k] = Complex32(grid_re[m_idx] * scale, grid_im[m_idx] * scale);
}

@compute @workgroup_size(64, 1, 1)
fn fast_type2_load_1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid = gid.x;
    if grid >= params.m {
        return;
    }
    grid_re[grid] = 0.0;
    grid_im[grid] = 0.0;

    var is_active = false;
    var k = 0u;
    if grid <= params.n / 2u {
        is_active = true;
        k = grid;
    } else {
        let signed_grid = i32(grid) - i32(params.m);
        let candidate = signed_grid + i32(params.n);
        if candidate > i32(params.n / 2u) && candidate < i32(params.n) {
            is_active = true;
            k = u32(candidate);
        }
    }
    if is_active {
        let scale = deconv[k].re;
        grid_re[grid] = coefficients[k].re * scale;
        grid_im[grid] = coefficients[k].im * scale;
    }
}

@compute @workgroup_size(64, 1, 1)
fn fast_type2_interpolate_1d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample = gid.x;
    if sample >= params.sample_count {
        return;
    }

    let t = f32(params.m) * x_mod(sample) / params.length;
    var re = 0.0;
    var im = 0.0;
    for (var grid: u32 = 0u; grid < params.m; grid = grid + 1u) {
        let weight = kb_kernel(periodic_delta(f32(grid), t));
        if weight != 0.0 {
            re = re + grid_re[grid] * weight;
            im = im + grid_im[grid] * weight;
        }
    }
    output_values[sample] = Complex32(re, im);
}
