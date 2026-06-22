enable f16;

struct FftParams {
    n: u32,
    stage: u32,
    inverse: u32,
    batch_count: u32,
}

@group(0) @binding(0)
var<storage, read> src_re: array<f16>;

@group(0) @binding(1)
var<storage, read> src_im: array<f16>;

@group(0) @binding(2)
var<storage, read_write> dst_re: array<f16>;

@group(0) @binding(3)
var<storage, read_write> dst_im: array<f16>;

@group(1) @binding(0)
var<uniform> params: FftParams;

const TWO_PI: f32 = 6.28318530717958647692;

fn cmul_re(a_re: f16, a_im: f16, b_re: f16, b_im: f16) -> f16 {
    return a_re * b_re - a_im * b_im;
}

fn cmul_im(a_re: f16, a_im: f16, b_re: f16, b_im: f16) -> f16 {
    return a_re * b_im + a_im * b_re;
}

@compute @workgroup_size(256, 1, 1)
fn fft_stockham(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let half_n = params.n >> 1u;
    let total = half_n * params.batch_count;
    if id >= total {
        return;
    }

    let row = id / half_n;
    let local_id = id % half_n;
    let stride = 1u << params.stage;
    let groups = params.n / (stride << 1u);
    let j = local_id / groups;
    let k = local_id - j * groups;
    let base = row * params.n;
    let src_base = base + (j << 1u) * groups + k;
    let dst_base = base + j * groups + k;

    var angle = -TWO_PI * f32(j) / f32(stride << 1u);
    if params.inverse != 0u {
        angle = -angle;
    }
    let w_re = f16(cos(angle));
    let w_im = f16(sin(angle));

    let a_re = src_re[src_base];
    let a_im = src_im[src_base];
    let b_idx = src_base + groups;
    let b_re = src_re[b_idx];
    let b_im = src_im[b_idx];
    let wb_re = cmul_re(w_re, w_im, b_re, b_im);
    let wb_im = cmul_im(w_re, w_im, b_re, b_im);

    dst_re[dst_base] = a_re + wb_re;
    dst_im[dst_base] = a_im + wb_im;
    let hi = dst_base + half_n;
    dst_re[hi] = a_re - wb_re;
    dst_im[hi] = a_im - wb_im;
}

@compute @workgroup_size(256, 1, 1)
fn fft_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.batch_count;
    if i >= total {
        return;
    }
    dst_re[i] = src_re[i];
    dst_im[i] = src_im[i];
}

@compute @workgroup_size(256, 1, 1)
fn fft_scale_to_dst(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.batch_count;
    if i >= total {
        return;
    }
    let inv_n = f16(1.0 / f32(params.n));
    dst_re[i] = src_re[i] * inv_n;
    dst_im[i] = src_im[i] * inv_n;
}
