enable f16;

struct FftParams {
    n: u32,
    stage: u32,
    inverse: u32,
    batch_count: u32,
}

@group(0) @binding(0)
var<storage, read_write> data_re: array<f16>;

@group(0) @binding(1)
var<storage, read_write> data_im: array<f16>;

@group(1) @binding(0)
var<uniform> params: FftParams;

const TWO_PI: f32 = 6.28318530717958647692;

fn cmul_re(a_re: f16, a_im: f16, b_re: f16, b_im: f16) -> f16 {
    return a_re * b_re - a_im * b_im;
}

fn cmul_im(a_re: f16, a_im: f16, b_re: f16, b_im: f16) -> f16 {
    return a_re * b_im + a_im * b_re;
}

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var value = x;
    var reversed: u32 = 0u;
    var remaining = bits;
    loop {
        if remaining == 0u {
            break;
        }
        reversed = (reversed << 1u) | (value & 1u);
        value >>= 1u;
        remaining -= 1u;
    }
    return reversed;
}

@compute @workgroup_size(256, 1, 1)
fn fft_bitrev(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.batch_count;
    if i >= total {
        return;
    }
    let row = i / params.n;
    let local_i = i % params.n;

    var log2n: u32 = 0u;
    var tmp = params.n >> 1u;
    loop {
        if tmp == 0u {
            break;
        }
        log2n += 1u;
        tmp >>= 1u;
    }

    let local_j = bit_reverse(local_i, log2n);
    if local_j > local_i {
        let base = row * params.n;
        let i_idx = base + local_i;
        let j_idx = base + local_j;
        let re_i = data_re[i_idx];
        let im_i = data_im[i_idx];
        data_re[i_idx] = data_re[j_idx];
        data_im[i_idx] = data_im[j_idx];
        data_re[j_idx] = re_i;
        data_im[j_idx] = im_i;
    }
}

@compute @workgroup_size(256, 1, 1)
fn fft_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let half_n = params.n >> 1u;
    let total = half_n * params.batch_count;
    if id >= total {
        return;
    }
    let row = id / half_n;
    let local_id = id % half_n;

    let h = 1u << params.stage;
    let group_size = h << 1u;
    let group_idx = local_id / h;
    let local_idx = local_id % h;
    let base = row * params.n;
    let even = base + group_idx * group_size + local_idx;
    let odd = even + h;

    // Twiddle factor computed in f32 for precision, then narrowed to f16.
    var angle: f32 = -TWO_PI * f32(local_idx) / f32(group_size);
    if params.inverse != 0u {
        angle = -angle;
    }

    let w_re: f16 = f16(cos(angle));
    let w_im: f16 = f16(sin(angle));

    let e_re = data_re[even];
    let e_im = data_im[even];
    let o_re = data_re[odd];
    let o_im = data_im[odd];

    let wo_re = cmul_re(w_re, w_im, o_re, o_im);
    let wo_im = cmul_im(w_re, w_im, o_re, o_im);

    data_re[even] = e_re + wo_re;
    data_im[even] = e_im + wo_im;
    data_re[odd] = e_re - wo_re;
    data_im[odd] = e_im - wo_im;
}

@compute @workgroup_size(256, 1, 1)
fn fft_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.batch_count;
    if i >= total {
        return;
    }
    // Reciprocal computed in f32 for precision, then narrowed to f16.
    let inv_n: f16 = f16(1.0 / f32(params.n));
    data_re[i] = data_re[i] * inv_n;
    data_im[i] = data_im[i] * inv_n;
}
