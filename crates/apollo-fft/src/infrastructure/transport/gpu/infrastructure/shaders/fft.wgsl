struct FftParams {
    n: u32,
    stage: u32,
    inverse: u32,
    batch_count: u32,
}

@group(0) @binding(0)
var<storage, read_write> data_re: array<f32>;

@group(0) @binding(1)
var<storage, read_write> data_im: array<f32>;

@group(1) @binding(0)
var<uniform> params: FftParams;

const TWO_PI: f32 = 6.28318530717958647692;

fn cmul_re(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> f32 {
    return a_re * b_re - a_im * b_im;
}

fn cmul_im(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> f32 {
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

fn bit_reverse_base4(x: u32, digits: u32) -> u32 {
    var value = x;
    var reversed: u32 = 0u;
    var remaining = digits;
    loop {
        if remaining == 0u {
            break;
        }
        reversed = (reversed << 2u) | (value & 3u);
        value >>= 2u;
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
fn fft_bitrev_radix4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.batch_count;
    if i >= total {
        return;
    }
    let row = i / params.n;
    let local_i = i % params.n;

    var digits: u32 = 0u;
    var tmp = params.n;
    loop {
        if tmp <= 1u {
            break;
        }
        digits += 1u;
        tmp >>= 2u;
    }

    let local_j = bit_reverse_base4(local_i, digits);
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

    var angle = -TWO_PI * f32(local_idx) / f32(group_size);
    if params.inverse != 0u {
        angle = -angle;
    }

    let w_re = cos(angle);
    let w_im = sin(angle);

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
fn fft_forward_radix4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let quarter_n = params.n >> 2u;
    let total = quarter_n * params.batch_count;
    if id >= total {
        return;
    }
    let row = id / quarter_n;
    let local_id = id % quarter_n;

    let m = 1u << (params.stage * 2u);
    let group_size = m << 2u;
    let group_idx = local_id / m;
    let local_idx = local_id % m;
    let base = row * params.n + group_idx * group_size + local_idx;
    let i0 = base;
    let i1 = base + m;
    let i2 = base + (m << 1u);
    let i3 = base + (m * 3u);

    var angle = -TWO_PI * f32(local_idx) / f32(group_size);
    if params.inverse != 0u {
        angle = -angle;
    }

    let w1_re = cos(angle);
    let w1_im = sin(angle);
    let w2_re = cos(2.0 * angle);
    let w2_im = sin(2.0 * angle);
    let w3_re = cos(3.0 * angle);
    let w3_im = sin(3.0 * angle);

    let a0_re = data_re[i0];
    let a0_im = data_im[i0];
    let a1_re = data_re[i1];
    let a1_im = data_im[i1];
    let a2_re = data_re[i2];
    let a2_im = data_im[i2];
    let a3_re = data_re[i3];
    let a3_im = data_im[i3];

    let b1_re = cmul_re(w1_re, w1_im, a1_re, a1_im);
    let b1_im = cmul_im(w1_re, w1_im, a1_re, a1_im);
    let b2_re = cmul_re(w2_re, w2_im, a2_re, a2_im);
    let b2_im = cmul_im(w2_re, w2_im, a2_re, a2_im);
    let b3_re = cmul_re(w3_re, w3_im, a3_re, a3_im);
    let b3_im = cmul_im(w3_re, w3_im, a3_re, a3_im);

    let t0_re = a0_re + b2_re;
    let t0_im = a0_im + b2_im;
    let t1_re = a0_re - b2_re;
    let t1_im = a0_im - b2_im;
    let t2_re = b1_re + b3_re;
    let t2_im = b1_im + b3_im;
    let t3_re = b1_re - b3_re;
    let t3_im = b1_im - b3_im;

    let y0_re = t0_re + t2_re;
    let y0_im = t0_im + t2_im;
    let y2_re = t0_re - t2_re;
    let y2_im = t0_im - t2_im;

    var y1_re: f32;
    var y1_im: f32;
    var y3_re: f32;
    var y3_im: f32;

    if params.inverse != 0u {
        y1_re = t1_re - t3_im;
        y1_im = t1_im + t3_re;
        y3_re = t1_re + t3_im;
        y3_im = t1_im - t3_re;
    } else {
        y1_re = t1_re + t3_im;
        y1_im = t1_im - t3_re;
        y3_re = t1_re - t3_im;
        y3_im = t1_im + t3_re;
    }

    data_re[i0] = y0_re;
    data_im[i0] = y0_im;
    data_re[i1] = y1_re;
    data_im[i1] = y1_im;
    data_re[i2] = y2_re;
    data_im[i2] = y2_im;
    data_re[i3] = y3_re;
    data_im[i3] = y3_im;
}

@compute @workgroup_size(256, 1, 1)
fn fft_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.batch_count;
    if i >= total {
        return;
    }
    let inv_n = 1.0 / f32(params.n);
    data_re[i] *= inv_n;
    data_im[i] *= inv_n;
}
