const TAU: f32 = 6.28318530717958647692;

struct FftStageParams {
    frame_count: u32,
    frame_len: u32,
    stage: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> src_re: array<f32>;
@group(0) @binding(1) var<storage, read> src_im: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_re: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_im: array<f32>;

@group(1) @binding(0) var<uniform> fft_params: FftStageParams;

fn cmul_re(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> f32 {
    return a_re * b_re - a_im * b_im;
}

fn cmul_im(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> f32 {
    return a_re * b_im + a_im * b_re;
}

fn stockham_stage(id: u32, sign: f32) {
    let half_n = fft_params.frame_len >> 1u;
    let total = fft_params.frame_count * half_n;
    if id >= total {
        return;
    }

    let frame_m = id / half_n;
    let local_id = id % half_n;
    let half = 1u << fft_params.stage;
    let groups = fft_params.frame_len / (half << 1u);
    let j = local_id / groups;
    let g = local_id - j * groups;
    let base = frame_m * fft_params.frame_len;
    let src_low = base + (j << 1u) * groups + g;
    let src_high = src_low + groups;
    let dst_low = base + j * groups + g;
    let dst_high = dst_low + half_n;

    let angle = sign * TAU * f32(j) / f32(half << 1u);
    let w_re = cos(angle);
    let w_im = sin(angle);
    let a_re = src_re[src_low];
    let a_im = src_im[src_low];
    let b_re = src_re[src_high];
    let b_im = src_im[src_high];
    let wb_re = cmul_re(w_re, w_im, b_re, b_im);
    let wb_im = cmul_im(w_re, w_im, b_re, b_im);

    dst_re[dst_low] = a_re + wb_re;
    dst_im[dst_low] = a_im + wb_im;
    dst_re[dst_high] = a_re - wb_re;
    dst_im[dst_high] = a_im - wb_im;
}

@compute @workgroup_size(256, 1, 1)
fn stft_stockham_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    stockham_stage(gid.x, -1.0);
}

@compute @workgroup_size(256, 1, 1)
fn stft_stockham_inverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    stockham_stage(gid.x, 1.0);
}

@compute @workgroup_size(256, 1, 1)
fn stft_stockham_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = fft_params.frame_count * fft_params.frame_len;
    if idx >= total {
        return;
    }
    dst_re[idx] = src_re[idx];
    dst_im[idx] = src_im[idx];
}
