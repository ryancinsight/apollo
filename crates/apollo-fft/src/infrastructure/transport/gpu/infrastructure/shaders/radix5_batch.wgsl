enable f16;

struct Radix5Params {
    batch_count: u32,
    pair_count: u32,
    inverse: u32,
    _pad: u32,
}

@group(0) @binding(0)
var<storage, read> input_re: array<vec2<f16>>;
@group(0) @binding(1)
var<storage, read> input_im: array<vec2<f16>>;
@group(0) @binding(2)
var<storage, read_write> output_re: array<vec2<f16>>;
@group(0) @binding(3)
var<storage, read_write> output_im: array<vec2<f16>>;
@group(1) @binding(0)
var<uniform> params: Radix5Params;

const C1: f16 = 0.30901697h;
const C2: f16 = -0.80901700h;
const S1: f16 = 0.95105654h;
const S2: f16 = 0.58778524h;

fn load_re_pair(point: u32, pair: u32) -> vec2<f16> {
    return input_re[point * params.pair_count + pair];
}

fn load_im_pair(point: u32, pair: u32) -> vec2<f16> {
    return input_im[point * params.pair_count + pair];
}

fn store_re_pair(point: u32, pair: u32, value: vec2<f16>) {
    output_re[point * params.pair_count + pair] = value;
}

fn store_im_pair(point: u32, pair: u32, value: vec2<f16>) {
    output_im[point * params.pair_count + pair] = value;
}

@compute @workgroup_size(128, 1, 1)
fn radix5_batch_native_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair = gid.x;
    if pair >= params.pair_count {
        return;
    }

    let x0r = load_re_pair(0u, pair);
    let x1r = load_re_pair(1u, pair);
    let x2r = load_re_pair(2u, pair);
    let x3r = load_re_pair(3u, pair);
    let x4r = load_re_pair(4u, pair);
    let x0i = load_im_pair(0u, pair);
    let x1i = load_im_pair(1u, pair);
    let x2i = load_im_pair(2u, pair);
    let x3i = load_im_pair(3u, pair);
    let x4i = load_im_pair(4u, pair);

    let a1r = x1r + x4r;
    let a1i = x1i + x4i;
    let a2r = x2r + x3r;
    let a2i = x2i + x3i;
    let b1r = x1r - x4r;
    let b1i = x1i - x4i;
    let b2r = x2r - x3r;
    let b2i = x2i - x3i;

    store_re_pair(0u, pair, x0r + a1r + a2r);
    store_im_pair(0u, pair, x0i + a1i + a2i);

    let p1r = fma(vec2<f16>(C2), a2r, fma(vec2<f16>(C1), a1r, x0r));
    let p1i = fma(vec2<f16>(C2), a2i, fma(vec2<f16>(C1), a1i, x0i));
    let q1r = fma(vec2<f16>(S2), b2r, vec2<f16>(S1) * b1r);
    let q1i = fma(vec2<f16>(S2), b2i, vec2<f16>(S1) * b1i);
    let p2r = fma(vec2<f16>(C1), a2r, fma(vec2<f16>(C2), a1r, x0r));
    let p2i = fma(vec2<f16>(C1), a2i, fma(vec2<f16>(C2), a1i, x0i));
    let q2r = fma(vec2<f16>(-S1), b2r, vec2<f16>(S2) * b1r);
    let q2i = fma(vec2<f16>(-S1), b2i, vec2<f16>(S2) * b1i);

    if params.inverse == 0u {
        store_re_pair(1u, pair, p1r + q1i);
        store_im_pair(1u, pair, p1i - q1r);
        store_re_pair(2u, pair, p2r + q2i);
        store_im_pair(2u, pair, p2i - q2r);
        store_re_pair(3u, pair, p2r - q2i);
        store_im_pair(3u, pair, p2i + q2r);
        store_re_pair(4u, pair, p1r - q1i);
        store_im_pair(4u, pair, p1i + q1r);
    } else {
        store_re_pair(1u, pair, p1r - q1i);
        store_im_pair(1u, pair, p1i + q1r);
        store_re_pair(2u, pair, p2r - q2i);
        store_im_pair(2u, pair, p2i + q2r);
        store_re_pair(3u, pair, p2r + q2i);
        store_im_pair(3u, pair, p2i - q2r);
        store_re_pair(4u, pair, p1r + q1i);
        store_im_pair(4u, pair, p1i - q1r);
    }
}
