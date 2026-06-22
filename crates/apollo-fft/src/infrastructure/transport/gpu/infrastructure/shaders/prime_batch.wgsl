enable f16;

struct PrimeParams {
    batch_count: u32,
    pair_count: u32,
    radix: u32,
    inverse: u32,
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
var<uniform> params: PrimeParams;

const TAU: f16 = 6.283185h;

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
fn prime_batch_native_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair = gid.x;
    if pair >= params.pair_count {
        return;
    }

    let sign = select(1.0h, -1.0h, params.inverse != 0u);
    for (var k = 0u; k < params.radix; k = k + 1u) {
        var yr = vec2<f16>(0.0h);
        var yi = vec2<f16>(0.0h);
        for (var j = 0u; j < params.radix; j = j + 1u) {
            let angle = sign * TAU * f16(j * k) / f16(params.radix);
            let c = vec2<f16>(cos(angle));
            let s = vec2<f16>(sin(angle));
            let xr = load_re_pair(j, pair);
            let xi = load_im_pair(j, pair);
            yr = fma(xi, s, fma(xr, c, yr));
            yi = fma(xi, c, fma(-xr, s, yi));
        }
        store_re_pair(k, pair, yr);
        store_im_pair(k, pair, yi);
    }
}
