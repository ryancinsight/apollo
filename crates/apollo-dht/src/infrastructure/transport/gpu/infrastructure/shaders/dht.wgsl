struct DhtParams {
    len: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

@group(0) @binding(2)
var<uniform> params: DhtParams;

const TAU: f32 = 6.28318530717958647692;

fn hartley_cas(theta: f32) -> f32 {
    return cos(theta) + sin(theta);
}

@compute @workgroup_size(64, 1, 1)
fn dht_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.len {
        return;
    }

    let factor = TAU / f32(params.len);
    var sum = 0.0;
    for (var n: u32 = 0u; n < params.len; n = n + 1u) {
        let angle = factor * f32(k * n);
        sum = sum + input_data[n] * hartley_cas(angle);
    }
    output_data[k] = sum;
}

@compute @workgroup_size(64, 1, 1)
fn dht_scale_inverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len {
        return;
    }
    output_data[i] = output_data[i] / f32(params.len);
}
