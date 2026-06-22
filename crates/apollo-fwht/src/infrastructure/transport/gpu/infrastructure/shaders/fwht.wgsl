struct FwhtParams {
    len: u32,
    stride: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

@group(0) @binding(1)
var<uniform> params: FwhtParams;

@compute @workgroup_size(256, 1, 1)
fn fwht_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_id = gid.x;
    let half_len = params.len >> 1u;
    if pair_id >= half_len {
        return;
    }

    let block = params.stride << 1u;
    let group_idx = pair_id / params.stride;
    let lane = pair_id % params.stride;
    let left = group_idx * block + lane;
    let right = left + params.stride;

    let a = data[left];
    let b = data[right];
    data[left] = a + b;
    data[right] = a - b;
}

@compute @workgroup_size(256, 1, 1)
fn fwht_scale_inverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len {
        return;
    }
    data[i] = data[i] / f32(params.len);
}
