struct FftParams {
    n: u32,
    stage: u32,
    inverse: u32,
    batch_count: u32,
}

struct PackParams {
    nx: u32,
    ny: u32,
    nz: u32,
    axis: u32,
    fft_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read_write> data_re: array<f32>;
@group(0) @binding(1)
var<storage, read_write> data_im: array<f32>;

@group(1) @binding(0)
var<uniform> params: FftParams;

@group(2) @binding(0)
var<storage, read_write> volume_re: array<f32>;
@group(2) @binding(1)
var<storage, read_write> volume_im: array<f32>;
@group(2) @binding(2)
var<uniform> pack_params: PackParams;

fn volume_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return (ix * pack_params.ny + iy) * pack_params.nz + iz;
}

@compute @workgroup_size(256, 1, 1)
fn fft_pack_axis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let axis_len = params.n;
    let total = axis_len * params.batch_count;
    if idx >= total {
        return;
    }

    let row = idx / axis_len;
    let local = idx % axis_len;
    let workspace_idx = row * pack_params.fft_len + local;

    var ix: u32 = 0u;
    var iy: u32 = 0u;
    var iz: u32 = 0u;

    if pack_params.axis == 2u {
        ix = row / pack_params.ny;
        iy = row % pack_params.ny;
        iz = local;
    } else if pack_params.axis == 1u {
        ix = row / pack_params.nz;
        iz = row % pack_params.nz;
        iy = local;
    } else {
        iy = row / pack_params.nz;
        iz = row % pack_params.nz;
        ix = local;
    }

    let src = volume_index(ix, iy, iz);
    data_re[workspace_idx] = volume_re[src];
    data_im[workspace_idx] = volume_im[src];
}

@compute @workgroup_size(256, 1, 1)
fn fft_unpack_axis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let axis_len = params.n;
    let total = axis_len * params.batch_count;
    if idx >= total {
        return;
    }

    let row = idx / axis_len;
    let local = idx % axis_len;
    let workspace_idx = row * pack_params.fft_len + local;

    var ix: u32 = 0u;
    var iy: u32 = 0u;
    var iz: u32 = 0u;

    if pack_params.axis == 2u {
        ix = row / pack_params.ny;
        iy = row % pack_params.ny;
        iz = local;
    } else if pack_params.axis == 1u {
        ix = row / pack_params.nz;
        iz = row % pack_params.nz;
        iy = local;
    } else {
        iy = row / pack_params.nz;
        iz = row % pack_params.nz;
        ix = local;
    }

    let dst = volume_index(ix, iy, iz);
    volume_re[dst] = data_re[workspace_idx];
    volume_im[dst] = data_im[workspace_idx];
}
