struct PackParams {
    n: u32,
    stage: u32,
    inverse: u32,
    batch_count: u32,
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

@group(0) @binding(2)
var<storage, read_write> volume_re: array<f32>;
@group(0) @binding(3)
var<storage, read_write> volume_im: array<f32>;
@group(0) @binding(4)
var<uniform> params: PackParams;

fn volume_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return (ix * params.ny + iy) * params.nz + iz;
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
    let workspace_idx = row * params.fft_len + local;

    var ix: u32 = 0u;
    var iy: u32 = 0u;
    var iz: u32 = 0u;

    if params.axis == 2u {
        ix = row / params.ny;
        iy = row % params.ny;
        iz = local;
    } else if params.axis == 1u {
        ix = row / params.nz;
        iz = row % params.nz;
        iy = local;
    } else {
        iy = row / params.nz;
        iz = row % params.nz;
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
    let workspace_idx = row * params.fft_len + local;

    var ix: u32 = 0u;
    var iy: u32 = 0u;
    var iz: u32 = 0u;

    if params.axis == 2u {
        ix = row / params.ny;
        iy = row % params.ny;
        iz = local;
    } else if params.axis == 1u {
        ix = row / params.nz;
        iz = row % params.nz;
        iy = local;
    } else {
        iy = row / params.nz;
        iz = row % params.nz;
        ix = local;
    }

    let dst = volume_index(ix, iy, iz);
    volume_re[dst] = data_re[workspace_idx];
    volume_im[dst] = data_im[workspace_idx];
}
