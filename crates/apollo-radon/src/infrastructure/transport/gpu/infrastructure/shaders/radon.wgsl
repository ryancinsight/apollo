struct RadonParams {
    rows: u32,
    cols: u32,
    angle_count: u32,
    detector_count: u32,
    detector_spacing: f32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0)
var<storage, read> image_values: array<f32>;

@group(0) @binding(1)
var<storage, read> angle_values: array<f32>;

@group(0) @binding(2)
var<storage, read_write> sinogram_values: array<f32>;

@group(0) @binding(3)
var<uniform> params: RadonParams;

@compute @workgroup_size(64, 1, 1)
fn radon_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx = gid.x;
    let total = params.angle_count * params.detector_count;
    if linear_idx >= total {
        return;
    }

    let angle_index = linear_idx / params.detector_count;
    let detector_index = linear_idx % params.detector_count;
    let theta = angle_values[angle_index];
    let sin_theta = sin(theta);
    let cos_theta = cos(theta);
    let detector_center = (f32(detector_index) - 0.5 * (f32(params.detector_count) - 1.0)) * params.detector_spacing;

    var projection = 0.0;
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        let y = f32(r) - 0.5 * (f32(params.rows) - 1.0);
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            let x = f32(c) - 0.5 * (f32(params.cols) - 1.0);
            let det_coord = x * cos_theta + y * sin_theta;
            let fractional = det_coord / params.detector_spacing + 0.5 * (f32(params.detector_count) - 1.0);
            let left = floor(fractional);
            if left < 0.0 || left > f32(params.detector_count - 1u) {
                continue;
            }
            let right_weight = fractional - left;
            let left_weight = 1.0 - right_weight;
            let pixel = image_values[r * params.cols + c];
            if u32(left) == detector_index {
                projection = projection + pixel * left_weight;
            }
            if right_weight > 0.0 && u32(left) + 1u == detector_index && detector_index < params.detector_count {
                projection = projection + pixel * right_weight;
            }
        }
    }
    sinogram_values[linear_idx] = projection;
}
