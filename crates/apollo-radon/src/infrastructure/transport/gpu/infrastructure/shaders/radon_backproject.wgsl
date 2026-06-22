// Adjoint backprojection kernel.
//
// Formal basis: Radon adjoint = backprojection operator (Natterer 2001, §II.2).
// For each pixel (row, col):
//   bp[r,c] = Σ_{angle_idx} interp(sinogram[angle_idx, ·], det_coord(r,c,θ))
// where
//   det_coord = x·cos(θ) + y·sin(θ),  x = col − (cols−1)/2,  y = row − (rows−1)/2
//   fractional = det_coord / detector_spacing + 0.5·(detector_count − 1)
//   linear interp: s0·(1−w) + s1·w,  out-of-range → 0.0

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

// Binding 0: sinogram input (read-only), row-major: [angle * detector_count + det]
@group(0) @binding(0) var<storage, read> sinogram_input: array<f32>;
// Binding 1: angles (read-only), one f32 per angle in radians
@group(0) @binding(1) var<storage, read> angle_input: array<f32>;
// Binding 2: backprojected image output (read_write), row-major: [row * cols + col]
@group(0) @binding(2) var<storage, read_write> image_output: array<f32>;
// Binding 3: geometry parameters
@group(0) @binding(3) var<uniform> params: RadonParams;

@compute @workgroup_size(64, 1, 1)
fn radon_backproject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx: u32 = gid.x;
    if linear_idx >= params.rows * params.cols { return; }
    let row: u32 = linear_idx / params.cols;
    let col: u32 = linear_idx % params.cols;
    // Pixel center coordinates relative to image center.
    let y: f32 = f32(row) - 0.5 * (f32(params.rows) - 1.0);
    let x: f32 = f32(col) - 0.5 * (f32(params.cols) - 1.0);
    var value: f32 = 0.0;
    for (var ai: u32 = 0u; ai < params.angle_count; ai = ai + 1u) {
        let theta: f32 = angle_input[ai];
        // Project pixel center onto the detector axis.
        let det_coord: f32 = x * cos(theta) + y * sin(theta);
        // Convert physical coordinate to fractional detector index.
        let fractional: f32 = det_coord / params.detector_spacing + 0.5 * (f32(params.detector_count) - 1.0);
        let d0: i32 = i32(floor(fractional));
        let d1: i32 = d0 + 1;
        let w1: f32 = fractional - f32(d0);
        let w0: f32 = 1.0 - w1;
        var s0: f32 = 0.0;
        var s1: f32 = 0.0;
        if d0 >= 0 && u32(d0) < params.detector_count {
            s0 = sinogram_input[ai * params.detector_count + u32(d0)];
        }
        if d1 >= 0 && u32(d1) < params.detector_count {
            s1 = sinogram_input[ai * params.detector_count + u32(d1)];
        }
        value = value + w0 * s0 + w1 * s1;
    }
    image_output[linear_idx] = value;
}
