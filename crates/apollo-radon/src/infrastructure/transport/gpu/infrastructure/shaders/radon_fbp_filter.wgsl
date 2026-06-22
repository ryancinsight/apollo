// Ramp-filter pass for filtered backprojection (Ram-Lak filter).
//
// Formal basis: Ram-Lak ramp filter for CT filtered backprojection.
// References:
//   Bracewell & Riddle (1967), "Inversion of fan-beam scans in radio astronomy."
//   Shepp & Logan (1974), "The Fourier reconstruction of a head section."
//
// Algorithm: circular convolution of each projection row with the ramp impulse response h.
//   filtered[angle * D + d] = sum_{d'=0}^{D-1} sinogram[angle * D + d'] * h[(d - d' + D) % D]
//
// Filter kernel h = IFFT(R) is computed on the host (f64 Apollo FFT, then cast to f32):
//   R[k] = 2pi * |signed_k| / (N * detector_spacing),  signed_k = k if k<=N/2 else k-N.
//
// Reuses the same 4-binding layout as forward and backproject passes:
//   binding 0: sinogram input (read)
//   binding 1: filter kernel h (read; replaces the angles buffer for this pass)
//   binding 2: filtered sinogram output (read_write)
//   binding 3: RadonParams uniform (rows, cols, angle_count, detector_count, detector_spacing)
//
// Each GPU thread handles one (angle_idx, det) output element.
// No data races: each thread writes a unique output index.

struct RadonParams {
    rows: u32,
    cols: u32,
    angle_count: u32,
    detector_count: u32,
    detector_spacing: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read>       sinogram_input:  array<f32>;
@group(0) @binding(1) var<storage, read>       filter_kernel:   array<f32>;
@group(0) @binding(2) var<storage, read_write> filtered_output: array<f32>;
@group(0) @binding(3) var<uniform>             params:          RadonParams;

@compute @workgroup_size(64, 1, 1)
fn radon_fbp_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx: u32 = gid.x;
    let total:      u32 = params.angle_count * params.detector_count;
    if linear_idx >= total { return; }

    let angle_idx: u32 = linear_idx / params.detector_count;
    let det:       u32 = linear_idx % params.detector_count;
    let D:         u32 = params.detector_count;

    var acc: f32 = 0.0;
    for (var dp: u32 = 0u; dp < D; dp = dp + 1u) {
        // Circular wrap: shift = (det + D - dp) % D
        let shift: u32 = (det + D - dp) % D;
        acc = acc + sinogram_input[angle_idx * D + dp] * filter_kernel[shift];
    }
    filtered_output[linear_idx] = acc;
}
