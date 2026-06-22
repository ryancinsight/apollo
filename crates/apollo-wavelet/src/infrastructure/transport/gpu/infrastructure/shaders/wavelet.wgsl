// Haar Discrete Wavelet Transform analysis and synthesis kernels.
//
// Orthogonality proof (analysis pass):
//   A[k] = (x[2k] + x[2k+1]) / sqrt(2)   -- lowpass  (scaling function)
//   D[k] = (x[2k] - x[2k+1]) / sqrt(2)   -- highpass (wavelet function)
// Perfect reconstruction (synthesis pass):
//   x[2k]   = (A[k] + D[k]) / sqrt(2)
//   x[2k+1] = (A[k] - D[k]) / sqrt(2)
// Proof: substitute A, D: x[2k] = ((a+b)/r + (a-b)/r)/r = 2a/(r*r) = a. QED.
// where r = sqrt(2).
//
// Multi-pass strategy (forward, L levels):
//   Level 0: input[0..N]     -> [A0|D0]  (A0 in [0..N/2], D0 in [N/2..N])
//   Level 1: input[0..N/2]   -> [A1|D1]  writes only first N/2 positions
//   ...
//   Final:   [A_L | D_{L-1} | ... | D_0]  (Mallat ordering)
// After each pass, the output buffer is copied back to the input buffer so
// the detail coefficients placed in the upper half of previous levels are
// preserved while only the approximation (lower half) is refined.

struct WaveletParams {
    // Current working length for this pass.
    len:  u32,
    _p0:  u32,
    _p1:  u32,
    _p2:  u32,
}

@group(0) @binding(0) var<storage, read>       input_data:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform>             params:      WaveletParams;

// 1/sqrt(2) -- Haar orthonormal scaling coefficient.
const SQRT2_INV: f32 = 0.70710678118654752440;

// Forward Haar analysis pass.
// Reads input_data[0..len], writes:
//   output_data[0..len/2]         = approximation coefficients
//   output_data[len/2..len]       = detail coefficients
@compute @workgroup_size(256, 1, 1)
fn haar_analysis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k    = gid.x;
    let half = params.len / 2u;
    if k >= half { return; }
    let a = input_data[2u * k];
    let b = input_data[2u * k + 1u];
    output_data[k]          = (a + b) * SQRT2_INV;
    output_data[half + k]   = (a - b) * SQRT2_INV;
}

// Inverse Haar synthesis pass.
// Reads input_data[0..len] where [0..len/2]=approx, [len/2..len]=detail.
// Writes output_data[0..len] = reconstructed signal.
@compute @workgroup_size(256, 1, 1)
fn haar_synthesis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k    = gid.x;
    let half = params.len / 2u;
    if k >= half { return; }
    let a = input_data[k];
    let d = input_data[half + k];
    output_data[2u * k]       = (a + d) * SQRT2_INV;
    output_data[2u * k + 1u]  = (a - d) * SQRT2_INV;
}
