// Bluestein chirp-Z WGPU kernels for native f16 arithmetic.
//
// # Mathematical contract
//
// Implements the Bluestein (1970) chirp-Z reduction of the N-point DFT to a
// length-M circular convolution (M = next_pow2(2N-1)). Using the identity
//   nk = (n^2 + k^2 - (k-n)^2) / 2
// the DFT reduces to:
//   X[k] = W^{-k^2/2} * (a * h)[k]
// where
//   a[n] = x[n] * W^{n^2/2}  (premultiply, with zero-padding for n >= N)
//   h[j] = W^{-j^2/2}        (chirp kernel, precomputed in host f32 -> f16)
//   (a * h) is the M-point circular convolution via radix-2 FFT
//
// Normalization: the 1/M factor from the inverse radix-2 FFT (fft_scale in
// fft_native_f16.wgsl) already normalizes the convolution correctly.
// chirp_scale is therefore a no-op; it exists only to match the ChirpData
// pipeline-handle layout expected by the Rust dispatch code.
//
// Twiddle precision: all trigonometric values are computed in f32 and then
// narrowed to f16 via explicit cast. This bounds twiddle error at f32 precision
// before the f16 butterfly accumulation, matching the strategy in
// fft_native_f16.wgsl.
//
// Dispatch contract: every entry point uses gid.x as a flat 1D index.
// The Rust dispatcher MUST use:
//   dispatch_workgroups((total + 255) / 256, 1, 1)
// where:
//   chirp_premul   total = params.m * params.batch_count
//   chirp_pointmul total = params.m * params.batch_count
//   chirp_postmul  total = params.n * params.batch_count
//   chirp_negate_im total = params.n * params.batch_count
//   chirp_scale    total = params.n * params.batch_count (no-op body)
//
// This module requires `enable f16;` (wgpu::Features::SHADER_F16).

enable f16;

struct ChirpParams {
    n: u32,
    m: u32,
    batch_count: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read_write> data_re: array<f16>;

@group(0) @binding(1)
var<storage, read_write> data_im: array<f16>;

// chirp_re and chirp_im are treated as read-only by all entry points in this
// module.  The binding type is Storage (read_write) at the WGPU API level so
// that a single BindGroupLayout covers all four bindings uniformly; the shader
// never writes to these two bindings.
@group(0) @binding(2)
var<storage, read_write> chirp_re: array<f16>;

@group(0) @binding(3)
var<storage, read_write> chirp_im: array<f16>;

@group(1) @binding(0)
var<uniform> params: ChirpParams;

// PI in f32 precision; twiddle factors are computed in f32 before narrowing
// to f16 to bound two-source accumulation error.
const PI: f32 = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// chirp_premul
//
// Applies the Bluestein premultiplication:
//   a[n] = x[n] * exp(+pi*i * n^2 / N)   for n in [0, N)
//   a[n] = 0                               for n in [N, M)
//
// Total elements = M * batch_count.  The dispatch covers the full padded
// workspace so that the zero-padding is written explicitly rather than relying
// on buffer initialisation.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn chirp_premul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.m * params.batch_count;
    if idx >= total {
        return;
    }
    let local_idx = idx % params.m;

    // Zero-pad the region [N, M) for each row.
    if local_idx >= params.n {
        data_re[idx] = f16(0.0);
        data_im[idx] = f16(0.0);
        return;
    }

    // Twiddle: exp(+pi*i * n^2 / N) = cos(pi*n^2/N) + i*sin(pi*n^2/N).
    // Computed in f32, narrowed to f16.
    let n_f = f32(local_idx);
    let arg = -PI * n_f * n_f / f32(params.n);
    let cos_arg: f16 = f16(cos(arg));
    let sin_arg: f16 = f16(sin(arg));

    let re = data_re[idx];
    let im = data_im[idx];
    data_re[idx] = re * cos_arg - im * sin_arg;
    data_im[idx] = re * sin_arg + im * cos_arg;
}

// ---------------------------------------------------------------------------
// chirp_pointmul
//
// Pointwise multiplication of the FFT of the premultiplied sequence with the
// precomputed FFT of the chirp kernel H:
//   data[k] *= H[k % M]
//
// Total elements = M * batch_count.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn chirp_pointmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.m * params.batch_count;
    if idx >= total {
        return;
    }

    let a_re = data_re[idx];
    let a_im = data_im[idx];
    let local_idx = idx % params.m;
    let h_re = chirp_re[local_idx];
    let h_im = chirp_im[local_idx];

    data_re[idx] = a_re * h_re - a_im * h_im;
    data_im[idx] = a_re * h_im + a_im * h_re;
}

// ---------------------------------------------------------------------------
// chirp_scale
//
// Applies the inverse-DFT 1/N normalization for a Bluestein axis. The 1/M
// normalization from radix2_inv belongs to the convolution evaluation; it is
// not the inverse transform normalization for the original N-point axis.
//
// Total elements = N * batch_count.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn chirp_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx = gid.x;
    let total = params.n * params.batch_count;
    if linear_idx >= total {
        return;
    }
    let row = linear_idx / params.n;
    let local_idx = linear_idx % params.n;
    let idx = row * params.m + local_idx;
    let inv_n: f16 = f16(1.0 / f32(params.n));
    data_re[idx] = data_re[idx] * inv_n;
    data_im[idx] = data_im[idx] * inv_n;
}

// ---------------------------------------------------------------------------
// chirp_postmul
//
// Applies the Bluestein postmultiplication:
//   X[k] = result[k] * exp(+pi*i * k^2 / N)
//
// This finalises the Bluestein formula X[k] = W^{-k^2/2} * (a * h)[k] where
// the W^{-k^2/2} = exp(+pi*i * k^2 / N) factor is applied here.
//
// Total elements = N * batch_count.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn chirp_postmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx = gid.x;
    let total = params.n * params.batch_count;
    if linear_idx >= total {
        return;
    }
    let row = linear_idx / params.n;
    let local_idx = linear_idx % params.n;
    let idx = row * params.m + local_idx;

    let k_f = f32(local_idx);
    let arg = -PI * k_f * k_f / f32(params.n);
    let cos_arg: f16 = f16(cos(arg));
    let sin_arg: f16 = f16(sin(arg));

    let re = data_re[idx];
    let im = data_im[idx];
    data_re[idx] = re * cos_arg - im * sin_arg;
    data_im[idx] = re * sin_arg + im * cos_arg;
}

// ---------------------------------------------------------------------------
// chirp_negate_im
//
// Negates the imaginary component of the workspace; used in the inverse
// transform path to conjugate the input before applying the forward chirp
// reduction.
//
// Total elements = N * batch_count.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn chirp_negate_im(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_idx = gid.x;
    let total = params.n * params.batch_count;
    if linear_idx >= total {
        return;
    }
    let row = linear_idx / params.n;
    let local_idx = linear_idx % params.n;
    let idx = row * params.m + local_idx;
    data_im[idx] = -data_im[idx];
}
