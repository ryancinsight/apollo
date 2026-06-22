// Short-Time Fourier Transform Bluestein/Chirp-Z kernel for non-power-of-two frame lengths.
//
// Implements Bluestein's identity (Rabiner, Schafer & Rader, 1969; Bluestein, 1970):
//
//   X[k] = exp(-pi*i*k^2/N) * conv(f, h)[k]
//
// where W = exp(-2*pi*i/N) for the forward DFT, and:
//   f[n] = x[n] * exp(-pi*i*n^2/N)   (premultiply)
//   h[n] = exp(+pi*i*n^2/N)           (convolution kernel, stored conjugated)
//   X[k] = exp(-pi*i*k^2/N) * conv(f,h)[k]   (postmultiply)
//
// Kernel storage: h_stored[n] = exp(-pi*i*n^2/N) = conj(h[n]).
//   Forward path (stft_chirp_pointmul_fwd) conjugates h_stored to recover h_fwd.
//   Inverse path (stft_chirp_pointmul)     uses h_stored = h_inv directly.
//
// ## Forward dispatch sequence
//   Pass A (stft_chirp_premul_fwd):    f[n] = hann(n)*x[n] * exp(-pi*i*n^2/N)
//   Pass B: Radix-2 forward FFT over M.
//   Pass C (stft_chirp_pointmul_fwd):  pointwise multiply by conj(h_stored) = h_fwd.
//   Pass D: Radix-2 inverse FFT over M.
//   Pass E (stft_chirp_postmul_fwd):   X[k] = conv[k] * exp(-pi*i*k^2/N)
//
// ## Inverse dispatch sequence
//   Pass A (stft_chirp_premul_inv):    g[k] = X[k] * exp(+pi*i*k^2/N)
//   Passes B-D: identical to forward (using h_stored = h_inv).
//   Pass E (stft_chirp_postmul_inv):   x[n] = conv[n] * exp(+pi*i*n^2/N) / N; take real part.

const TAU:   f32 = 6.28318530717958647692;
const PI:    f32 = 3.14159265358979323846;

struct ComplexValue {
    re: f32,
    im: f32,
}

struct StftChirpParams {
    frame_count: u32,
    frame_len:   u32,
    chirp_len:   u32,
    hop_len:     u32,
    signal_len:  u32,
    _pad0:       u32,
    _pad1:       u32,
    _pad2:       u32,
}

@group(0) @binding(0) var<storage, read_write> chirp_re:  array<f32>;
@group(0) @binding(1) var<storage, read_write> chirp_im:  array<f32>;
@group(0) @binding(2) var<storage, read>       h_fft_re:  array<f32>;
@group(0) @binding(3) var<storage, read>       h_fft_im:  array<f32>;
@group(1) @binding(0) var<uniform> cp: StftChirpParams;
@group(2) @binding(0) var<storage, read>       input_data:  array<f32>;
@group(2) @binding(1) var<storage, read_write> output_data: array<ComplexValue>;

fn hann_analysis(n: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u { return 1.0; }
    return 0.5 - 0.5 * cos(TAU * f32(n) / f32(frame_len - 1u));
}

// Pass A (forward): premultiply x[n] * exp(-pi*i*n^2/N), zero-pad to M.
@compute @workgroup_size(256, 1, 1)
fn stft_chirp_premul_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = cp.frame_count * cp.chirp_len;
    if idx >= total { return; }
    let frame_m: u32 = idx / cp.chirp_len;
    let n:       u32 = idx % cp.chirp_len;
    if n >= cp.frame_len {
        chirp_re[idx] = 0.0;
        chirp_im[idx] = 0.0;
        return;
    }
    let center: u32 = frame_m * cp.hop_len;
    let half:   u32 = cp.frame_len / 2u;
    var sig_re: f32 = 0.0;
    if n + center >= half && n + center - half < cp.signal_len {
        let sig_idx: u32 = n + center - half;
        sig_re = hann_analysis(n, cp.frame_len) * input_data[sig_idx];
    }
    // exp(-pi*i*n^2/N) = (cos, -sin)
    let arg: f32 = PI * f32(n) * f32(n) / f32(cp.frame_len);
    chirp_re[idx] =  sig_re * cos(arg);
    chirp_im[idx] = -sig_re * sin(arg);
}

// Pass A (inverse): premultiply X[k] * exp(+pi*i*k^2/N), zero-pad to M.
@compute @workgroup_size(256, 1, 1)
fn stft_chirp_premul_inv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = cp.frame_count * cp.chirp_len;
    if idx >= total { return; }
    let frame_m: u32 = idx / cp.chirp_len;
    let n:       u32 = idx % cp.chirp_len;
    if n >= cp.frame_len {
        chirp_re[idx] = 0.0;
        chirp_im[idx] = 0.0;
        return;
    }
    let spec_idx: u32 = 2u * (frame_m * cp.frame_len + n);
    let re: f32 = input_data[spec_idx];
    let im: f32 = input_data[spec_idx + 1u];
    // exp(+pi*i*n^2/N) = (cos, +sin): (re+i*im)*(cos+i*sin) = (re*cos-im*sin) + i*(re*sin+im*cos)
    let arg: f32 = PI * f32(n) * f32(n) / f32(cp.frame_len);
    let c:   f32 = cos(arg);
    let s:   f32 = sin(arg);
    chirp_re[idx] = re * c - im * s;
    chirp_im[idx] = re * s + im * c;
}

// Pass C (inverse): pointwise multiply by h_stored = exp(-pi*i*j^2/N) (correct for inverse).
@compute @workgroup_size(256, 1, 1)
fn stft_chirp_pointmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = cp.frame_count * cp.chirp_len;
    if idx >= total { return; }
    let local_j: u32 = idx % cp.chirp_len;
    let a_re: f32 = chirp_re[idx];
    let a_im: f32 = chirp_im[idx];
    let h_re: f32 = h_fft_re[local_j];
    let h_im: f32 = h_fft_im[local_j];
    chirp_re[idx] = a_re * h_re - a_im * h_im;
    chirp_im[idx] = a_re * h_im + a_im * h_re;
}

// Pass C (forward): pointwise multiply by conj(h_stored) = h_fwd = exp(+pi*i*j^2/N).
@compute @workgroup_size(256, 1, 1)
fn stft_chirp_pointmul_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = cp.frame_count * cp.chirp_len;
    if idx >= total { return; }
    let local_j: u32 = idx % cp.chirp_len;
    let a_re: f32 = chirp_re[idx];
    let a_im: f32 = chirp_im[idx];
    let h_re: f32 =  h_fft_re[local_j];
    let h_im: f32 = -h_fft_im[local_j];  // conjugate: h_stored -> h_fwd
    chirp_re[idx] = a_re * h_re - a_im * h_im;
    chirp_im[idx] = a_re * h_im + a_im * h_re;
}

// Pass E (forward): postmultiply conv[k] * exp(-pi*i*k^2/N), write complex output.
@compute @workgroup_size(256, 1, 1)
fn stft_chirp_postmul_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = cp.frame_count * cp.frame_len;
    if idx >= total { return; }
    let frame_m: u32 = idx / cp.frame_len;
    let k:       u32 = idx % cp.frame_len;
    let chirp_idx: u32 = frame_m * cp.chirp_len + k;
    let y_re: f32 = chirp_re[chirp_idx];
    let y_im: f32 = chirp_im[chirp_idx];
    // exp(-pi*i*k^2/N) = (cos, -sin): (y_re+i*y_im)*(cos-i*sin) = (y_re*cos+y_im*sin) + i*(-y_re*sin+y_im*cos)
    let arg: f32 = PI * f32(k) * f32(k) / f32(cp.frame_len);
    let c:   f32 = cos(arg);
    let s:   f32 = sin(arg);
    output_data[idx] = ComplexValue(y_re * c + y_im * s, -y_re * s + y_im * c);
}

// Pass E (inverse): postmultiply conv[n] * exp(+pi*i*n^2/N) / N, apply Hann, write real.
@group(2) @binding(1) var<storage, read_write> frame_data_out: array<f32>;

fn hann_synthesis(k: u32, frame_len: u32) -> f32 {
    if frame_len <= 1u { return 1.0; }
    return 0.5 - 0.5 * cos(TAU * f32(k) / f32(frame_len - 1u));
}

@compute @workgroup_size(256, 1, 1)
fn stft_chirp_postmul_inv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx:   u32 = gid.x;
    let total: u32 = cp.frame_count * cp.frame_len;
    if idx >= total { return; }
    let frame_m: u32 = idx / cp.frame_len;
    let k:       u32 = idx % cp.frame_len;
    let chirp_idx: u32 = frame_m * cp.chirp_len + k;
    let y_re: f32 = chirp_re[chirp_idx];
    let y_im: f32 = chirp_im[chirp_idx];
    // exp(+pi*i*k^2/N) = (cos, +sin): real part = y_re*cos - y_im*sin.
    // Scale by 1/N; 1/M is absorbed by the Radix-2 IFFT.
    let arg: f32 = PI * f32(k) * f32(k) / f32(cp.frame_len);
    let c:   f32 = cos(arg);
    let s:   f32 = sin(arg);
    let x_re: f32 = (y_re * c - y_im * s) / f32(cp.frame_len);
    frame_data_out[idx] = hann_synthesis(k, cp.frame_len) * x_re;
}
