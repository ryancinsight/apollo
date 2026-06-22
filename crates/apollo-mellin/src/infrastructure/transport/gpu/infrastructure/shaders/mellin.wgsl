struct ComplexValue {
    re: f32,
    im: f32,
}

struct MellinParams {
    signal_len: u32,
    samples: u32,
    signal_min: f32,
    signal_max: f32,
    log_min: f32,
    log_max: f32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0)
var<storage, read> input_signal: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_values: array<f32>;

@group(0) @binding(2)
var<uniform> params: MellinParams;

@group(0) @binding(0)
var<storage, read> log_samples: array<f32>;

@group(0) @binding(1)
var<storage, read_write> spectrum_out: array<ComplexValue>;

const TAU: f32 = 6.28318530717958647692;

@compute @workgroup_size(64, 1, 1)
fn mellin_resample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.samples {
        return;
    }

    if params.signal_len == 0u {
        output_values[i] = 0.0;
        return;
    }

    let log_step = select(
        0.0,
        (params.log_max - params.log_min) / f32(params.samples - 1u),
        params.samples > 1u
    );
    let current_scale = exp(params.log_min + f32(i) * log_step);
    let domain_width = params.signal_max - params.signal_min;
    if current_scale < params.signal_min || current_scale > params.signal_max || domain_width <= 0.0 {
        output_values[i] = 0.0;
        return;
    }

    if params.signal_len == 1u {
        output_values[i] = input_signal[0u];
        return;
    }

    let fraction = (current_scale - params.signal_min) / domain_width;
    let exact_idx = fraction * f32(params.signal_len - 1u);
    let lower_idx = u32(floor(exact_idx));
    let upper_idx = min(lower_idx + 1u, params.signal_len - 1u);
    let weight = exact_idx - f32(lower_idx);
    output_values[i] = input_signal[lower_idx] * (1.0 - weight) + input_signal[upper_idx] * weight;
}

@compute @workgroup_size(64, 1, 1)
fn mellin_spectrum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.samples {
        return;
    }

    let du = select(
        1.0,
        (params.log_max - params.log_min) / f32(params.samples - 1u),
        params.samples > 1u
    );
    let factor = -TAU / f32(params.samples);
    var sum_re = 0.0;
    var sum_im = 0.0;
    for (var n: u32 = 0u; n < params.samples; n = n + 1u) {
        let angle = factor * f32(k * n);
        let sample = log_samples[n];
        sum_re = sum_re + sample * cos(angle);
        sum_im = sum_im + sample * sin(angle);
    }
    spectrum_out[k].re = du * sum_re;
    spectrum_out[k].im = du * sum_im;
}

// ---------------------------------------------------------------------------
// Inverse Mellin kernels
// ---------------------------------------------------------------------------
//
// Inverse step 1: recover log-domain samples from spectrum via IDFT.
//   g[n] = (1 / (N * du)) * Re{ sum_k F[k] * exp(+2*pi*i*k*n/N) }
//
// Inverse step 2: exp-resample g back to a linear output domain.
//
// The two passes share the `MellinParams` uniform; log_samples / spectrum_out
// bindings are swapped relative to the forward passes (spectrum is now input).
// A second params struct `InverseMellinParams` carries the output domain.

struct InverseMellinParams {
    samples:    u32,
    out_len:    u32,
    log_min:    f32,
    log_max:    f32,
    out_min:    f32,
    out_max:    f32,
    _pad0:      u32,
    _pad1:      u32,
}

@group(0) @binding(0)
var<storage, read> spectrum_in: array<ComplexValue>;

@group(0) @binding(1)
var<storage, read_write> inv_log_samples: array<f32>;

@group(0) @binding(2)
var<uniform> inv_params: InverseMellinParams;

/// IDFT of spectrum → log-domain samples.
@compute @workgroup_size(64, 1, 1)
fn mellin_inverse_spectrum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if n >= inv_params.samples {
        return;
    }

    let N = inv_params.samples;
    let log_min = inv_params.log_min;
    let log_max = inv_params.log_max;
    let du = select(
        1.0,
        (log_max - log_min) / f32(N - 1u),
        N > 1u
    );
    let inv_du = select(1.0, 1.0 / du, du > 1e-30);
    let factor = TAU / f32(N);

    var re_sum = 0.0;
    for (var k: u32 = 0u; k < N; k = k + 1u) {
        let angle = factor * f32(k * n);
        re_sum = re_sum + spectrum_in[k].re * cos(angle) - spectrum_in[k].im * sin(angle);
    }
    // g[n] = (1/(N*du)) * re_sum  (real-valued: imaginary part vanishes for real signals)
    inv_log_samples[n] = re_sum * inv_du / f32(N);
}

// ---------------------------------------------------------------------------
// Inverse step 2: exp-resample log-domain samples → linear output signal.

struct ExpResampleParams {
    samples:    u32,
    out_len:    u32,
    log_min:    f32,
    log_max:    f32,
    out_min:    f32,
    out_max:    f32,
    _pad0:      u32,
    _pad1:      u32,
}

@group(0) @binding(0)
var<storage, read> er_log_samples: array<f32>;

@group(0) @binding(1)
var<storage, read_write> er_output: array<f32>;

@group(0) @binding(2)
var<uniform> er_params: ExpResampleParams;

/// Exp-resample: map log-domain g[n] back to a linear output grid.
@compute @workgroup_size(64, 1, 1)
fn mellin_exp_resample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= er_params.out_len {
        return;
    }

    let N = er_params.samples;
    let out_step = select(
        0.0,
        (er_params.out_max - er_params.out_min) / f32(er_params.out_len - 1u),
        er_params.out_len > 1u
    );
    let r = er_params.out_min + f32(i) * out_step;
    if r <= 0.0 {
        er_output[i] = 0.0;
        return;
    }

    let u = log(r);
    let log_min = er_params.log_min;
    let log_max = er_params.log_max;
    if u < log_min || u > log_max {
        er_output[i] = 0.0;
        return;
    }

    let du = select(0.0, (log_max - log_min) / f32(N - 1u), N > 1u);
    if du < 1e-30 {
        er_output[i] = er_log_samples[0];
        return;
    }

    let exact_idx = (u - log_min) / du;
    let lower = u32(floor(exact_idx));
    let upper = min(lower + 1u, N - 1u);
    let frac = exact_idx - f32(lower);
    er_output[i] = er_log_samples[lower] * (1.0 - frac) + er_log_samples[upper] * frac;
}
