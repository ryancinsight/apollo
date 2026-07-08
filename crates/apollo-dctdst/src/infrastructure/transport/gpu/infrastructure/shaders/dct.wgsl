// Batched, strided direct real-to-real DCT/DST kernel.
//
// One dispatch transforms `num_fibers` independent 1-D fibers of length `len`,
// each addressed by a base offset plus a per-element stride. This lets a
// separable 2-D/3-D transform run every axis pass on-device (data stays in GPU
// buffers between passes) instead of downloading each 1-D line to the host.
//
// Fiber `f` (0..num_fibers) decodes into a two-level index `(a, b)`:
//   a = f / fiber_dim_b,  b = f % fiber_dim_b
//   base = a*fiber_stride_a + b*fiber_stride_b
// Element `n` of the fiber lives at `base + n*elem_stride`. This addressing
// covers both axes of a 2-D field and all three axes of a 3-D field. The plain
// 1-D case is `num_fibers = 1, elem_stride = 1, base = 0`.
struct DctParams {
    len: u32,
    mode: u32,
    scale_bits: u32,
    elem_stride: u32,
    num_fibers: u32,
    fiber_stride_a: u32,
    fiber_stride_b: u32,
    fiber_dim_b: u32,
}

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

@group(0) @binding(2)
var<uniform> params: DctParams;

const PI: f32 = 3.14159265358979323846;

fn fiber_base(fiber: u32) -> u32 {
    let a = fiber / params.fiber_dim_b;
    let b = fiber % params.fiber_dim_b;
    return a * params.fiber_stride_a + b * params.fiber_stride_b;
}

@compute @workgroup_size(64, 1, 1)
fn dct_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let fiber = gid.y;
    if k >= params.len || fiber >= params.num_fibers {
        return;
    }
    let base = fiber_base(fiber);
    let es = params.elem_stride;
    let factor = PI / f32(params.len);
    var sum = 0.0;

    if params.mode == 0u {
        // DCT-II: X[k] = sum_n x[n]*cos(pi/N*(n+0.5)*k)
        for (var n: u32 = 0u; n < params.len; n = n + 1u) {
            let angle = factor * (f32(n) + 0.5) * f32(k);
            sum = sum + input_data[base + n * es] * cos(angle);
        }
    } else if params.mode == 1u {
        // DCT-III: X[k] = 0.5*x[0] + sum_{n=1}^{N-1} x[n]*cos(pi/N*n*(k+0.5))
        sum = input_data[base] * 0.5;
        for (var n: u32 = 1u; n < params.len; n = n + 1u) {
            let angle = factor * f32(n) * (f32(k) + 0.5);
            sum = sum + input_data[base + n * es] * cos(angle);
        }
    } else if params.mode == 2u {
        // DST-II: X[k] = sum_n x[n]*sin(pi/N*(n+0.5)*(k+1))
        for (var n: u32 = 0u; n < params.len; n = n + 1u) {
            let angle = factor * (f32(n) + 0.5) * (f32(k) + 1.0);
            sum = sum + input_data[base + n * es] * sin(angle);
        }
    } else if params.mode == 3u {
        // DST-III: X[k] = (-1)^k*0.5*x[N-1] + sum_{n=0}^{N-2} x[n]*sin(pi/N*(n+1)*(k+0.5))
        let sign = select(1.0, -1.0, (k & 1u) == 1u);
        sum = sign * input_data[base + (params.len - 1u) * es] * 0.5;
        for (var n: u32 = 0u; n + 1u < params.len; n = n + 1u) {
            let angle = factor * (f32(n) + 1.0) * (f32(k) + 0.5);
            sum = sum + input_data[base + n * es] * sin(angle);
        }
    } else if params.mode == 4u {
        // DCT-I: X[k] = x[0] + (-1)^k*x[N-1] + 2*sum_{n=1}^{N-2} x[n]*cos(pi*n*k/(N-1))
        // Requires N >= 2; host rejects N < 2 before dispatch so params.len >= 2 here.
        let factor1 = PI / f32(params.len - 1u);
        let sign1 = select(1.0, -1.0, (k & 1u) == 1u);
        sum = input_data[base] + sign1 * input_data[base + (params.len - 1u) * es];
        for (var n: u32 = 1u; n + 1u < params.len; n = n + 1u) {
            let angle = factor1 * f32(n) * f32(k);
            sum = sum + 2.0 * input_data[base + n * es] * cos(angle);
        }
    } else if params.mode == 5u {
        // DCT-IV: X[k] = sum_n x[n]*cos(pi*(n+0.5)*(k+0.5)/N)
        for (var n: u32 = 0u; n < params.len; n = n + 1u) {
            let angle = factor * (f32(n) + 0.5) * (f32(k) + 0.5);
            sum = sum + input_data[base + n * es] * cos(angle);
        }
    } else if params.mode == 6u {
        // DST-I: X[k] = 2*sum_n x[n]*sin(pi*(n+1)*(k+1)/(N+1))
        let factor6 = PI / f32(params.len + 1u);
        for (var n: u32 = 0u; n < params.len; n = n + 1u) {
            let angle = factor6 * f32(n + 1u) * f32(k + 1u);
            sum = sum + input_data[base + n * es] * sin(angle);
        }
        sum = 2.0 * sum;
    } else {
        // mode == 7u: DST-IV: X[k] = sum_n x[n]*sin(pi*(n+0.5)*(k+0.5)/N)
        for (var n: u32 = 0u; n < params.len; n = n + 1u) {
            let angle = factor * (f32(n) + 0.5) * (f32(k) + 0.5);
            sum = sum + input_data[base + n * es] * sin(angle);
        }
    }

    output_data[base + k * es] = sum;
}

// Scale every element of a densely-packed buffer of `num_fibers * len` values.
// Each separable pass writes every position exactly once, so the result is
// dense and the scale runs over the flat element count (fiber layout is
// irrelevant here). The 1-D case reduces to `len`.
@compute @workgroup_size(64, 1, 1)
fn dct_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.num_fibers * params.len;
    if i >= total {
        return;
    }
    output_data[i] = output_data[i] * bitcast<f32>(params.scale_bits);
}
