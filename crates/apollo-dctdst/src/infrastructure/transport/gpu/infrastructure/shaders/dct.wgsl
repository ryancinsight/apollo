struct DctParams {
    len: u32,
    mode: u32,
    scale_bits: u32,
    _padding: u32,
}

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

@group(0) @binding(2)
var<uniform> params: DctParams;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(64, 1, 1)
fn dct_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.len {
        return;
    }

    let factor = PI / f32(params.len);
    var sum = 0.0;

    if params.mode == 0u {
        // DCT-II: X[k] = sum_n x[n]*cos(pi/N*(n+0.5)*k)
        for (var n: u32 = 0u; n < params.len; n = n + 1u) {
            let angle = factor * (f32(n) + 0.5) * f32(k);
            sum = sum + input_data[n] * cos(angle);
        }
    } else if params.mode == 1u {
        // DCT-III: X[k] = 0.5*x[0] + sum_{n=1}^{N-1} x[n]*cos(pi/N*n*(k+0.5))
        sum = input_data[0] * 0.5;
        for (var n: u32 = 1u; n < params.len; n = n + 1u) {
            let angle = factor * f32(n) * (f32(k) + 0.5);
            sum = sum + input_data[n] * cos(angle);
        }
    } else if params.mode == 2u {
        // DST-II: X[k] = sum_n x[n]*sin(pi/N*(n+0.5)*(k+1))
        for (var n: u32 = 0u; n < params.len; n = n + 1u) {
            let angle = factor * (f32(n) + 0.5) * (f32(k) + 1.0);
            sum = sum + input_data[n] * sin(angle);
        }
    } else if params.mode == 3u {
        // DST-III: X[k] = (-1)^k*0.5*x[N-1] + sum_{n=0}^{N-2} x[n]*sin(pi/N*(n+1)*(k+0.5))
        let sign = select(1.0, -1.0, (k & 1u) == 1u);
        sum = sign * input_data[params.len - 1u] * 0.5;
        for (var n: u32 = 0u; n + 1u < params.len; n = n + 1u) {
            let angle = factor * (f32(n) + 1.0) * (f32(k) + 0.5);
            sum = sum + input_data[n] * sin(angle);
        }
    } else if params.mode == 4u {
        // DCT-I: X[k] = x[0] + (-1)^k*x[N-1] + 2*sum_{n=1}^{N-2} x[n]*cos(pi*n*k/(N-1))
        // Requires N >= 2; host rejects N < 2 before dispatch so params.len >= 2 here.
        let factor1 = PI / f32(params.len - 1u);
        let sign1 = select(1.0, -1.0, (k & 1u) == 1u);
        sum = input_data[0u] + sign1 * input_data[params.len - 1u];
        for (var n: u32 = 1u; n + 1u < params.len; n = n + 1u) {
            let angle = factor1 * f32(n) * f32(k);
            sum = sum + 2.0 * input_data[n] * cos(angle);
        }
    } else if params.mode == 5u {
        // DCT-IV: X[k] = sum_n x[n]*cos(pi*(n+0.5)*(k+0.5)/N)
        for (var n: u32 = 0u; n < params.len; n = n + 1u) {
            let angle = factor * (f32(n) + 0.5) * (f32(k) + 0.5);
            sum = sum + input_data[n] * cos(angle);
        }
    } else if params.mode == 6u {
        // DST-I: X[k] = 2*sum_n x[n]*sin(pi*(n+1)*(k+1)/(N+1))
        let factor6 = PI / f32(params.len + 1u);
        for (var n: u32 = 0u; n < params.len; n = n + 1u) {
            let angle = factor6 * f32(n + 1u) * f32(k + 1u);
            sum = sum + input_data[n] * sin(angle);
        }
        sum = 2.0 * sum;
    } else {
        // mode == 7u: DST-IV: X[k] = sum_n x[n]*sin(pi*(n+0.5)*(k+0.5)/N)
        for (var n: u32 = 0u; n < params.len; n = n + 1u) {
            let angle = factor * (f32(n) + 0.5) * (f32(k) + 0.5);
            sum = sum + input_data[n] * sin(angle);
        }
    }

    output_data[k] = sum;
}

@compute @workgroup_size(64, 1, 1)
fn dct_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len {
        return;
    }
    output_data[i] = output_data[i] * bitcast<f32>(params.scale_bits);
}
