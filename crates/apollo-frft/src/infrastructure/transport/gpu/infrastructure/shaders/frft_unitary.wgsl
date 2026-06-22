// Unitary discrete fractional Fourier transform (Candan 2000).
// Three-pass: V^T*x (step=0), phase (step=1), V*c (step=2).
// V stored column-major (f32): V[row,col] = v_mat[row + col*n].
struct UnitaryParams { len: u32, step: u32, order: f32, _pad: u32 }

@group(0) @binding(0) var<storage, read>       input_data:   array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       v_mat:        array<f32>;
@group(0) @binding(2) var<storage, read_write> intermediate: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output_data:  array<vec2<f32>>;
@group(0) @binding(4) var<uniform>             params:       UnitaryParams;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(64, 1, 1)
fn unitary_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.len;
    if idx >= n { return; }
    if params.step == 0u {
        // V^T * x: c[k] = sum_j v_mat[j + k*n] * x[j], k = idx
        var acc = vec2<f32>(0.0, 0.0);
        for (var j: u32 = 0u; j < n; j++) {
            let v_jk = v_mat[j + idx * n];
            acc += vec2<f32>(v_jk * input_data[j].x, v_jk * input_data[j].y);
        }
        intermediate[idx] = acc;
    } else if params.step == 1u {
        // phase: c[k] *= exp(-i * order * k * pi / 2)
        let phase = -params.order * f32(idx) * (PI / 2.0);
        let c_re = cos(phase);
        let c_im = sin(phase);
        let c = intermediate[idx];
        intermediate[idx] = vec2<f32>(c.x * c_re - c.y * c_im, c.x * c_im + c.y * c_re);
    } else {
        // V * c: y[j] = sum_k v_mat[j + k*n] * c[k], j = idx
        var acc = vec2<f32>(0.0, 0.0);
        for (var k: u32 = 0u; k < n; k++) {
            let v_jk = v_mat[idx + k * n];
            let ck = intermediate[k];
            acc += vec2<f32>(v_jk * ck.x, v_jk * ck.y);
        }
        output_data[idx] = acc;
    }
}
