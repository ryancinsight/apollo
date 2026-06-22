// Graph Fourier Transform compute kernel.
//
// Forward (mode 0): X[k] = sum_i U[i,k] * signal[i]   (U^T x)
// Inverse (mode 1): x[i] = sum_k U[i,k] * X[k]        (U X)
// Basis is column-major: basis_data[i + k*N] = U[i,k].
// U is orthonormal so forward_then_inverse recovers the signal.

// Must match Rust GftParams repr(C) layout (16 bytes).
struct GftParams {
    len:   u32,
    mode:  u32,
    _pad0: u32,
    _pad1: u32,
}

// Bindings: 0=input, 1=output(rw), 2=basis(ro), 3=params(uniform).
@group(0) @binding(0) var<storage, read>       input_data:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<storage, read>       basis_data:  array<f32>;
@group(0) @binding(3) var<uniform>             params:      GftParams;

@compute @workgroup_size(64, 1, 1)
fn gft_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    let n = params.len;
    if out_idx >= n { return; }

    var acc = 0.0f;

    if params.mode == 0u {
        // Forward: X[k] = sum_i basis_data[i + k*n] * input_data[i]
        let k = out_idx;
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            acc = acc + basis_data[i + k * n] * input_data[i];
        }
    } else {
        // Inverse: x[i] = sum_k basis_data[i + k*n] * input_data[k]
        let i = out_idx;
        for (var k: u32 = 0u; k < n; k = k + 1u) {
            acc = acc + basis_data[i + k * n] * input_data[k];
        }
    }

    output_data[out_idx] = acc;
}
