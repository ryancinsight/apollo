@group(0) @binding(0)
var<storage, read> grid_values: array<GridSample>;

@group(0) @binding(1)
var<storage, read_write> generated_basis: array<Complex32>;

@group(0) @binding(2)
var<uniform> basis_params: BasisParams;

@compute @workgroup_size(64, 1, 1)
fn sht_basis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let total = basis_params.mode_count * basis_params.sample_count;
    if (index >= total) {
        return;
    }

    var mode = 0u;
    var sample_index = 0u;
    if (basis_params.weighted == 1u) {
        mode = index / basis_params.sample_count;
        sample_index = index - mode * basis_params.sample_count;
    } else {
        sample_index = index / basis_params.mode_count;
        mode = index - sample_index * basis_params.mode_count;
    }

    let degree = mode_degree(mode);
    let order = mode_order(mode, degree);
    let sample = grid_values[sample_index];
    var value = spherical_harmonic(degree, order, sample);
    if (basis_params.conjugate == 1u) {
        value = complex_conj(value);
    }
    if (basis_params.weighted == 1u) {
        value = Complex32(value.re * sample.weight, value.im * sample.weight);
    }
    generated_basis[index] = value;
}
