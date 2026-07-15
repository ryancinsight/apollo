@group(0) @binding(0) var<storage, read> bin_data: array<ComplexValue>;
@group(0) @binding(1) var<storage, read_write> output_data: array<ComplexValue>;
@group(0) @binding(2) var<uniform> params: SdftParams;

@compute @workgroup_size(64, 1, 1)
fn sdft_inverse_bins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample = gid.x;
    if sample >= params.window_len { return; }
    var accumulator = vec2<f32>(0.0, 0.0);
    for (var bin: u32 = 0u; bin < params.bin_count; bin = bin + 1u) {
        let angle = TAU * f32(bin) * f32(sample) / f32(params.window_len);
        accumulator = accumulator + cmul(
            vec2<f32>(bin_data[bin].re, bin_data[bin].im),
            vec2<f32>(cos(angle), sin(angle)),
        );
    }
    let scale = 1.0 / f32(params.window_len);
    output_data[sample].re = accumulator.x * scale;
    output_data[sample].im = accumulator.y * scale;
}
