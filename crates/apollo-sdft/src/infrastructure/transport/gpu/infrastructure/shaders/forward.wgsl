@group(0) @binding(0) var<storage, read> window_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<ComplexValue>;
@group(0) @binding(2) var<uniform> params: SdftParams;

@compute @workgroup_size(64, 1, 1)
fn sdft_direct_bins(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bin = gid.x;
    if bin >= params.bin_count { return; }
    var accumulator = vec2<f32>(0.0, 0.0);
    for (var sample: u32 = 0u; sample < params.window_len; sample = sample + 1u) {
        let angle = -TAU * f32(bin) * f32(sample) / f32(params.window_len);
        accumulator = accumulator + cmul(
            vec2<f32>(window_data[sample], 0.0),
            vec2<f32>(cos(angle), sin(angle)),
        );
    }
    output_data[bin].re = accumulator.x;
    output_data[bin].im = accumulator.y;
}
