struct SdftParams {
    window_len: u32,
    bin_count: u32,
    padding0: u32,
    padding1: u32,
}

struct ComplexValue {
    re: f32,
    im: f32,
}

const TAU: f32 = 6.283185307179586476925;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}
