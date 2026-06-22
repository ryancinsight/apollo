// Direct discrete fractional Fourier transform.
// Rotation angle: alpha = order * pi/2.
// Centred coordinates: xj = j - (N-1)/2, uk = k - (N-1)/2.
// mode 0 = identity, mode 1 = centred DFT,
// mode 2 = reversal, mode 3 = centred IDFT, mode 4 = general FrFT.

struct ComplexValue { re: f32, im: f32 }

// Must match Rust FrftParams repr(C) layout exactly (32 bytes).
struct FrftParams {
    len:      u32,
    mode:     u32,
    cot:      f32,
    csc:      f32,
    scale_re: f32,
    scale_im: f32,
    _pad0:    u32,
    _pad1:    u32,
}

@group(0) @binding(0) var<storage, read>       input_data:  array<ComplexValue>;
@group(0) @binding(1) var<storage, read_write> output_data: array<ComplexValue>;
@group(0) @binding(2) var<uniform>             params:      FrftParams;

const PI:  f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y,
                     a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64, 1, 1)
fn frft_transform(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let n = params.len;
    if k >= n { return; }

    let c  = 0.5 * f32(n - 1u);
    let uk = f32(k) - c;

    if params.mode == 0u {
        output_data[k].re = input_data[k].re;
        output_data[k].im = input_data[k].im;
        return;
    }
    if params.mode == 2u {
        output_data[k].re = input_data[n - 1u - k].re;
        output_data[k].im = input_data[n - 1u - k].im;
        return;
    }

    // Modes 1 (centred DFT), 3 (centred IDFT), 4 (general FrFT).
    let inv_n = 1.0 / f32(n);
    var acc   = vec2<f32>(0.0, 0.0);

    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let xj    = f32(j) - c;
        let x_val = vec2<f32>(input_data[j].re, input_data[j].im);
        var angle: f32 = 0.0;
        if params.mode == 1u {
            angle = -TAU * xj * uk * inv_n;
        } else if params.mode == 3u {
            angle = TAU * xj * uk * inv_n;
        } else {
            angle = PI * ((xj * xj + uk * uk) * params.cot
                          - 2.0 * xj * uk * params.csc) * inv_n;
        }
        let twiddle = vec2<f32>(cos(angle), sin(angle));
        acc = acc + cmul(x_val, twiddle);
    }

    if params.mode == 4u {
        let scale = vec2<f32>(params.scale_re, params.scale_im);
        let out   = cmul(acc, scale);
        output_data[k].re = out.x;
        output_data[k].im = out.y;
    } else {
        let norm = 1.0 / sqrt(f32(n));
        output_data[k].re = acc.x * norm;
        output_data[k].im = acc.y * norm;
    }
}
