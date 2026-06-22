//! Fast 3D Kaiser-Bessel NUFFT WGPU compute shaders.
//!
//! Implements GPU-side Type-1 (spread → FFT → deconvolve) and Type-2
//! (deconvolve → IFFT → interpolate) fast gridded 3D NUFFT paths.
//!
//! Mathematical reference: Fessler & Sutton (2003), IEEE Trans. Signal Process.
//!
//! Normalization convention:
//! - Type-1 path: forward FFT on oversampled grid, then extract/deconvolve.
//!   The forward FFT is unnormalized (no 1/N factor).
//! - Type-2 path: load/deconvolve onto grid, then inverse FFT. The inverse
//!   FFT applied by GpuFft3d::encode_inverse_split divides by the oversampled
//!   length per axis (mx, my, mz), producing a normalized IDFT. The 3D path
//!   uses the normalized IFFT output directly (no pre-scaling needed) because
//!   the CPU 3D type-2 path also works with normalized IDFT values.
//!
//! Grid layout: row-major (x-major) flat index = ix*(my*mz) + iy*mz + iz.
//! Mode layout: row-major flat index = kx*(ny*nz) + ky*nz + kz.
//! Deconvolution layout: flat [deconv_x[0..nx), deconv_y[0..ny), deconv_z[0..nz)].

struct Complex32 {
    re: f32,
    im: f32,
}

struct Position3Pod {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

struct FastNufftParams3D {
    nx: u32, ny: u32, nz: u32, mx: u32,
    my: u32, mz: u32, sample_count: u32, kernel_width: u32,
    lx: f32, ly: f32, lz: f32, beta: f32,
    i0_beta: f32, _pad0: f32, _pad1: f32, _pad2: f32,
}

@group(0) @binding(0)
var<storage, read> positions: array<Position3Pod>;

@group(0) @binding(1)
var<storage, read> values: array<Complex32>;

@group(0) @binding(2)
var<storage, read_write> grid_re: array<f32>;

@group(0) @binding(3)
var<storage, read_write> grid_im: array<f32>;

@group(0) @binding(4)
var<storage, read> deconv_xyz: array<f32>;

@group(0) @binding(5)
var<storage, read_write> output_values: array<Complex32>;

@group(0) @binding(6)
var<storage, read> coefficients: array<Complex32>;

@group(0) @binding(7)
var<uniform> params: FastNufftParams3D;

// Modified Bessel function I_0(x) via convergent series.
// I_0(x) = sum_{k=0}^inf (x^2/4)^k / (k!)^2; terms are positive for real x.
fn i0(value: f32) -> f32 {
    let scaled = 0.25 * value * value;
    var sum = 1.0;
    var term = 1.0;
    for (var k: u32 = 1u; k <= 128u; k = k + 1u) {
        let kf = f32(k);
        term = term * scaled / (kf * kf);
        sum = sum + term;
        if term <= 1.1920929e-7 * sum {
            break;
        }
    }
    return sum;
}

// Kaiser-Bessel kernel: phi(x;W,beta) = I_0(beta*sqrt(1-(x/W)^2)) / I_0(beta)
// Returns 0 for |x| >= W.
fn kb_kernel(delta: f32) -> f32 {
    let width = f32(params.kernel_width);
    let u2 = (delta / width) * (delta / width);
    if u2 >= 1.0 {
        return 0.0;
    }
    return i0(params.beta * sqrt(1.0 - u2)) / params.i0_beta;
}

// Signed FFT index: [0..N/2] -> [0..N/2], (N/2..N) -> (N/2-N..0)
fn signed_index(index: u32, len: u32) -> i32 {
    if index <= len / 2u {
        return i32(index);
    }
    return i32(index) - i32(len);
}

// Wrap periodic grid delta to nearest-neighbor range [-m/2, m/2].
fn periodic_delta_wrap(raw: f32, m: f32) -> f32 {
    return raw - round(raw / m) * m;
}

// Periodic-modulo position: x mod L in [0, L).
fn x_mod_x(s: u32) -> f32 {
    let x = positions[s].x;
    return x - floor(x / params.lx) * params.lx;
}
fn x_mod_y(s: u32) -> f32 {
    let y = positions[s].y;
    return y - floor(y / params.ly) * params.ly;
}
fn x_mod_z(s: u32) -> f32 {
    let z = positions[s].z;
    return z - floor(z / params.lz) * params.lz;
}

// Map mode index k to oversampled grid index: g = (signed_index(k,n) mod m + m) mod m.
fn mode_to_grid_idx(k: u32, n: u32, m: u32) -> u32 {
    let s = signed_index(k, n);
    return u32(((s % i32(m)) + i32(m)) % i32(m));
}

// --- Type-1 fast: spread non-uniform samples to oversampled grid ---
// One thread per oversampled grid cell (flat = ix*(my*mz) + iy*mz + iz).
// For each sample j, accumulates weighted contribution using the KB kernel
// independently on each axis (separable spreading).
@compute @workgroup_size(64, 1, 1)
fn fast_type1_spread_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if flat >= params.mx * params.my * params.mz {
        return;
    }
    let ix = flat / (params.my * params.mz);
    let iy = (flat / params.mz) % params.my;
    let iz = flat % params.mz;

    var re = 0.0;
    var im = 0.0;
    for (var s = 0u; s < params.sample_count; s = s + 1u) {
        let tx = f32(params.mx) * x_mod_x(s) / params.lx;
        let ty = f32(params.my) * x_mod_y(s) / params.ly;
        let tz = f32(params.mz) * x_mod_z(s) / params.lz;
        let wx = kb_kernel(periodic_delta_wrap(f32(ix) - tx, f32(params.mx)));
        if wx != 0.0 {
            let wy = kb_kernel(periodic_delta_wrap(f32(iy) - ty, f32(params.my)));
            if wy != 0.0 {
                let wz = kb_kernel(periodic_delta_wrap(f32(iz) - tz, f32(params.mz)));
                if wz != 0.0 {
                    let w = wx * wy * wz;
                    re = re + values[s].re * w;
                    im = im + values[s].im * w;
                }
            }
        }
    }
    grid_re[flat] = re;
    grid_im[flat] = im;
}

// --- Type-1 fast: deconvolve and extract output modes from FFT'd grid ---
// One thread per output mode (flat = kx*(ny*nz) + ky*nz + kz).
// Applies separable deconvolution: D[kx,ky,kz] = D_x[kx]*D_y[ky]*D_z[kz].
@compute @workgroup_size(64, 1, 1)
fn fast_type1_extract_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if flat >= params.nx * params.ny * params.nz {
        return;
    }
    let kx = flat / (params.ny * params.nz);
    let ky = (flat / params.nz) % params.ny;
    let kz = flat % params.nz;

    let mx_idx = mode_to_grid_idx(kx, params.nx, params.mx);
    let my_idx = mode_to_grid_idx(ky, params.ny, params.my);
    let mz_idx = mode_to_grid_idx(kz, params.nz, params.mz);

    let grid_flat = mx_idx * params.my * params.mz + my_idx * params.mz + mz_idx;
    // Separable deconvolution factor (stored concatenated in deconv_xyz)
    let scale = deconv_xyz[kx] * deconv_xyz[params.nx + ky] * deconv_xyz[params.nx + params.ny + kz];
    output_values[flat] = Complex32(grid_re[grid_flat] * scale, grid_im[grid_flat] * scale);
}

// --- Type-2 fast: load deconvolved Fourier coefficients onto oversampled grid ---
// One thread per oversampled grid cell. Determines the unique mode (kx,ky,kz)
// (if any) that maps to this cell, applies separable deconvolution, and writes it.
// The reverse mapping from grid index to mode index mirrors fft_signed_index:
//   ix <= mx/2 and ix <= nx/2 → mode kx = ix (positive bin)
//   ix > mx/2: signed = ix - mx; candidate = signed + nx;
//              if candidate in (nx/2, nx) → mode kx = candidate (negative bin)
@compute @workgroup_size(64, 1, 1)
fn fast_type2_load_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if flat >= params.mx * params.my * params.mz {
        return;
    }
    let ix = flat / (params.my * params.mz);
    let iy = (flat / params.mz) % params.my;
    let iz = flat % params.mz;

    // x-axis reverse mapping
    var kx = 0u;
    var active_x = false;
    if ix <= params.mx / 2u {
        if ix <= params.nx / 2u {
            active_x = true;
            kx = ix;
        }
    } else {
        let signed_ix = i32(ix) - i32(params.mx);
        let candidate_x = signed_ix + i32(params.nx);
        if candidate_x > i32(params.nx / 2u) && candidate_x < i32(params.nx) {
            active_x = true;
            kx = u32(candidate_x);
        }
    }

    // y-axis reverse mapping
    var ky = 0u;
    var active_y = false;
    if iy <= params.my / 2u {
        if iy <= params.ny / 2u {
            active_y = true;
            ky = iy;
        }
    } else {
        let signed_iy = i32(iy) - i32(params.my);
        let candidate_y = signed_iy + i32(params.ny);
        if candidate_y > i32(params.ny / 2u) && candidate_y < i32(params.ny) {
            active_y = true;
            ky = u32(candidate_y);
        }
    }

    // z-axis reverse mapping
    var kz = 0u;
    var active_z = false;
    if iz <= params.mz / 2u {
        if iz <= params.nz / 2u {
            active_z = true;
            kz = iz;
        }
    } else {
        let signed_iz = i32(iz) - i32(params.mz);
        let candidate_z = signed_iz + i32(params.nz);
        if candidate_z > i32(params.nz / 2u) && candidate_z < i32(params.nz) {
            active_z = true;
            kz = u32(candidate_z);
        }
    }

    if active_x && active_y && active_z {
        let flat_mode = kx * params.ny * params.nz + ky * params.nz + kz;
        let scale = deconv_xyz[kx] * deconv_xyz[params.nx + ky] * deconv_xyz[params.nx + params.ny + kz];
        grid_re[flat] = coefficients[flat_mode].re * scale;
        grid_im[flat] = coefficients[flat_mode].im * scale;
    } else {
        grid_re[flat] = 0.0;
        grid_im[flat] = 0.0;
    }
}

// --- Type-2 fast: interpolate IFFT'd grid at non-uniform positions ---
// One thread per non-uniform sample. Reads from the IFFT'd oversampled grid and
// accumulates the separable KB kernel-weighted sum across all grid cells.
@compute @workgroup_size(64, 1, 1)
fn fast_type2_interpolate_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample = gid.x;
    if sample >= params.sample_count {
        return;
    }
    let tx = f32(params.mx) * x_mod_x(sample) / params.lx;
    let ty = f32(params.my) * x_mod_y(sample) / params.ly;
    let tz = f32(params.mz) * x_mod_z(sample) / params.lz;

    var re = 0.0;
    var im = 0.0;
    for (var ix = 0u; ix < params.mx; ix = ix + 1u) {
        let wx = kb_kernel(periodic_delta_wrap(f32(ix) - tx, f32(params.mx)));
        if wx != 0.0 {
            for (var iy = 0u; iy < params.my; iy = iy + 1u) {
                let wy = kb_kernel(periodic_delta_wrap(f32(iy) - ty, f32(params.my)));
                if wy != 0.0 {
                    let wxy = wx * wy;
                    for (var iz = 0u; iz < params.mz; iz = iz + 1u) {
                        let wz = kb_kernel(periodic_delta_wrap(f32(iz) - tz, f32(params.mz)));
                        if wz != 0.0 {
                            let gflat = ix * params.my * params.mz + iy * params.mz + iz;
                            re = re + grid_re[gflat] * wxy * wz;
                            im = im + grid_im[gflat] * wxy * wz;
                        }
                    }
                }
            }
        }
    }
    output_values[sample] = Complex32(re, im);
}
