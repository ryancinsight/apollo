#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
use num_complex::{Complex32, Complex64};

pub(crate) fn stage_triple_scalar_one_impl<C>(
    src: &[C],
    dst: &mut [C],
    src_base: usize,
    dst_base: usize,
    quarter_groups: usize,
    eighth_n: usize,
    quarter_n: usize,
    half_n: usize,
    k: usize,
    w1: C,
    w2a: C,
    w2b: C,
    w3a: C,
    w3b: C,
    w3c: C,
    w3d: C,
) where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    let x0 = src[src_base + k];
    let x1 = src[src_base + quarter_groups + k];
    let x2 = src[src_base + 2 * quarter_groups + k];
    let x3 = src[src_base + 3 * quarter_groups + k];
    let x4 = src[src_base + 4 * quarter_groups + k] * w1;
    let x5 = src[src_base + 5 * quarter_groups + k] * w1;
    let x6 = src[src_base + 6 * quarter_groups + k] * w1;
    let x7 = src[src_base + 7 * quarter_groups + k] * w1;

    let s0 = x0 + x4;
    let s1 = x1 + x5;
    let s2 = x2 + x6;
    let s3 = x3 + x7;
    let d0 = x0 - x4;
    let d1 = x1 - x5;
    let d2 = x2 - x6;
    let d3 = x3 - x7;

    let t2 = s2 * w2a;
    let t3 = s3 * w2a;
    let u2 = d2 * w2b;
    let u3 = d3 * w2b;
    let p0 = s0 + t2;
    let p1 = s1 + t3;
    let p4 = s0 - t2;
    let p5 = s1 - t3;
    let p2 = d0 + u2;
    let p3 = d1 + u3;
    let p6 = d0 - u2;
    let p7 = d1 - u3;

    let q0 = p1 * w3a;
    let q1 = p3 * w3b;
    let q2 = p5 * w3c;
    let q3 = p7 * w3d;
    let out_base = dst_base + k;
    dst[out_base] = p0 + q0;
    dst[half_n + out_base] = p0 - q0;
    dst[eighth_n + out_base] = p2 + q1;
    dst[half_n + eighth_n + out_base] = p2 - q1;
    dst[quarter_n + out_base] = p4 + q2;
    dst[half_n + quarter_n + out_base] = p4 - q2;
    dst[quarter_n + eighth_n + out_base] = p6 + q3;
    dst[half_n + quarter_n + eighth_n + out_base] = p6 - q3;
}

#[inline]
pub(crate) fn stage_triple_impl<C>(
    src: &[C],
    dst: &mut [C],
    radix: usize,
    first_twiddles: &[C],
    second_twiddles: &[C],
    third_twiddles: &[C],
) where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    let n = src.len();
    let groups = n / (radix << 1);
    let quarter_groups = groups >> 2;
    let eighth_n = n >> 3;
    let quarter_n = n >> 2;
    let half_n = n >> 1;

    for j in 0..radix {
        let src_base = j * groups * 2;
        let dst_base = j * quarter_groups;
        for k in 0..quarter_groups {
            stage_triple_scalar_one_impl(
                src,
                dst,
                src_base,
                dst_base,
                quarter_groups,
                eighth_n,
                quarter_n,
                half_n,
                k,
                first_twiddles[j],
                second_twiddles[j],
                second_twiddles[j + radix],
                third_twiddles[j],
                third_twiddles[j + radix],
                third_twiddles[j + 2 * radix],
                third_twiddles[j + 3 * radix],
            );
        }
    }
}

macro_rules! stockham_quad_unrolled {
    (
        $src:ident, $dst:ident, $src_base:expr, $dst_base:expr,
        $sixteenth_groups:expr, $sixteenth_n:expr, $k:expr,
        $w1:expr, $w20:expr, $w21:expr,
        $w30:expr, $w31:expr, $w32:expr, $w33:expr,
        $w40:expr, $w41:expr, $w42:expr, $w43:expr,
        $w44:expr, $w45:expr, $w46:expr, $w47:expr
    ) => {{
        let x0 = $src[$src_base + $k];
        let x1 = $src[$src_base + $sixteenth_groups + $k];
        let x2 = $src[$src_base + 2 * $sixteenth_groups + $k];
        let x3 = $src[$src_base + 3 * $sixteenth_groups + $k];
        let x4 = $src[$src_base + 4 * $sixteenth_groups + $k];
        let x5 = $src[$src_base + 5 * $sixteenth_groups + $k];
        let x6 = $src[$src_base + 6 * $sixteenth_groups + $k];
        let x7 = $src[$src_base + 7 * $sixteenth_groups + $k];
        let x8 = $src[$src_base + 8 * $sixteenth_groups + $k] * $w1;
        let x9 = $src[$src_base + 9 * $sixteenth_groups + $k] * $w1;
        let x10 = $src[$src_base + 10 * $sixteenth_groups + $k] * $w1;
        let x11 = $src[$src_base + 11 * $sixteenth_groups + $k] * $w1;
        let x12 = $src[$src_base + 12 * $sixteenth_groups + $k] * $w1;
        let x13 = $src[$src_base + 13 * $sixteenth_groups + $k] * $w1;
        let x14 = $src[$src_base + 14 * $sixteenth_groups + $k] * $w1;
        let x15 = $src[$src_base + 15 * $sixteenth_groups + $k] * $w1;

        let y0 = x0 + x8;
        let y1 = x1 + x9;
        let y2 = x2 + x10;
        let y3 = x3 + x11;
        let y4 = x4 + x12;
        let y5 = x5 + x13;
        let y6 = x6 + x14;
        let y7 = x7 + x15;
        let y8 = x0 - x8;
        let y9 = x1 - x9;
        let y10 = x2 - x10;
        let y11 = x3 - x11;
        let y12 = x4 - x12;
        let y13 = x5 - x13;
        let y14 = x6 - x14;
        let y15 = x7 - x15;

        let t4 = y4 * $w20;
        let t5 = y5 * $w20;
        let t6 = y6 * $w20;
        let t7 = y7 * $w20;
        let t12 = y12 * $w21;
        let t13 = y13 * $w21;
        let t14 = y14 * $w21;
        let t15 = y15 * $w21;

        let z0 = y0 + t4;
        let z1 = y1 + t5;
        let z2 = y2 + t6;
        let z3 = y3 + t7;
        let z4 = y8 + t12;
        let z5 = y9 + t13;
        let z6 = y10 + t14;
        let z7 = y11 + t15;
        let z8 = y0 - t4;
        let z9 = y1 - t5;
        let z10 = y2 - t6;
        let z11 = y3 - t7;
        let z12 = y8 - t12;
        let z13 = y9 - t13;
        let z14 = y10 - t14;
        let z15 = y11 - t15;

        let u2 = z2 * $w30;
        let u3 = z3 * $w30;
        let u6 = z6 * $w31;
        let u7 = z7 * $w31;
        let u10 = z10 * $w32;
        let u11 = z11 * $w32;
        let u14 = z14 * $w33;
        let u15 = z15 * $w33;

        let p0 = z0 + u2;
        let p1 = z1 + u3;
        let p2 = z4 + u6;
        let p3 = z5 + u7;
        let p4 = z8 + u10;
        let p5 = z9 + u11;
        let p6 = z12 + u14;
        let p7 = z13 + u15;
        let p8 = z0 - u2;
        let p9 = z1 - u3;
        let p10 = z4 - u6;
        let p11 = z5 - u7;
        let p12 = z8 - u10;
        let p13 = z9 - u11;
        let p14 = z12 - u14;
        let p15 = z13 - u15;

        let q1 = p1 * $w40;
        let q3 = p3 * $w41;
        let q5 = p5 * $w42;
        let q7 = p7 * $w43;
        let q9 = p9 * $w44;
        let q11 = p11 * $w45;
        let q13 = p13 * $w46;
        let q15 = p15 * $w47;

        let out_base = $dst_base + $k;
        $dst[out_base] = p0 + q1;
        $dst[out_base + $sixteenth_n] = p2 + q3;
        $dst[out_base + 2 * $sixteenth_n] = p4 + q5;
        $dst[out_base + 3 * $sixteenth_n] = p6 + q7;
        $dst[out_base + 4 * $sixteenth_n] = p8 + q9;
        $dst[out_base + 5 * $sixteenth_n] = p10 + q11;
        $dst[out_base + 6 * $sixteenth_n] = p12 + q13;
        $dst[out_base + 7 * $sixteenth_n] = p14 + q15;
        $dst[out_base + 8 * $sixteenth_n] = p0 - q1;
        $dst[out_base + 9 * $sixteenth_n] = p2 - q3;
        $dst[out_base + 10 * $sixteenth_n] = p4 - q5;
        $dst[out_base + 11 * $sixteenth_n] = p6 - q7;
        $dst[out_base + 12 * $sixteenth_n] = p8 - q9;
        $dst[out_base + 13 * $sixteenth_n] = p10 - q11;
        $dst[out_base + 14 * $sixteenth_n] = p12 - q13;
        $dst[out_base + 15 * $sixteenth_n] = p14 - q15;
    }};
}

/// Fuses four adjacent radix-2 Stockham stages as a radix-16 autosort codelet.
///
/// ## Proof sketch
///
/// Let `r` be the incoming Stockham stride and `G = N/(2r)`. A four-stage
/// fusion is valid only when `G >= 8`; each independent work item is identified
/// by `(j, k)` with `j < r` and `k < G/8`. The 16 inputs are
/// `x_m = src[j*2G + m*(G/8) + k]`.
///
/// For local stage `t` there are `2^t` branches. Branch `b` applies the exact
/// scalar Stockham recurrence to the two halves of a local block of length
/// `16/2^t`, using twiddle table element `W_t[j + b*r]`, then writes sums to
/// the low local half and differences to the high local half. This is the same
/// index relation as `stage64`, restricted to the 16 values reachable from
/// `(j, k)`. Induction over the four local stages proves that the final local
/// array equals four scalar Stockham passes, and storing local index `m` at
/// `dst[j*(G/8) + m*(N/16) + k]` preserves the global autosort order. Only
/// fixed-size stack arrays are used, so the codelet has no heap scratch.
#[inline]
pub(crate) fn stage_quad_impl<C>(
    src: &[C],
    dst: &mut [C],
    radix: usize,
    first_twiddles: &[C],
    second_twiddles: &[C],
    third_twiddles: &[C],
    fourth_twiddles: &[C],
) where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    let n = src.len();
    let groups = n / (radix << 1);
    let sixteenth_groups = groups >> 3;
    let sixteenth_n = n >> 4;

    for j in 0..radix {
        let src_base = j * groups * 2;
        let dst_base = j * sixteenth_groups;
        let w1 = first_twiddles[j];
        let w20 = second_twiddles[j];
        let w21 = second_twiddles[j + radix];
        let w30 = third_twiddles[j];
        let w31 = third_twiddles[j + radix];
        let w32 = third_twiddles[j + 2 * radix];
        let w33 = third_twiddles[j + 3 * radix];
        let w40 = fourth_twiddles[j];
        let w41 = fourth_twiddles[j + radix];
        let w42 = fourth_twiddles[j + 2 * radix];
        let w43 = fourth_twiddles[j + 3 * radix];
        let w44 = fourth_twiddles[j + 4 * radix];
        let w45 = fourth_twiddles[j + 5 * radix];
        let w46 = fourth_twiddles[j + 6 * radix];
        let w47 = fourth_twiddles[j + 7 * radix];
        for k in 0..sixteenth_groups {
            stockham_quad_unrolled!(
                src,
                dst,
                src_base,
                dst_base,
                sixteenth_groups,
                sixteenth_n,
                k,
                w1,
                w20,
                w21,
                w30,
                w31,
                w32,
                w33,
                w40,
                w41,
                w42,
                w43,
                w44,
                w45,
                w46,
                w47
            );
        }
    }
}

/// Fuses two adjacent radix-2 Stockham stages.
///
/// For a starting stage radix `r`, the scalar Stockham recurrence first forms
/// `a = x_0 + w_j x_2` and `b = x_0 - w_j x_2` inside each length-`2r`
/// group. The next stage applies the same recurrence at radix `2r` to the
/// adjacent `k` pairs. Substituting the first equations into the second gives
/// the four outputs below with twiddles `{w_j, w_j', w_{j+r}'}`. The function
/// evaluates that substitution directly, so it removes one full scratch
/// traversal while producing byte-for-byte the same stage order as two scalar
/// Stockham passes.
#[inline]
pub(crate) fn stage_pair_impl<C>(
    src: &[C],
    dst: &mut [C],
    radix: usize,
    first_twiddles: &[C],
    second_twiddles: &[C],
) where
    C: Copy + std::ops::Add<Output = C> + std::ops::Sub<Output = C> + std::ops::Mul<Output = C>,
{
    if radix == 1 {
        let n = src.len();
        let quarter_n = n >> 2;
        let half_n = n >> 1;
        let w3 = second_twiddles[1];
        for k in 0..quarter_n {
            let x0 = src[k];
            let x1 = src[quarter_n + k];
            let x2 = src[half_n + k];
            let x3 = src[half_n + quarter_n + k];
            let a0 = x0 + x2;
            let a1 = x1 + x3;
            let b0 = x0 - x2;
            let b1 = x1 - x3;
            let c1 = b1 * w3;
            dst[k] = a0 + a1;
            dst[half_n + k] = a0 - a1;
            dst[quarter_n + k] = b0 + c1;
            dst[half_n + quarter_n + k] = b0 - c1;
        }
        return;
    }

    let n = src.len();
    let groups = n / (radix << 1);
    let half_groups = groups >> 1;
    let quarter_n = n >> 2;
    let half_n = n >> 1;

    for j in 0..radix {
        let w1 = first_twiddles[j];
        let w2 = second_twiddles[j];
        let w3 = second_twiddles[j + radix];
        let src_base = j * groups * 2;
        let dst_base = j * half_groups;
        for k in 0..half_groups {
            let x0 = src[src_base + k];
            let x1 = src[src_base + half_groups + k];
            let x2 = src[src_base + groups + k] * w1;
            let x3 = src[src_base + groups + half_groups + k] * w1;
            let a0 = x0 + x2;
            let a1 = x1 + x3;
            let b0 = x0 - x2;
            let b1 = x1 - x3;
            let c0 = a1 * w2;
            let c1 = b1 * w3;
            dst[dst_base + k] = a0 + c0;
            dst[dst_base + half_n + k] = a0 - c0;
            dst[dst_base + quarter_n + k] = b0 + c1;
            dst[dst_base + half_n + quarter_n + k] = b0 - c1;
        }
    }
}
