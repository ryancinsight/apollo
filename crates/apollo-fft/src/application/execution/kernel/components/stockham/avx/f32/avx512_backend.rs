use super::super::backend::StockhamAvxBackend;
use num_complex::Complex32;
use std::arch::x86_64::*;

#[derive(Copy, Clone)]
pub(crate) struct Avx512BackendF32;

impl StockhamAvxBackend for Avx512BackendF32 {
    type Real = f32;
    type Complex = Complex32;
    type Vector = __m512;

    const COMPLEX_PER_VECTOR: usize = 8;

    #[inline(always)]
    unsafe fn unpack_complex(c: Complex32) -> (f32, f32) {
        (c.re, c.im)
    }

    #[inline(always)]
    unsafe fn complex_mul(a: Complex32, b: Complex32) -> Complex32 {
        a * b
    }

    #[inline(always)]
    unsafe fn complex_add(a: Complex32, b: Complex32) -> Complex32 {
        a + b
    }

    #[inline(always)]
    unsafe fn complex_sub(a: Complex32, b: Complex32) -> Complex32 {
        a - b
    }

    #[inline(always)]
    unsafe fn set1_real(val: f32) -> __m512 {
        _mm512_set1_ps(val)
    }

    #[inline(always)]
    unsafe fn set1_imag(val: f32) -> __m512 {
        _mm512_set1_ps(val)
    }

    #[inline(always)]
    unsafe fn loadu_complex(ptr: *const Complex32) -> __m512 {
        _mm512_loadu_ps(ptr.cast::<f32>())
    }

    #[inline(always)]
    unsafe fn storeu_complex(ptr: *mut Complex32, val: __m512) {
        _mm512_storeu_ps(ptr.cast::<f32>(), val)
    }

    #[inline(always)]
    unsafe fn add(a: __m512, b: __m512) -> __m512 {
        _mm512_add_ps(a, b)
    }

    #[inline(always)]
    unsafe fn sub(a: __m512, b: __m512) -> __m512 {
        _mm512_sub_ps(a, b)
    }

    #[inline(always)]
    unsafe fn mul(a: __m512, b: __m512) -> __m512 {
        _mm512_mul_ps(a, b)
    }

    #[inline(always)]
    unsafe fn fmaddsub(a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmaddsub_ps(a, b, c)
    }

    #[inline(always)]
    unsafe fn permute_complex_swap(a: __m512) -> __m512 {
        _mm512_permute_ps(a, 0b1011_0001)
    }

    #[inline(always)]
    unsafe fn rotate_quarter_turn(v: __m512, sign: f32) -> __m512 {
        let swapped = _mm512_permute_ps(v, 0b1011_0001);
        let mask = if sign > 0.0 {
            _mm512_set_ps(
                0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0,
                -0.0,
            )
        } else {
            _mm512_set_ps(
                -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
                0.0,
            )
        };
        _mm512_xor_ps(swapped, mask)
    }

    #[inline(always)]
    unsafe fn stage_groups_one(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        twiddles: &[Complex32],
    ) {
        let half_n = radix;
        let vector_end = radix & !7usize;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        let twiddle_ptr = twiddles.as_ptr();

        let idx_a = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let idx_b = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

        let mut j = 0usize;
        while j < vector_end {
            let d0 = _mm512_castps_pd(_mm512_loadu_ps(src_ptr.add(j << 1).cast::<f32>()));
            let d1 = _mm512_castps_pd(_mm512_loadu_ps(src_ptr.add((j << 1) + 8).cast::<f32>()));

            let a = _mm512_castpd_ps(_mm512_permutex2var_pd(d0, idx_a, d1));
            let b = _mm512_castpd_ps(_mm512_permutex2var_pd(d0, idx_b, d1));

            let w = _mm512_loadu_ps(twiddle_ptr.add(j).cast::<f32>());
            let wr = _mm512_moveldup_ps(w);
            let wi = _mm512_movehdup_ps(w);

            let product = Self::cmul(wr, wi, b);

            let s = _mm512_add_ps(a, product);
            let t = _mm512_sub_ps(a, product);

            _mm512_storeu_ps(dst_ptr.add(j).cast::<f32>(), s);
            _mm512_storeu_ps(dst_ptr.add(half_n + j).cast::<f32>(), t);

            j += 8;
        }
        while j < radix {
            let a = src[j << 1];
            let b = src[(j << 1) + 1] * twiddles[j];
            dst[j] = a + b;
            dst[half_n + j] = a - b;
            j += 1;
        }
    }

    #[inline(always)]
    unsafe fn stage_pair_groups_two(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
    ) {
        let quarter_n = radix;
        let half_n = radix << 1;
        let vector_end = radix & !7usize;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        let first_ptr = first_twiddles.as_ptr();
        let second_ptr = second_twiddles.as_ptr();

        let idx_a = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let idx_b = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

        let mut j = 0usize;
        while j < vector_end {
            let d0 = _mm512_castps_pd(_mm512_loadu_ps(src_ptr.add(j * 4).cast::<f32>()));
            let d1 = _mm512_castps_pd(_mm512_loadu_ps(src_ptr.add((j + 2) * 4).cast::<f32>()));
            let d2 = _mm512_castps_pd(_mm512_loadu_ps(src_ptr.add((j + 4) * 4).cast::<f32>()));
            let d3 = _mm512_castps_pd(_mm512_loadu_ps(src_ptr.add((j + 6) * 4).cast::<f32>()));

            let t0 = _mm512_shuffle_f64x2::<0x88>(d0, d1);
            let t1 = _mm512_shuffle_f64x2::<0xdd>(d0, d1);
            let t2 = _mm512_shuffle_f64x2::<0x88>(d2, d3);
            let t3 = _mm512_shuffle_f64x2::<0xdd>(d2, d3);

            let x0 = _mm512_castpd_ps(_mm512_permutex2var_pd(t0, idx_a, t2));
            let x1 = _mm512_castpd_ps(_mm512_permutex2var_pd(t0, idx_b, t2));
            let raw_x2 = _mm512_castpd_ps(_mm512_permutex2var_pd(t1, idx_a, t3));
            let raw_x3 = _mm512_castpd_ps(_mm512_permutex2var_pd(t1, idx_b, t3));

            let w1 = _mm512_loadu_ps(first_ptr.add(j).cast::<f32>());
            let w1r = _mm512_moveldup_ps(w1);
            let w1i = _mm512_movehdup_ps(w1);

            let x2 = Self::cmul(w1r, w1i, raw_x2);
            let x3 = Self::cmul(w1r, w1i, raw_x3);

            let a0 = _mm512_add_ps(x0, x2);
            let a1 = _mm512_add_ps(x1, x3);
            let b0 = _mm512_sub_ps(x0, x2);
            let b1 = _mm512_sub_ps(x1, x3);

            let w2 = _mm512_loadu_ps(second_ptr.add(j).cast::<f32>());
            let w2r = _mm512_moveldup_ps(w2);
            let w2i = _mm512_movehdup_ps(w2);
            let c0 = Self::cmul(w2r, w2i, a1);

            let w3 = _mm512_loadu_ps(second_ptr.add(j + radix).cast::<f32>());
            let w3r = _mm512_moveldup_ps(w3);
            let w3i = _mm512_movehdup_ps(w3);
            let c1 = Self::cmul(w3r, w3i, b1);

            _mm512_storeu_ps(dst_ptr.add(j).cast::<f32>(), _mm512_add_ps(a0, c0));
            _mm512_storeu_ps(dst_ptr.add(j + half_n).cast::<f32>(), _mm512_sub_ps(a0, c0));
            _mm512_storeu_ps(
                dst_ptr.add(j + quarter_n).cast::<f32>(),
                _mm512_add_ps(b0, c1),
            );
            _mm512_storeu_ps(
                dst_ptr.add(j + half_n + quarter_n).cast::<f32>(),
                _mm512_sub_ps(b0, c1),
            );

            j += 8;
        }
        while j < radix {
            let x0 = src[j * 4];
            let x1 = src[j * 4 + 1];
            let x2 = src[j * 4 + 2] * first_twiddles[j];
            let x3 = src[j * 4 + 3] * first_twiddles[j];
            let a0 = x0 + x2;
            let a1 = x1 + x3;
            let b0 = x0 - x2;
            let b1 = x1 - x3;
            let c0 = a1 * second_twiddles[j];
            let c1 = b1 * second_twiddles[j + radix];
            dst[j] = a0 + c0;
            dst[j + half_n] = a0 - c0;
            dst[j + quarter_n] = b0 + c1;
            dst[j + half_n + quarter_n] = b0 - c1;
            j += 1;
        }
    }

    #[inline(always)]
    unsafe fn stage_triple_quarter_groups_one(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
    ) {
        crate::application::execution::kernel::components::stockham::butterfly::stage_triple_impl::<
            _,
            1024,
        >(
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
        );
    }

    #[inline(always)]
    unsafe fn stockham_quad_groups_eight_low_live(
        src: &[Complex32],
        dst: &mut [Complex32],
        radix: usize,
        first_twiddles: &[Complex32],
        second_twiddles: &[Complex32],
        third_twiddles: &[Complex32],
        fourth_twiddles: &[Complex32],
    ) {
        crate::application::execution::kernel::components::stockham::avx::f32::quad::stockham_quad_groups_eight32(
            src, dst, radix, first_twiddles, second_twiddles, third_twiddles, fourth_twiddles,
        )
    }
}
