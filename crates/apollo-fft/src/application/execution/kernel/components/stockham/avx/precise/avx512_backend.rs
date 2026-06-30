use super::super::backend::StockhamAvxBackend;
use eunomia::Complex64;
use std::arch::x86_64::{
    __m512d, _mm512_add_pd, _mm512_fmaddsub_pd, _mm512_loadu_pd, _mm512_mul_pd, _mm512_permute_pd,
    _mm512_set1_pd, _mm512_set_pd, _mm512_shuffle_f64x2, _mm512_storeu_pd, _mm512_sub_pd,
    _mm512_xor_pd,
};

#[derive(Copy, Clone)]
pub(crate) struct Avx512BackendPrecise;

impl StockhamAvxBackend for Avx512BackendPrecise {
    type Real = f64;
    type Complex = Complex64;
    type Vector = __m512d;

    const COMPLEX_PER_VECTOR: usize = 4;

    #[inline]
    unsafe fn unpack_complex(c: Complex64) -> (f64, f64) {
        (c.re, c.im)
    }

    #[inline]
    unsafe fn complex_mul(a: Complex64, b: Complex64) -> Complex64 {
        a * b
    }

    #[inline]
    unsafe fn complex_add(a: Complex64, b: Complex64) -> Complex64 {
        a + b
    }

    #[inline]
    unsafe fn complex_sub(a: Complex64, b: Complex64) -> Complex64 {
        a - b
    }

    #[inline]
    unsafe fn set1_real(val: f64) -> __m512d {
        _mm512_set1_pd(val)
    }

    #[inline]
    unsafe fn set1_imag(val: f64) -> __m512d {
        _mm512_set1_pd(val)
    }

    #[inline]
    unsafe fn loadu_complex(ptr: *const Complex64) -> __m512d {
        _mm512_loadu_pd(ptr.cast::<f64>())
    }

    #[inline]
    unsafe fn storeu_complex(ptr: *mut Complex64, val: __m512d) {
        _mm512_storeu_pd(ptr.cast::<f64>(), val)
    }

    #[inline]
    unsafe fn add(a: __m512d, b: __m512d) -> __m512d {
        _mm512_add_pd(a, b)
    }

    #[inline]
    unsafe fn sub(a: __m512d, b: __m512d) -> __m512d {
        _mm512_sub_pd(a, b)
    }

    #[inline]
    unsafe fn mul(a: __m512d, b: __m512d) -> __m512d {
        _mm512_mul_pd(a, b)
    }

    #[inline]
    unsafe fn fmaddsub(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmaddsub_pd(a, b, c)
    }

    #[inline]
    unsafe fn permute_complex_swap(a: __m512d) -> __m512d {
        _mm512_permute_pd(a, 0x55)
    }

    #[inline]
    unsafe fn rotate_quarter_turn(v: __m512d, sign: f64) -> __m512d {
        let swapped = _mm512_permute_pd(v, 0x55);
        let mask = if sign > 0.0 {
            _mm512_set_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0)
        } else {
            _mm512_set_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0)
        };
        _mm512_xor_pd(swapped, mask)
    }

    #[inline]
    unsafe fn stage_groups_one(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        twiddles: &[Complex64],
    ) {
        let half_n = radix;
        let vector_end = radix & !3usize;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        let twiddle_ptr = twiddles.as_ptr();

        let mut j = 0usize;
        while j < vector_end {
            let d01 = _mm512_loadu_pd(src_ptr.add(j << 1).cast::<f64>());
            let d23 = _mm512_loadu_pd(src_ptr.add((j << 1) + 4).cast::<f64>());

            let a = _mm512_shuffle_f64x2::<0x88>(d01, d23);
            let b = _mm512_shuffle_f64x2::<0xdd>(d01, d23);

            let w = _mm512_loadu_pd(twiddle_ptr.add(j).cast::<f64>());
            let wr = _mm512_permute_pd::<0x00>(w);
            let wi = _mm512_permute_pd::<0xFF>(w);

            let product = Self::cmul(wr, wi, b);

            let s = _mm512_add_pd(a, product);
            let t = _mm512_sub_pd(a, product);

            _mm512_storeu_pd(dst_ptr.add(j).cast::<f64>(), s);
            _mm512_storeu_pd(dst_ptr.add(half_n + j).cast::<f64>(), t);

            j += 4;
        }
        while j < radix {
            let a = src[j << 1];
            let b = src[(j << 1) + 1] * twiddles[j];
            dst[j] = a + b;
            dst[half_n + j] = a - b;
            j += 1;
        }
    }

    #[inline]
    unsafe fn stage_pair_groups_two(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
    ) {
        let quarter_n = radix;
        let half_n = radix << 1;
        let vector_end = radix & !3usize;
        let src_ptr = src.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        let first_ptr = first_twiddles.as_ptr();
        let second_ptr = second_twiddles.as_ptr();

        let mut j = 0usize;
        while j < vector_end {
            let d0 = _mm512_loadu_pd(src_ptr.add(j * 4).cast::<f64>());
            let d1 = _mm512_loadu_pd(src_ptr.add((j + 1) * 4).cast::<f64>());
            let d2 = _mm512_loadu_pd(src_ptr.add((j + 2) * 4).cast::<f64>());
            let d3 = _mm512_loadu_pd(src_ptr.add((j + 3) * 4).cast::<f64>());

            let t0 = _mm512_shuffle_f64x2::<0x44>(d0, d1);
            let t1 = _mm512_shuffle_f64x2::<0xee>(d0, d1);
            let t2 = _mm512_shuffle_f64x2::<0x44>(d2, d3);
            let t3 = _mm512_shuffle_f64x2::<0xee>(d2, d3);

            let x0 = _mm512_shuffle_f64x2::<0x88>(t0, t2);
            let x1 = _mm512_shuffle_f64x2::<0xdd>(t0, t2);
            let raw_x2 = _mm512_shuffle_f64x2::<0x88>(t1, t3);
            let raw_x3 = _mm512_shuffle_f64x2::<0xdd>(t1, t3);

            let w1 = _mm512_loadu_pd(first_ptr.add(j).cast::<f64>());
            let w1r = _mm512_permute_pd::<0x00>(w1);
            let w1i = _mm512_permute_pd::<0xFF>(w1);

            let x2 = Self::cmul(w1r, w1i, raw_x2);
            let x3 = Self::cmul(w1r, w1i, raw_x3);

            let a0 = _mm512_add_pd(x0, x2);
            let a1 = _mm512_add_pd(x1, x3);
            let b0 = _mm512_sub_pd(x0, x2);
            let b1 = _mm512_sub_pd(x1, x3);

            let w2 = _mm512_loadu_pd(second_ptr.add(j).cast::<f64>());
            let w2r = _mm512_permute_pd::<0x00>(w2);
            let w2i = _mm512_permute_pd::<0xFF>(w2);
            let c0 = Self::cmul(w2r, w2i, a1);

            let w3 = _mm512_loadu_pd(second_ptr.add(j + radix).cast::<f64>());
            let w3r = _mm512_permute_pd::<0x00>(w3);
            let w3i = _mm512_permute_pd::<0xFF>(w3);
            let c1 = Self::cmul(w3r, w3i, b1);

            _mm512_storeu_pd(dst_ptr.add(j).cast::<f64>(), _mm512_add_pd(a0, c0));
            _mm512_storeu_pd(dst_ptr.add(j + half_n).cast::<f64>(), _mm512_sub_pd(a0, c0));
            _mm512_storeu_pd(
                dst_ptr.add(j + quarter_n).cast::<f64>(),
                _mm512_add_pd(b0, c1),
            );
            _mm512_storeu_pd(
                dst_ptr.add(j + half_n + quarter_n).cast::<f64>(),
                _mm512_sub_pd(b0, c1),
            );

            j += 4;
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

    #[inline]
    unsafe fn stage_triple_quarter_groups_one(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
    ) {
        crate::application::execution::kernel::components::stockham::butterfly::stage_triple_impl::<
            _,
            512,
        >(
            src,
            dst,
            radix,
            first_twiddles,
            second_twiddles,
            third_twiddles,
        );
    }

    #[inline]
    unsafe fn stockham_quad_groups_eight_low_live(
        src: &[Complex64],
        dst: &mut [Complex64],
        radix: usize,
        first_twiddles: &[Complex64],
        second_twiddles: &[Complex64],
        third_twiddles: &[Complex64],
        fourth_twiddles: &[Complex64],
    ) {
        crate::application::execution::kernel::components::stockham::avx::precise::quad::stockham_quad_groups_eight_precise(
            src, dst, radix, first_twiddles, second_twiddles, third_twiddles, fourth_twiddles,
        )
    }
}
