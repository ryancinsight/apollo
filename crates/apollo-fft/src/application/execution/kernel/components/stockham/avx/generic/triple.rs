use super::super::backend::StockhamAvxBackend;

mod n1024;
mod n128;
mod n256;
mod n32;
mod n32768;
mod n512;
mod n64;

pub(crate) use n1024::stage_triple_radix1_n1024_avx_fma;
pub(crate) use n128::stage_triple_radix1_n128_avx_fma;
pub(crate) use n256::stage_triple_radix1_n256_avx_fma;
pub(crate) use n32::stage_triple_radix1_n32_avx_fma;
pub(crate) use n32768::stage_triple_radix1_n32768_avx_fma;
pub(crate) use n512::stage_triple_radix1_n512_avx_fma;
pub(crate) use n64::stage_triple_radix1_n64_avx_fma;

#[inline]
unsafe fn radix1_triple_do_one<B: StockhamAvxBackend>(
    src_ptr: *const B::Complex,
    dst_ptr: *mut B::Complex,
    k: usize,
    eighth_n: usize,
    half_n: usize,
    quarter_n: usize,
    w2_quarter_turn_sign: B::Real,
    w3br: B::Vector,
    w3bi: B::Vector,
    w3dr: B::Vector,
    w3di: B::Vector,
    w3_quarter_turn_sign: B::Real,
) {
    let x0 = B::loadu_complex(src_ptr.add(k));
    let x2 = B::loadu_complex(src_ptr.add(2 * eighth_n + k));
    let x4 = B::loadu_complex(src_ptr.add(4 * eighth_n + k));
    let x6 = B::loadu_complex(src_ptr.add(6 * eighth_n + k));

    let s0 = B::add(x0, x4);
    let s2 = B::add(x2, x6);
    let d0 = B::sub(x0, x4);
    let d2 = B::sub(x2, x6);

    let u2 = B::rotate_quarter_turn(d2, w2_quarter_turn_sign);
    let p0 = B::add(s0, s2);
    let p2 = B::add(d0, u2);
    let p4 = B::sub(s0, s2);
    let p6 = B::sub(d0, u2);

    let x1 = B::loadu_complex(src_ptr.add(eighth_n + k));
    let x3 = B::loadu_complex(src_ptr.add(3 * eighth_n + k));
    let x5 = B::loadu_complex(src_ptr.add(5 * eighth_n + k));
    let x7 = B::loadu_complex(src_ptr.add(7 * eighth_n + k));

    let s1 = B::add(x1, x5);
    let s3 = B::add(x3, x7);
    let d1 = B::sub(x1, x5);
    let d3 = B::sub(x3, x7);
    let u3 = B::rotate_quarter_turn(d3, w2_quarter_turn_sign);
    let p1 = B::add(s1, s3);
    let p3 = B::add(d1, u3);
    let p5 = B::sub(s1, s3);
    let p7 = B::sub(d1, u3);

    let q2 = B::rotate_quarter_turn(p5, w3_quarter_turn_sign);
    B::storeu_complex(dst_ptr.add(k), B::add(p0, p1));
    B::storeu_complex(dst_ptr.add(half_n + k), B::sub(p0, p1));
    B::storeu_complex(dst_ptr.add(quarter_n + k), B::add(p4, q2));
    B::storeu_complex(dst_ptr.add(half_n + quarter_n + k), B::sub(p4, q2));

    let q1 = B::cmul(w3br, w3bi, p3);
    let q3 = B::cmul(w3dr, w3di, p7);
    B::storeu_complex(dst_ptr.add(eighth_n + k), B::add(p2, q1));
    B::storeu_complex(dst_ptr.add(half_n + eighth_n + k), B::sub(p2, q1));
    B::storeu_complex(dst_ptr.add(quarter_n + eighth_n + k), B::add(p6, q3));
    B::storeu_complex(
        dst_ptr.add(half_n + quarter_n + eighth_n + k),
        B::sub(p6, q3),
    );
}
