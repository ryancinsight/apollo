//! Separated-stream 11-point DFTs (f32).
//!
//! [`super::odd_prime_pair_simd`] vectorizes one 11-point DFT across its
//! five k lanes. N = 121 needs 22 such calls, and each call reloads the
//! pair tables, re-broadcasts every sum, and pays a full scalar
//! prologue/epilogue. The separated-stream form instead runs 11 DFTs
//! together: one scalar prologue per stream gathers sums/idiffs into a
//! stream-major working set, the k dots vectorize across 8 streams plus a
//! 3-lane scalar tail, and one scalar epilogue per stream scatters the
//! results. Table traffic drops 11x and the broadcasts become contiguous
//! vector loads.
//!
//! Each stream keeps the scalar m-ascending summation order; only
//! multiply-add rounding changes (one FMA rounding instead of two),
//! within the same f32 tolerance the single-call leaf documents.

use eunomia::Complex32;

use super::odd_prime_pair::PrimePairTable;
#[cfg(target_arch = "x86_64")]
use super::odd_prime_pair_interleave::{deinterleave8, store_interleaved8};

#[cfg(target_arch = "x86_64")]
mod rows;

/// Test hook running 11 strided 11-point DFTs through one AVX2/FMA dispatch.
///
/// Stream `s` reads `in_starts[s] + i * in_step` and writes
/// `out_starts[s] + i * out_step` for `i` in `0..11`; all 121 input slots
/// must be in bounds and pairwise disjoint, and likewise the outputs.
/// Returns `false` without touching `base` when the host lacks AVX2+FMA
/// (or off x86-64). Production callers go through the N = 121 chunked
/// driver, which probes once for the whole transform.
#[cfg(test)]
#[inline]
pub(crate) fn try_dft11_streams<const INVERSE: bool>(
    base: &mut [Complex32],
    in_starts: &[usize; 11],
    in_step: usize,
    out_starts: &[usize; 11],
    out_step: usize,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA were just probed on this thread; the caller
        // guarantees the 121 input and 121 output slots as documented.
        unsafe {
            dft11_streams_simd::<INVERSE>(base, in_starts, in_step, out_starts, out_step);
        }
        return true;
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (base, in_starts, in_step, out_starts, out_step);
    false
}

/// Pair tables in (m, k) orientation for stream-major dots: `cos_mk[m][k]`
/// is `cos_table[k][m]`, so the inner k loop walks one contiguous row.
struct PairTablesMk {
    cos_mk: [[f32; 5]; 5],
    sin_mk: [[f32; 5]; 5],
}

static PAIR11_MK: std::sync::LazyLock<PairTablesMk> = std::sync::LazyLock::new(|| {
    let cos = <f32 as PrimePairTable<11, 5>>::cos_table();
    let sin = <f32 as PrimePairTable<11, 5>>::sin_table();
    let mut t = PairTablesMk {
        cos_mk: [[0.0; 5]; 5],
        sin_mk: [[0.0; 5]; 5],
    };
    for m in 0..5 {
        for k in 0..5 {
            t.cos_mk[m][k] = cos[k][m];
            t.sin_mk[m][k] = sin[k][m];
        }
    }
    t
});

/// Stream-major working set for 11 pair-basis DFTs: `sums_re[m][s]` is
/// stream `s`, pair `m`. The driver runs prologue, k dots, epilogue; all
/// reads complete before any write, so differing input/output layouts are
/// safe in place.
#[cfg(target_arch = "x86_64")]
struct StreamsScratch {
    sums_re: [[f32; 11]; 5],
    sums_im: [[f32; 11]; 5],
    idiffs_re: [[f32; 11]; 5],
    idiffs_im: [[f32; 11]; 5],
    x0_re: [f32; 11],
    x0_im: [f32; 11],
    y0_re: [f32; 11],
    y0_im: [f32; 11],
    bre: [[f32; 8]; 5],
    bim: [[f32; 8]; 5],
    dre: [[f32; 8]; 5],
    dim: [[f32; 8]; 5],
    tail_bre: [[f32; 3]; 5],
    tail_bim: [[f32; 3]; 5],
    tail_dre: [[f32; 3]; 5],
    tail_dim: [[f32; 3]; 5],
}

/// True when the 11 starts run consecutively, so step-11 lanes across
/// streams 0..8 are contiguous and the vector stages apply.
#[cfg(target_arch = "x86_64")]
fn starts_consecutive(starts: &[usize; 11]) -> bool {
    let mut s = 1;
    while s < 11 {
        if starts[s] != starts[s - 1] + 1 {
            return false;
        }
        s += 1;
    }
    true
}

#[cfg(target_arch = "x86_64")]
impl StreamsScratch {
    fn zeroed() -> Self {
        Self {
            sums_re: [[0.0; 11]; 5],
            sums_im: [[0.0; 11]; 5],
            idiffs_re: [[0.0; 11]; 5],
            idiffs_im: [[0.0; 11]; 5],
            x0_re: [0.0; 11],
            x0_im: [0.0; 11],
            y0_re: [0.0; 11],
            y0_im: [0.0; 11],
            bre: [[0.0; 8]; 5],
            bim: [[0.0; 8]; 5],
            dre: [[0.0; 8]; 5],
            dim: [[0.0; 8]; 5],
            tail_bre: [[0.0; 3]; 5],
            tail_bim: [[0.0; 3]; 5],
            tail_dre: [[0.0; 3]; 5],
            tail_dim: [[0.0; 3]; 5],
        }
    }

    /// Scalar prologue over streams `S0..S1`, identical to
    /// `dft_pair_impl` at N = 11. Reads only; every write lands in the
    /// working set.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all 121 stream input slots in bounds.
    unsafe fn prologue_scalar<const INVERSE: bool, const S0: usize, const S1: usize>(
        &mut self,
        base: &[Complex32],
        in_starts: &[usize; 11],
        in_step: usize,
    ) {
        let sign = if INVERSE { 1.0f32 } else { -1.0f32 };
        // SAFETY: the caller guarantees all 121 stream slots in bounds,
        // and the (m + 1) / (10 - m) indices stay below 11.
        unsafe {
            for s in S0..S1 {
                let start = in_starts[s];
                let c0 = *base.get_unchecked(start);
                self.x0_re[s] = c0.re;
                self.x0_im[s] = c0.im;
                let mut acc_re = c0.re;
                let mut acc_im = c0.im;
                for m in 0..5 {
                    let a = *base.get_unchecked(start + (m + 1) * in_step);
                    let b = *base.get_unchecked(start + (10 - m) * in_step);
                    let sum_re = a.re + b.re;
                    let sum_im = a.im + b.im;
                    acc_re += sum_re;
                    acc_im += sum_im;
                    self.sums_re[m][s] = sum_re;
                    self.sums_im[m][s] = sum_im;
                    self.idiffs_re[m][s] = -(a.im - b.im) * sign;
                    self.idiffs_im[m][s] = (a.re - b.re) * sign;
                }
                self.y0_re[s] = acc_re;
                self.y0_im[s] = acc_im;
            }
        }
    }

    /// Vector prologue across streams 0..8 for column-major layouts.
    ///
    /// Consecutive starts with step 11 make the eight lanes at each input
    /// contiguous, so every lane vector loads as two complex blocks and
    /// deinterleaves into packed re/im rows. Per-lane arithmetic matches
    /// [`Self::prologue_scalar`] exactly (same add/sub order, exact sign
    /// flips), so both paths are bitwise identical; streams 8..11 run
    /// the scalar tail.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all 121 stream input slots in bounds,
    /// `in_step == 11`, and consecutive `in_starts`; `first` must equal
    /// `in_starts[0]`. AVX2 and FMA must be established.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn prologue_vec<const INVERSE: bool>(
        &mut self,
        base: &[Complex32],
        in_starts: &[usize; 11],
        first: usize,
    ) {
        use core::arch::x86_64::{
            _mm256_add_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
            _mm256_xor_ps,
        };

        // SAFETY: AVX2+FMA are established by the caller; the consecutive
        // step-11 precondition makes every `first + i * 11 + lane` below
        // name typed in-bounds slots, and the stores land in the working
        // set.
        unsafe {
            let f = base.as_ptr().cast::<f32>();
            let (x0re, x0im) = deinterleave8(f.add(first * 2));
            _mm256_storeu_ps(self.x0_re.as_mut_ptr(), x0re);
            _mm256_storeu_ps(self.x0_im.as_mut_ptr(), x0im);
            let mut acc_re = x0re;
            let mut acc_im = x0im;
            // Sign masks replicating `-(a - b) * sign` / `(a - b) * sign`
            // bit for bit: negation and scaling by exactly +-1 are pure
            // sign flips.
            let negate = _mm256_set1_ps(-0.0);
            let flip_re = if INVERSE { negate } else { _mm256_setzero_ps() };
            let flip_im = if INVERSE { _mm256_setzero_ps() } else { negate };
            for m in 0..5 {
                let (a_re, a_im) = deinterleave8(f.add((first + (m + 1) * 11) * 2));
                let (b_re, b_im) = deinterleave8(f.add((first + (10 - m) * 11) * 2));
                let sum_re = _mm256_add_ps(a_re, b_re);
                let sum_im = _mm256_add_ps(a_im, b_im);
                acc_re = _mm256_add_ps(acc_re, sum_re);
                acc_im = _mm256_add_ps(acc_im, sum_im);
                _mm256_storeu_ps(self.sums_re[m].as_mut_ptr(), sum_re);
                _mm256_storeu_ps(self.sums_im[m].as_mut_ptr(), sum_im);
                let d_re = _mm256_sub_ps(a_re, b_re);
                let d_im = _mm256_sub_ps(a_im, b_im);
                _mm256_storeu_ps(self.idiffs_re[m].as_mut_ptr(), _mm256_xor_ps(d_im, flip_re));
                _mm256_storeu_ps(self.idiffs_im[m].as_mut_ptr(), _mm256_xor_ps(d_re, flip_im));
            }
            _mm256_storeu_ps(self.y0_re.as_mut_ptr(), acc_re);
            _mm256_storeu_ps(self.y0_im.as_mut_ptr(), acc_im);
            // Scalar tail for streams 8..11.
            self.prologue_scalar::<INVERSE, 8, 11>(base, in_starts, 11);
        }
    }

    /// K dots across 8 streams plus a 3-lane scalar tail.
    ///
    /// # Safety
    ///
    /// The caller must have established AVX2 and FMA support.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn k_dots(&mut self) {
        use core::arch::x86_64::{
            _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
        };

        // SAFETY: AVX2+FMA are established by the caller; every load/store
        // below names typed in-bounds lanes of the working set, and the `u`
        // intrinsics permit unaligned addresses. The k-outer order keeps 10
        // live vector registers (4 accumulators, 4 stream loads, 2 table
        // broadcasts), spill-free on AVX2; each stream still sums m-ascending.
        unsafe {
            let t = &*PAIR11_MK;
            for k in 0..5 {
                let mut acc_bre = _mm256_setzero_ps();
                let mut acc_bim = _mm256_setzero_ps();
                let mut acc_dre = _mm256_setzero_ps();
                let mut acc_dim = _mm256_setzero_ps();
                for m in 0..5 {
                    // Contiguous 8-wide stream loads; the tail stays scalar.
                    let sre = _mm256_loadu_ps(self.sums_re[m].as_ptr());
                    let sim = _mm256_loadu_ps(self.sums_im[m].as_ptr());
                    let dre = _mm256_loadu_ps(self.idiffs_re[m].as_ptr());
                    let dim = _mm256_loadu_ps(self.idiffs_im[m].as_ptr());
                    let tc = _mm256_set1_ps(t.cos_mk[m][k]);
                    let ts = _mm256_set1_ps(t.sin_mk[m][k]);
                    acc_bre = _mm256_fmadd_ps(sre, tc, acc_bre);
                    acc_bim = _mm256_fmadd_ps(sim, tc, acc_bim);
                    acc_dre = _mm256_fmadd_ps(dre, ts, acc_dre);
                    acc_dim = _mm256_fmadd_ps(dim, ts, acc_dim);
                    let ck = t.cos_mk[m][k];
                    let sk = t.sin_mk[m][k];
                    for s in 0..3 {
                        self.tail_bre[k][s] += self.sums_re[m][8 + s] * ck;
                        self.tail_bim[k][s] += self.sums_im[m][8 + s] * ck;
                        self.tail_dre[k][s] += self.idiffs_re[m][8 + s] * sk;
                        self.tail_dim[k][s] += self.idiffs_im[m][8 + s] * sk;
                    }
                }
                _mm256_storeu_ps(self.bre[k].as_mut_ptr(), acc_bre);
                _mm256_storeu_ps(self.bim[k].as_mut_ptr(), acc_bim);
                _mm256_storeu_ps(self.dre[k].as_mut_ptr(), acc_dre);
                _mm256_storeu_ps(self.dim[k].as_mut_ptr(), acc_dim);
            }
        }
    }

    /// Scalar epilogue over streams `S0..S1`; the only writes to `base`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all 121 stream output slots in bounds.
    unsafe fn epilogue_scalar<const S0: usize, const S1: usize>(
        &self,
        base: &mut [Complex32],
        out_starts: &[usize; 11],
        out_step: usize,
    ) {
        // SAFETY: the caller guarantees all 121 output slots in bounds,
        // and the (k + 1) / (10 - k) indices stay below 11.
        unsafe {
            for s in S0..S1 {
                let start = out_starts[s];
                let c0_re = self.x0_re[s];
                let c0_im = self.x0_im[s];
                *base.get_unchecked_mut(start) = Complex32::new(self.y0_re[s], self.y0_im[s]);
                for k in 0..5 {
                    let (bre_k, bim_k, dre_k, dim_k) = if s < 8 {
                        (
                            self.bre[k][s],
                            self.bim[k][s],
                            self.dre[k][s],
                            self.dim[k][s],
                        )
                    } else {
                        (
                            self.tail_bre[k][s - 8],
                            self.tail_bim[k][s - 8],
                            self.tail_dre[k][s - 8],
                            self.tail_dim[k][s - 8],
                        )
                    };
                    let base_re = c0_re + bre_k;
                    let base_im = c0_im + bim_k;
                    *base.get_unchecked_mut(start + (k + 1) * out_step) =
                        Complex32::new(base_re + dre_k, base_im + dim_k);
                    *base.get_unchecked_mut(start + (10 - k) * out_step) =
                        Complex32::new(base_re - dre_k, base_im - dim_k);
                }
            }
        }
    }

    /// Vector epilogue across streams 0..8 for column-major layouts.
    ///
    /// Consecutive starts with step 11 make the eight lanes at each
    /// output contiguous, so packed re/im rows interleave straight into
    /// the output slots. Per-lane arithmetic matches
    /// [`Self::epilogue_scalar`] exactly (same add/sub order), so both
    /// paths are bitwise identical; streams 8..11 run the scalar tail.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all 121 stream output slots in bounds,
    /// `out_step == 11`, and consecutive `out_starts`; `first` must equal
    /// `out_starts[0]`. AVX2 and FMA must be established.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn epilogue_vec(&self, base: &mut [Complex32], out_starts: &[usize; 11], first: usize) {
        use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_sub_ps};

        // SAFETY: AVX2+FMA are established by the caller; the consecutive
        // step-11 precondition makes every `first + i * 11 + lane` below
        // name typed in-bounds slots.
        unsafe {
            let f = base.as_mut_ptr().cast::<f32>();
            let yre = _mm256_loadu_ps(self.y0_re.as_ptr());
            let yim = _mm256_loadu_ps(self.y0_im.as_ptr());
            store_interleaved8(f.add(first * 2), yre, yim);
            let x0re = _mm256_loadu_ps(self.x0_re.as_ptr());
            let x0im = _mm256_loadu_ps(self.x0_im.as_ptr());
            for k in 0..5 {
                let bre = _mm256_loadu_ps(self.bre[k].as_ptr());
                let bim = _mm256_loadu_ps(self.bim[k].as_ptr());
                let dre = _mm256_loadu_ps(self.dre[k].as_ptr());
                let dim = _mm256_loadu_ps(self.dim[k].as_ptr());
                let base_re = _mm256_add_ps(x0re, bre);
                let base_im = _mm256_add_ps(x0im, bim);
                let o1_re = _mm256_add_ps(base_re, dre);
                let o1_im = _mm256_add_ps(base_im, dim);
                let o2_re = _mm256_sub_ps(base_re, dre);
                let o2_im = _mm256_sub_ps(base_im, dim);
                store_interleaved8(f.add((first + (k + 1) * 11) * 2), o1_re, o1_im);
                store_interleaved8(f.add((first + (10 - k) * 11) * 2), o2_re, o2_im);
            }
            // Scalar tail for streams 8..11. The vector stores above are
            // complete, so reborrowing `base` for the tail is sequential,
            // not aliased.
            self.epilogue_scalar::<8, 11>(base, out_starts, 11);
        }
    }
}

/// 11 stream-major pair-basis DFTs with AVX2/FMA k dots.
///
/// Column layouts (step 11 with consecutive starts) run the vector
/// prologue/epilogue; row inputs (step 1) run the vector row prologue.
/// Other input layouts and non-column outputs keep the scalar stages. All
/// paths compute each lane identically, so the choice is invisible in
/// the output bits.
///
/// # Safety
///
/// The caller must have established AVX2 and FMA support and must
/// guarantee the 121 input and 121 output slots documented on
/// [`try_dft11_streams`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dft11_streams_simd<const INVERSE: bool>(
    base: &mut [Complex32],
    in_starts: &[usize; 11],
    in_step: usize,
    out_starts: &[usize; 11],
    out_step: usize,
) {
    // SAFETY: AVX2+FMA plus the 121-slot layouts are established by the
    // caller, which is exactly what each stage below requires; the vector
    // stages additionally require their checked step/starts preconditions.
    unsafe {
        let mut scratch = StreamsScratch::zeroed();
        if in_step == 11 && starts_consecutive(in_starts) {
            scratch.prologue_vec::<INVERSE>(base, in_starts, in_starts[0]);
        } else if in_step == 1 {
            scratch.prologue_rows::<INVERSE>(base, in_starts);
        } else {
            scratch.prologue_scalar::<INVERSE, 0, 11>(base, in_starts, in_step);
        }
        scratch.k_dots();
        if out_step == 11 && starts_consecutive(out_starts) {
            scratch.epilogue_vec(base, out_starts, out_starts[0]);
        } else {
            scratch.epilogue_scalar::<0, 11>(base, out_starts, out_step);
        }
    }
}
