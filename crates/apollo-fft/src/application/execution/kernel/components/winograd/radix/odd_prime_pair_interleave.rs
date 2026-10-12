//! AVX2 complex lane-interleave primitives (f32).
//!
//! The separated-stream stages move between interleaved complex slots in
//! `base` and packed re/im rows in the stream-major working set. These
//! helpers bracket that boundary: [`deinterleave8`] splits 8 contiguous
//! complex values into packed rows on load, [`store_interleaved8`]
//! interleaves packed rows back into 8 contiguous slots on store.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m256;

/// Cross-lane 64-bit permute (`vpermq`) on packed floats.
///
/// stdarch exposes this only as `_mm256_permute4x64_epi64`; the casts
/// are free lane reinterpretations.
///
/// # Safety
///
/// The caller must have established AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn permute_qwords<const IMM8: i32>(v: __m256) -> __m256 {
    use core::arch::x86_64::{_mm256_castps_si256, _mm256_castsi256_ps, _mm256_permute4x64_epi64};

    // Pure register shuffle; the `unsafe fn` marker carries the caller's
    // AVX2-establishment contract (executing `vpermq` without AVX2 is UB).
    _mm256_castsi256_ps(_mm256_permute4x64_epi64::<IMM8>(_mm256_castps_si256(v)))
}

/// Deinterleave 8 contiguous complex values into packed re/im rows.
///
/// # Safety
///
/// `slot` must point at 16 readable f32 lanes, and the caller must have
/// established AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn deinterleave8(slot: *const f32) -> (__m256, __m256) {
    use core::arch::x86_64::{
        _mm256_castps128_ps256, _mm256_castps256_ps128, _mm256_insertf128_ps, _mm256_loadu_ps,
        _mm256_shuffle_ps,
    };

    // SAFETY: AVX2 is established by the caller; the two loads together
    // read exactly the 16 lanes the caller guarantees, and every lane
    // below names typed in-register data.
    unsafe {
        let b0 = _mm256_loadu_ps(slot);
        let b1 = _mm256_loadu_ps(slot.add(8));
        // [r0,r1,i0,i1 | r2,r3,i2,i3]: even 64-bit granules hold re.
        let s0 = _mm256_shuffle_ps::<0xD8>(b0, b0);
        let s1 = _mm256_shuffle_ps::<0xD8>(b1, b1);
        // Harvest the re granules (0, 2) and im granules (1, 3); the low
        // 128 bits of each result hold one packed half.
        let re0 = permute_qwords::<0x08>(s0);
        let re1 = permute_qwords::<0x08>(s1);
        let im0 = permute_qwords::<0x0D>(s0);
        let im1 = permute_qwords::<0x0D>(s1);
        let re = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm256_castps256_ps128(re0)),
            _mm256_castps256_ps128(re1),
            1,
        );
        let im = _mm256_insertf128_ps(
            _mm256_castps128_ps256(_mm256_castps256_ps128(im0)),
            _mm256_castps256_ps128(im1),
            1,
        );
        (re, im)
    }
}

/// Store packed re/im rows as 8 interleaved complex values.
///
/// # Safety
///
/// `slot` must point at 16 writable f32 lanes, and the caller must have
/// established AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn store_interleaved8(slot: *mut f32, re: __m256, im: __m256) {
    use core::arch::x86_64::{
        _mm256_permute2f128_ps, _mm256_storeu_ps, _mm256_unpackhi_ps, _mm256_unpacklo_ps,
    };

    // SAFETY: AVX2 is established by the caller; the two stores together
    // write exactly the 16 lanes the caller guarantees.
    unsafe {
        let lo = _mm256_unpacklo_ps(re, im);
        let hi = _mm256_unpackhi_ps(re, im);
        _mm256_storeu_ps(slot, _mm256_permute2f128_ps::<0x20>(lo, hi));
        _mm256_storeu_ps(slot.add(8), _mm256_permute2f128_ps::<0x31>(lo, hi));
    }
}
