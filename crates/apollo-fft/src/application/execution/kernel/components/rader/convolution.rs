use crate::application::execution::kernel::components::butterflies;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;

/// Winograd-pair fused forward+pointwise dispatch.
///
/// For supported Winograd pair primes (N = 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53),
/// calls `dft_pair_forward_with_pointwise` which fuses the forward Winograd DFT with
/// kernel-spectrum multiplication, eliminating the separate `pointwise_mul` pass.
///
/// Expands to a `match` expression evaluating to `bool`: `true` if the size matched
/// and the fused transform was applied, `false` otherwise.
///
/// Each pair is specified as `($n:literal, $h:literal)` — $n is the prime and $h is
/// the half-size `(n-1)/2`.
macro_rules! try_winograd_pair_forward_with_pointwise {
    ($F:ty, $data:expr, $kernel_spectrum:expr) => {{
        with_winograd_pair_primes!(try_winograd_pair_forward_with_pointwise!(
            $F,
            $data,
            $kernel_spectrum
        ))
    }};
    ($F:ty, $data:expr, $kernel_spectrum:expr, $(($n:literal, $h:literal)),+ $(,)?) => {{
        use crate::application::execution::kernel::components::winograd::radix::odd_prime_pair::{
            dft_pair_forward_with_pointwise, PrimePairTable,
        };
        match $data.len() {
            $(
                $n => {
                    let arr: &mut [<$F as MixedRadixScalar>::Complex; $n] = $data.try_into().unwrap();
                    let ks: &[<$F as MixedRadixScalar>::Complex; $n] = $kernel_spectrum.try_into().unwrap();
                    dft_pair_forward_with_pointwise::<$F, $n, $h>(
                        arr,
                        ks,
                        <$F as PrimePairTable<$n, $h>>::cos_table(),
                        <$F as PrimePairTable<$n, $h>>::sin_table(),
                    );
                    true
                }
            )+
            _ => false,
        }
    }};
}

macro_rules! with_winograd_pair_primes {
    (try_winograd_pair_forward_with_pointwise!($F:ty, $data:expr, $kernel_spectrum:expr)) => {{
        try_winograd_pair_forward_with_pointwise!(
            $F,
            $data,
            $kernel_spectrum,
            (11, 5),
            (13, 6),
            (17, 8),
            (19, 9),
            (23, 11),
            (29, 14),
            (31, 15),
            (37, 18),
            (41, 20),
            (43, 21),
            (47, 23),
            (53, 26),
        )
    }};
}

/// Minimum Rader convolution length `M = N - 1` for the half-cyclic CRT split.
///
/// The Liu-Tolimieri half-cyclic strategy factors
/// `x^(2m)-1 = (x^m-1)(x^m+1)`, computes one cyclic and one negacyclic
/// length-`m` convolution, then recombines the two residue classes. Below this
/// threshold, the split/twist/recombine passes cost more than the saved FFT
/// length in the current CPU kernel family.
pub(crate) const HALF_CYCLIC_THRESHOLD: usize = 1024;

/// In-place circular convolution via forward FFT -> pointwise multiply -> inverse FFT.
///
/// `padded` holds the input sequence on entry and the convolution result on exit.
/// `kernel_spectrum` is the precomputed direction-specific DFT of the convolution kernel.
///
/// ## Dispatch order
///
/// 1. **Winograd pair fused** — for N = 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
///    `dft_pair_forward_with_pointwise` fuses the forward Winograd DFT with
///    kernel-spectrum multiplication, eliminating the separate `pointwise_mul` pass.
/// 2. **Short Winograd** — const-generic codelet for other N ≤ 128 (small composites
///    like 6, 10, 64, 128, …) with `pointwise_mul` as a separate pass.
/// 3. **Prime-23 composite** — `composite_forward_with_pointwise` fuses the
///    kernel-spectrum multiply into the forward radix-composite stage.
/// 4. **Coprime PFA** — Good-Thomas factorisation with separate pointwise-mul.
/// 5. **Trampoline fallback** — prime sub-convolution length that re-enters the
///    full dispatch chain (may recursively call Rader).
///
/// Marked `#[inline(never)]` to bound stack depth: the forward/inverse dispatch
/// may recursively re-enter Rader for prime sub-lengths, and preventing inlining
/// keeps each Rader level's stack frame independent.
#[inline(never)]
pub(super) fn rader_convolve_inplace<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    padded: &mut [F::Complex],
    kernel_spectrum: &[F::Complex],
) {
    let len = padded.len();

    // Fast-path: Winograd pair primes with fused forward+pointwise.
    // Covers N = 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53.
    // The forward Winograd DFT is fused with kernel-spectrum multiplication,
    // eliminating the separate pointwise_mul pass.
    if try_winograd_pair_forward_with_pointwise!(F, padded, kernel_spectrum) {
        F::short_winograd::<true, true>(padded); // inverse + 1/N normalize
        return;
    }

    // Fast-path: const-generic short Winograd codelet (other N ≤ 128).
    // Covers small composites (6, 10, 12, 14, 64, 128, …) with zero dispatch-chain overhead.
    if F::short_winograd::<false, false>(padded) {
        F::pointwise_mul(padded, kernel_spectrum);
        F::short_winograd::<true, true>(padded); // inverse + 1/N normalize
        return;
    }

    if let Some(radices) =
        crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices(len)
    {
        F::composite_forward_with_pointwise(padded, &radices, kernel_spectrum);
        F::composite_inverse(padded, &radices);
    } else if let Some((n1, n2)) =
        crate::application::execution::kernel::mixed_radix::caches::cached_coprime_factors(len)
    {
        crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, false>(
            padded, n1, n2,
        );
        F::pointwise_mul(padded, kernel_spectrum);
        crate::application::execution::kernel::components::good_thomas::pfa_fft::<F, true>(
            padded, n1, n2,
        );
        F::normalize(padded, len);
    } else {
        // Trampoline: recursive dispatch for prime sub-convolution lengths
        // that are too large for short Winograd and not prime23-composite.
        rader_subconv_forward_inplace::<F>(padded);
        F::pointwise_mul(padded, kernel_spectrum);
        rader_subconv_inverse_inplace::<F>(padded);
    }
}

/// In-place circular convolution via half-cyclic Winograd/Nussbaumer CRT split.
///
/// Splits the length-M cyclic convolution (M = padded.len()) into two length-m
/// convolutions (m = M/2): one cyclic and one negacyclic, then recombines via CRT.
///
/// Marked `#[inline(never)]` to bound stack depth: the sub-convolution forward/
/// inverse dispatch may recursively re-enter Rader for prime sub-lengths, and
/// preventing inlining keeps each Rader level's stack frame independent.
#[inline(never)]
pub(super) fn rader_negacyclic_convolve_inplace<
    F: MixedRadixScalar<Complex = num_complex::Complex<F>>,
>(
    padded: &mut [F::Complex],
    kernel_cyc_spectrum: &[F::Complex],
    kernel_neg_spectrum: &[F::Complex],
    twiddles: &[F::Complex],
) {
    let m = padded.len() / 2;
    debug_assert_eq!(padded.len(), 2 * m);
    debug_assert_eq!(kernel_cyc_spectrum.len(), m);
    debug_assert_eq!(kernel_neg_spectrum.len(), m);
    debug_assert_eq!(twiddles.len(), m);

    let (first, second) = padded.split_at_mut(m);

    // Nussbaumer split: reduce input modulo (x^m - 1) and (x^m + 1).
    // The negacyclic half is twisted here, eliminating a separate full pass.
    let mut j = 0usize;
    let m4 = (m / 4) * 4;
    while j < m4 {
        let a0 = first[j];
        let b0 = second[j];
        let a1 = first[j + 1];
        let b1 = second[j + 1];
        let a2 = first[j + 2];
        let b2 = second[j + 2];
        let a3 = first[j + 3];
        let b3 = second[j + 3];
        let w0 = twiddles[j];
        let w1 = twiddles[j + 1];
        let w2 = twiddles[j + 2];
        let w3 = twiddles[j + 3];
        first[j] = a0 + b0;
        second[j] = (a0 - b0) * w0;
        first[j + 1] = a1 + b1;
        second[j + 1] = (a1 - b1) * w1;
        first[j + 2] = a2 + b2;
        second[j + 2] = (a2 - b2) * w2;
        first[j + 3] = a3 + b3;
        second[j + 3] = (a3 - b3) * w3;
        j += 4;
    }
    while j < m {
        let a = first[j];
        let b = second[j];
        let w = twiddles[j];
        first[j] = a + b;
        second[j] = (a - b) * w;
        j += 1;
    }

    // --- Cyclic convolution of length m (modulo x^m - 1) ---
    // Dispatch order: Winograd pair (fused) → short Winograd → prime23 composite (fused) → trampoline.
    if !try_winograd_pair_forward_with_pointwise!(F, first, kernel_cyc_spectrum) {
        if F::short_winograd::<false, false>(first) {
            F::pointwise_mul(first, kernel_cyc_spectrum);
        } else if let Some(radices) =
            crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices(
                first.len(),
            )
        {
            F::composite_forward_with_pointwise(first, &radices, kernel_cyc_spectrum);
        } else {
            rader_subconv_forward_inplace::<F>(first);
            F::pointwise_mul(first, kernel_cyc_spectrum);
        }
    }
    // Inverse: short Winograd (normalized) → trampoline full dispatch.
    if !F::short_winograd::<true, true>(first) {
        rader_subconv_inverse_inplace::<F>(first);
    }

    // --- Negacyclic convolution of length m (modulo x^m + 1) ---
    // Dispatch order: Winograd pair (fused) → short Winograd → prime23 composite (fused) → trampoline.
    if !try_winograd_pair_forward_with_pointwise!(F, second, kernel_neg_spectrum) {
        if F::short_winograd::<false, false>(second) {
            F::pointwise_mul(second, kernel_neg_spectrum);
        } else if let Some(radices) =
            crate::application::execution::kernel::mixed_radix::caches::cached_prime23_radices(
                second.len(),
            )
        {
            F::composite_forward_with_pointwise(second, &radices, kernel_neg_spectrum);
        } else {
            rader_subconv_forward_inplace::<F>(second);
            F::pointwise_mul(second, kernel_neg_spectrum);
        }
    }
    // Inverse: short Winograd (normalized) → trampoline full dispatch.
    if !F::short_winograd::<true, true>(second) {
        rader_subconv_inverse_inplace::<F>(second);
    }

    // --- CRT recombination ---
    // The negacyclic half is untwisted here, eliminating a separate full pass.
    let half = F::complex(0.5, 0.0);
    let mut j = 0usize;
    let m4 = (m / 4) * 4;
    while j < m4 {
        let c0 = first[j];
        let n0 = butterflies::mul_conj::<F>(second[j], twiddles[j]);
        let c1 = first[j + 1];
        let n1 = butterflies::mul_conj::<F>(second[j + 1], twiddles[j + 1]);
        let c2 = first[j + 2];
        let n2 = butterflies::mul_conj::<F>(second[j + 2], twiddles[j + 2]);
        let c3 = first[j + 3];
        let n3 = butterflies::mul_conj::<F>(second[j + 3], twiddles[j + 3]);
        first[j] = (c0 + n0) * half;
        second[j] = (c0 - n0) * half;
        first[j + 1] = (c1 + n1) * half;
        second[j + 1] = (c1 - n1) * half;
        first[j + 2] = (c2 + n2) * half;
        second[j + 2] = (c2 - n2) * half;
        first[j + 3] = (c3 + n3) * half;
        second[j + 3] = (c3 - n3) * half;
        j += 4;
    }
    while j < m {
        let c = first[j];
        let n = butterflies::mul_conj::<F>(second[j], twiddles[j]);
        first[j] = (c + n) * half;
        second[j] = (c - n) * half;
        j += 1;
    }
}

/// Trampoline: keep recursive sub-convolution dispatch out of this frame.
///
/// When the sub-convolution length is prime, `forward_inplace` re-enters the
/// full dispatch chain which may call back into Rader. `inline(never)` prevents
/// debug builds from accumulating a large monomorphized stack frame.
#[inline(never)]
fn rader_subconv_forward_inplace<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &mut [F::Complex],
) {
    crate::application::execution::kernel::mixed_radix::forward_inplace::<F>(data);
}

/// Trampoline: keep recursive sub-convolution dispatch out of this frame.
#[inline(never)]
fn rader_subconv_inverse_inplace<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &mut [F::Complex],
) {
    crate::application::execution::kernel::mixed_radix::inverse_inplace::<F>(data);
}
