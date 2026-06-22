//! Rader's Algorithm for prime-length FFTs.

pub(crate) mod bluestein;
pub(crate) mod convolution;
pub(crate) mod generator;
pub(crate) mod ordered;
pub(crate) mod static_rader;

use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use convolution::rader_convolve_inplace;
use convolution::rader_negacyclic_convolve_inplace;
use std::sync::Arc;


pub(crate) trait RaderConvolutionBackend {
    fn convolve<F, const INVERSE: bool>(
        data: &mut [F::Complex],
        n: usize,
        generator_inverse: usize,
    ) where
        F: MixedRadixScalar<Complex = num_complex::Complex<F>> + ShortWinogradScalar;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FullCyclic;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HalfCyclicWinograd;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Bluestein;

impl RaderConvolutionBackend for FullCyclic {
    #[inline]
    fn convolve<F, const INVERSE: bool>(data: &mut [F::Complex], n: usize, generator_inverse: usize)
    where
        F: MixedRadixScalar<Complex = num_complex::Complex<F>> + ShortWinogradScalar,
    {
        let kernel_spectrum = F::cached_rader_spectrum::<INVERSE>(n, generator_inverse);
        rader_convolve_inplace::<F>(data, kernel_spectrum.as_ref());
    }
}

impl RaderConvolutionBackend for HalfCyclicWinograd {
    #[inline]
    fn convolve<F, const INVERSE: bool>(data: &mut [F::Complex], n: usize, generator_inverse: usize)
    where
        F: MixedRadixScalar<Complex = num_complex::Complex<F>> + ShortWinogradScalar,
    {
        debug_assert_eq!(data.len() % 2, 0);
        let m = data.len() / 2;
        let (kernel_cyc, kernel_neg) =
            F::cached_rader_negacyclic_spectra::<INVERSE>(n, generator_inverse);
        let twiddles = F::cached_rader_neg_twiddles(m);

        rader_negacyclic_convolve_inplace::<F>(
            data,
            kernel_cyc.as_ref(),
            kernel_neg.as_ref(),
            twiddles.as_ref(),
        );
    }
}

impl RaderConvolutionBackend for Bluestein {
    #[inline]
    fn convolve<F, const INVERSE: bool>(data: &mut [F::Complex], n: usize, generator_inverse: usize)
    where
        F: MixedRadixScalar<Complex = num_complex::Complex<F>> + ShortWinogradScalar,
    {
        bluestein::rader_bluestein_convolve_inplace::<F, INVERSE>(data, n, generator_inverse);
    }
}

/// Rader's algorithm for prime N.
pub(crate) fn rader_fft<
    F: MixedRadixScalar<Complex = num_complex::Complex<F>> + ShortWinogradScalar,
    const INVERSE: bool,
>(
    data: &mut [F::Complex],
) {
    let n = data.len();
    debug_assert!(crate::application::execution::kernel::radix_shape::is_prime(n));

    if static_rader::try_static_rader::<F, INVERSE>(data, n) {
        return;
    }

    rader_runtime_impl::<F, INVERSE>(data, n);
}

#[cfg(any(test, debug_assertions, feature = "kernel-strategy-bench"))]
pub(crate) fn rader_fft_with_convolution_backend<
    F: MixedRadixScalar<Complex = num_complex::Complex<F>> + ShortWinogradScalar,
    const INVERSE: bool,
    B: RaderConvolutionBackend,
>(
    data: &mut [F::Complex],
) {
    let n = data.len();
    debug_assert!(crate::application::execution::kernel::radix_shape::is_prime(n));
    rader_runtime_impl_with_backend::<F, INVERSE, B>(data, n);
}

#[inline]
fn rader_runtime_impl<
    F: MixedRadixScalar<Complex = num_complex::Complex<F>> + ShortWinogradScalar,
    const INVERSE: bool,
>(
    data: &mut [F::Complex],
    n: usize,
) {
    if prefers_bluestein_for_rader::<F>(n) {
        rader_runtime_impl_with_backend::<F, INVERSE, Bluestein>(data, n);
    } else if prefers_half_cyclic_for_rader::<F>(n) {
        rader_runtime_impl_with_backend::<F, INVERSE, HalfCyclicWinograd>(data, n);
    } else {
        rader_runtime_impl_with_backend::<F, INVERSE, FullCyclic>(data, n);
    }
}

pub(crate) const BLUESTEIN_RADER_THRESHOLD: usize = 2048;

/// Returns true when the Rader convolution for length n should prefer the
/// Bluestein path. For f32 we bias to Bluestein + Stockham PoT for all runtime
/// primes (post-static): this routes through the most optimized f32 kernels
/// (AVX fused stockham) and the pooled TL bluestein/rader scratch, avoiding
/// variable composite/GT sub-dispatch in FullCyclic and reducing recursion risk.
#[inline]
pub(crate) fn prefers_bluestein_for_rader<F: MixedRadixScalar>(n: usize) -> bool {
    let m = n - 1;
    // f32: bias to Bluestein+Stockham for m>=128 + explicit 113/67 (md f32 rader worst 2-4x+ like 67/113/257/271; see gap
    // full f32 avx/pot sub with_scratch + dftN heap + pools for stack). Small f32 use FullCyclic (safe).
    // Targets md rader f32 ratios + mem (pool reuse in bluestein for kernel/convolve). Broadened from 256.
    (F::PREFER_BLUESTEIN_MID_RADER && (n == 113 || n == 67 || m >= 128))
        || m >= BLUESTEIN_RADER_THRESHOLD
        || !crate::application::execution::kernel::radix_shape::is_prime23_smooth(m)
}

#[inline]
pub(crate) fn prefers_half_cyclic_for_rader<F: MixedRadixScalar>(n: usize) -> bool {
    n > F::HALF_CYCLIC_RADER_THRESHOLD || F::HALF_CYCLIC_RADER_PRIMES.contains(&n)
}

#[inline(never)]
fn rader_runtime_impl_with_backend<
    F: MixedRadixScalar<Complex = num_complex::Complex<F>> + ShortWinogradScalar,
    const INVERSE: bool,
    B: RaderConvolutionBackend,
>(
    data: &mut [F::Complex],
    n: usize,
) {
    let (g, g_inv) = generator::primitive_root_and_inverse(n);

    let gather = cached_generator_order(n, g);

    let x0 = data[0];
    let l = n - 1;

    F::with_rader_padded_scratch(l, |padded| {
        let sum_x = gather_sum_slice::<F>(data, padded, &gather);
        B::convolve::<F, INVERSE>(padded, n, g_inv);
        data[0] = x0 + sum_x;
        scatter_slice::<F>(data, padded, x0, &gather);
    });
}

/// Optimized gather + sum: collects elements into `padded` while computing the sum.
///
/// The sum is computed over sequential `data[1..len+1]` for numerical consistency.
/// The permuted gather stores `data[gather[q]]` to `padded[q]`.
/// Both loops are vectorized with 4-way unrolling for better ILP.
#[inline]
fn gather_sum_slice<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &[F::Complex],
    padded: &mut [F::Complex],
    gather: &[usize],
) -> F::Complex {
    debug_assert!(padded.len() >= gather.len());
    debug_assert!(data.len() > gather.len());

    let len = gather.len();

    // Sequential sum over data[1..len+1] - maintains numerical consistency
    let len4 = (len / 4) * 4;
    let mut s0 = F::complex(0.0, 0.0);
    let mut s1 = F::complex(0.0, 0.0);
    let mut s2 = F::complex(0.0, 0.0);
    let mut s3 = F::complex(0.0, 0.0);
    let mut i = 0usize;
    while i < len4 {
        unsafe {
            s0 += *data.get_unchecked(1 + i);
            s1 += *data.get_unchecked(2 + i);
            s2 += *data.get_unchecked(3 + i);
            s3 += *data.get_unchecked(4 + i);
        }
        i += 4;
    }
    let mut sum_x = (s0 + s1) + (s2 + s3);
    while i < len {
        unsafe {
            sum_x += *data.get_unchecked(1 + i);
        }
        i += 1;
    }

    // Permuted gather: optimized 8-way unrolling via shared (ILP for larger m in rader, helps f32 rader md-worst like 67/271/113/257).
    crate::application::execution::kernel::components::butterflies::gather_unroll8(
        data, gather, padded,
    );
    sum_x
}

#[inline]
fn scatter_slice<F: MixedRadixScalar<Complex = num_complex::Complex<F>>>(
    data: &mut [F::Complex],
    padded: &[F::Complex],
    x0: F::Complex,
    generator_order: &[usize],
) {
    debug_assert!(padded.len() >= generator_order.len());
    debug_assert!(data.len() > generator_order.len());

    let len = generator_order.len();
    if len == 0 {
        return;
    }

    unsafe {
        *data.get_unchecked_mut(*generator_order.get_unchecked(0)) = x0 + *padded.get_unchecked(0);
    }

    let len4 = 1 + ((len - 1) / 4) * 4;
    let mut q = 1usize;
    while q < len4 {
        unsafe {
            *data.get_unchecked_mut(*generator_order.get_unchecked(len - q)) =
                x0 + *padded.get_unchecked(q);
            *data.get_unchecked_mut(*generator_order.get_unchecked(len - q - 1)) =
                x0 + *padded.get_unchecked(q + 1);
            *data.get_unchecked_mut(*generator_order.get_unchecked(len - q - 2)) =
                x0 + *padded.get_unchecked(q + 2);
            *data.get_unchecked_mut(*generator_order.get_unchecked(len - q - 3)) =
                x0 + *padded.get_unchecked(q + 3);
        }
        q += 4;
    }
    while q < len {
        unsafe {
            *data.get_unchecked_mut(*generator_order.get_unchecked(len - q)) =
                x0 + *padded.get_unchecked(q);
        }
        q += 1;
    }
}

/// Branchless inverse generator order lookup.
///
/// Returns `generator_order[0]` when `q == 0`, otherwise `generator_order[len - q]`.
/// Uses a branchless conditional selection to avoid pipeline bubbles from the `if`.
#[inline]
pub(crate) fn inverse_generator_order_at(generator_order: &[usize], q: usize) -> usize {
    debug_assert!(q < generator_order.len());
    // SAFETY: all callers pass q from a loop bounded by generator_order.len().
    unsafe {
        let len = generator_order.len();
        let idx_if_zero = 0usize;
        let idx_if_nonzero = len - q;
        // Branchless select: select idx_if_zero when q == 0, otherwise idx_if_nonzero.
        // This avoids the misprediction penalty of the conditional branch.
        let idx = if q == 0 { idx_if_zero } else { idx_if_nonzero };
        *generator_order.get_unchecked(idx)
    }
}

pub(crate) fn cached_generator_order(n: usize, g: usize) -> Arc<[usize]> {
    crate::application::execution::kernel::mixed_radix::caches::cached_rader_order(
        (n, g),
        |(n, g)| build_generator_order(n, g),
    )
}

fn build_generator_order(n: usize, g: usize) -> Vec<usize> {
    let l = n - 1;
    let mut order = Vec::with_capacity(l);
    let mut g_idx = 1usize;
    for _ in 0..l {
        order.push(g_idx);
        g_idx = (g_idx * g) % n;
    }
    order
}

#[cfg(test)]
mod const_consistency_tests {
    use super::generator::PRIMITIVE_ROOTS;
    use crate::application::execution::kernel::direct::dft_forward;
    use crate::application::execution::kernel::test_utils::max_abs_err_64;
    use num_complex::Complex64;

    fn signal(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|k| {
                let t = k as f64;
                Complex64::new((0.27 * t).sin(), (0.35 * t).cos())
            })
            .collect()
    }

    /// Run Rader forward+inverse roundtrip for a prime length and verify
    /// identity (unnormalized inverse: result = N·input).
    fn assert_rader_roundtrip(n: usize) {
        let input = signal(n);
        let mut data = input.clone();
        // Forward (unnormalized)
        super::rader_fft::<f64, false>(&mut data);
        // Inverse (unnormalized → result = N·input)
        super::rader_fft::<f64, true>(&mut data);
        for x in &mut data {
            *x /= n as f64;
        }
        let err = max_abs_err_64(&data, &input);
        assert!(
            err < 1.0e-10,
            "Rader roundtrip N={n} mismatch err={err:.2e}"
        );
    }

    /// Run Rader forward and verify matches direct DFT.
    fn assert_rader_forward_matches_direct(n: usize) {
        let input = signal(n);
        let expected = dft_forward(&input);
        let mut got = input.clone();
        super::rader_fft::<f64, false>(&mut got);
        let err = max_abs_err_64(&got, &expected);
        assert!(err < 1.0e-10, "Rader forward N={n} mismatch err={err:.2e}");
    }

    fn assert_rader_backend_forward_matches_direct<B: super::RaderConvolutionBackend>(
        n: usize,
        tolerance: f64,
    ) {
        let input = signal(n);
        let expected = dft_forward(&input);
        let mut got = input;
        super::rader_fft_with_convolution_backend::<f64, false, B>(&mut got);
        let err = max_abs_err_64(&got, &expected);
        assert!(
            err < tolerance,
            "Rader ZST backend forward N={n} mismatch err={err:.2e}"
        );
    }

    fn assert_rader_strategies_match(n: usize, tolerance: f64) {
        let input = signal(n);
        let mut full = input.clone();
        let mut half = input;
        super::rader_fft_with_convolution_backend::<f64, false, super::FullCyclic>(&mut full);
        super::rader_fft_with_convolution_backend::<f64, false, super::HalfCyclicWinograd>(
            &mut half,
        );
        let err = max_abs_err_64(&half, &full);
        assert!(
            err < tolerance,
            "Rader half-cyclic/full-cyclic N={n} mismatch err={err:.2e}"
        );
    }

    // ── Static Rader value-semantic tests ─────────────────────────────────

    #[test]
    fn static_rader_5_forward_matches_direct() {
        assert_rader_forward_matches_direct(5);
    }

    #[test]
    fn static_rader_7_forward_matches_direct() {
        assert_rader_forward_matches_direct(7);
    }

    #[test]
    fn static_rader_11_forward_matches_direct() {
        assert_rader_forward_matches_direct(11);
    }

    #[test]
    fn static_rader_13_forward_matches_direct() {
        assert_rader_forward_matches_direct(13);
    }

    #[test]
    fn static_rader_17_forward_matches_direct() {
        assert_rader_forward_matches_direct(17);
    }

    #[test]
    fn static_rader_19_forward_matches_direct() {
        assert_rader_forward_matches_direct(19);
    }

    #[test]
    fn static_rader_23_forward_matches_direct() {
        assert_rader_forward_matches_direct(23);
    }

    #[test]
    fn static_rader_31_forward_matches_direct() {
        assert_rader_forward_matches_direct(31);
    }

    #[test]
    fn static_rader_13_roundtrip() {
        assert_rader_roundtrip(13);
    }

    #[test]
    fn static_rader_17_roundtrip() {
        assert_rader_roundtrip(17);
    }

    #[test]
    fn static_rader_23_roundtrip() {
        assert_rader_roundtrip(23);
    }

    // ── Runtime Rader value-semantic tests ────────────────────────────────

    #[test]
    fn runtime_rader_29_forward_matches_direct() {
        assert_rader_forward_matches_direct(29);
    }

    #[test]
    fn runtime_rader_37_forward_matches_direct() {
        assert_rader_forward_matches_direct(37);
    }

    #[test]
    fn runtime_rader_41_forward_matches_direct() {
        assert_rader_forward_matches_direct(41);
    }

    #[test]
    fn runtime_rader_29_roundtrip() {
        assert_rader_roundtrip(29);
    }

    #[test]
    fn runtime_rader_37_roundtrip() {
        assert_rader_roundtrip(37);
    }

    #[test]
    fn runtime_rader_auto_521_forward_matches_direct() {
        assert_rader_forward_matches_direct(521);
    }

    #[test]
    fn runtime_rader_half_cyclic_521_forward_matches_direct() {
        // N=521: HalfCyclicWinograd allocates [Complex64; 520] on stack in debug mode.
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                assert_rader_backend_forward_matches_direct::<super::HalfCyclicWinograd>(
                    521, 1.0e-8,
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn runtime_rader_zst_backends_forward_match_direct() {
        assert_rader_backend_forward_matches_direct::<super::FullCyclic>(29, 1.0e-10);
        assert_rader_backend_forward_matches_direct::<super::Bluestein>(67, 1.0e-10);
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                assert_rader_backend_forward_matches_direct::<super::HalfCyclicWinograd>(
                    521, 1.0e-8,
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn rader_bluestein_policy_is_scalar_trait_driven() {
        assert!(super::prefers_bluestein_for_rader::<f32>(67));
        assert!(super::prefers_bluestein_for_rader::<f32>(193));
        assert!(super::prefers_bluestein_for_rader::<f32>(257));
        assert!(!super::prefers_bluestein_for_rader::<f64>(193));
        assert!(!super::prefers_bluestein_for_rader::<f64>(257));
    }

    #[test]
    fn runtime_rader_half_cyclic_521_matches_full_cyclic() {
        // N=521: large stack arrays; spawn on 8MB thread.
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| assert_rader_strategies_match(521, 1.0e-9))
            .unwrap()
            .join()
            .unwrap();
    }

    // ── Const consistency tests ───────────────────────────────────────────

    #[test]
    fn rader_convolution_backends_are_zero_sized() {
        assert_eq!(std::mem::size_of::<super::FullCyclic>(), 0);
        assert_eq!(std::mem::size_of::<super::HalfCyclicWinograd>(), 0);
        assert_eq!(std::mem::size_of::<super::Bluestein>(), 0);
    }

    /// Every entry in [`PRIMITIVE_ROOTS`] must agree with the dynamic
    /// general-purpose primitive-root computation.
    #[test]
    fn primitive_roots_table_matches_dynamic() {
        for &(prime, expected_generator) in PRIMITIVE_ROOTS {
            let got = super::generator::primitive_root(prime);
            assert_eq!(
                got, expected_generator,
                "PRIMITIVE_ROOTS[{prime}] = {expected_generator}, but primitive_root() returned {got}"
            );
        }
    }

    #[test]
    fn inverse_generator_order_matches_modular_inverse_powers() {
        for &(prime, generator) in PRIMITIVE_ROOTS {
            let order = super::build_generator_order(prime, generator);
            let generator_inverse = super::generator::inverse_mod(generator, prime);
            let mut inverse_power = 1usize;

            for q in 0..prime - 1 {
                assert_eq!(
                    super::inverse_generator_order_at(&order, q),
                    inverse_power,
                    "derived inverse order mismatch for prime={prime}, q={q}"
                );
                inverse_power = (inverse_power * generator_inverse) % prime;
            }
        }
    }

    /// [`STATIC_RADER_PRIMES`] must be a subset of [`PRIMITIVE_ROOTS`] primes.
    #[test]
    fn static_rader_primes_are_in_primitive_roots() {
        let known: Vec<usize> = PRIMITIVE_ROOTS.iter().map(|&(p, _)| p).collect();
        for &prime in super::static_rader::STATIC_RADER_PRIMES {
            assert!(
                known.contains(&prime),
                "STATIC_RADER_PRIMES contains {prime} which is not in PRIMITIVE_ROOTS"
            );
        }
    }
}
