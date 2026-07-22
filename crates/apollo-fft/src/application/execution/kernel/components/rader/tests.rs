use super::generator::PRIMITIVE_ROOTS;
use crate::application::execution::kernel::direct::dft_forward;
use crate::application::execution::kernel::mixed_radix::scalar::MixedRadixScalar;
use crate::application::execution::kernel::test_utils::max_abs_err_64;
use eunomia::Complex64;

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
    super::rader_fft_with_convolution_backend::<f64, false, super::HalfCyclicWinograd>(&mut half);
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
fn runtime_rader_caches_match_canonical_generator() {
    const PRIME: usize = 43;
    let generator = super::generator::primitive_root_and_inverse(PRIME);
    let expected_order = super::build_generator_order(PRIME, generator.root());
    let cached_order = super::cached_generator_order(PRIME);
    assert_eq!(cached_order.as_ref(), expected_order.as_slice());

    let mut inverse_power = 1usize;
    let kernel: Vec<_> = (0..PRIME - 1)
        .map(|_| {
            let angle = -std::f64::consts::TAU * (inverse_power as f64) / (PRIME as f64);
            inverse_power = (inverse_power * generator.inverse()) % PRIME;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect();
    let expected_spectrum = dft_forward(&kernel);
    let cached_spectrum = <f64 as MixedRadixScalar>::cached_rader_spectrum::<false>(PRIME);
    let error = max_abs_err_64(cached_spectrum.as_ref(), &expected_spectrum);
    assert!(
        error < 1.0e-10,
        "canonical Rader spectrum mismatch, error={error:.2e}"
    );
}

#[test]
fn runtime_rader_auto_521_forward_matches_direct() {
    assert_rader_forward_matches_direct(521);
}

#[test]
fn runtime_rader_half_cyclic_521_forward_matches_direct() {
    assert_rader_backend_forward_matches_direct::<super::HalfCyclicWinograd>(521, 1.0e-8);
}

#[test]
fn runtime_rader_zst_backends_forward_match_direct() {
    assert_rader_backend_forward_matches_direct::<super::FullCyclic>(29, 1.0e-10);
    assert_rader_backend_forward_matches_direct::<super::Bluestein>(67, 1.0e-10);
    assert_rader_backend_forward_matches_direct::<super::HalfCyclicWinograd>(521, 1.0e-8);
}

#[test]
fn cache_initialization_fits_standard_stack() {
    const STACK_BUDGET_BYTES: usize = 2 * 1024 * 1024;

    std::thread::Builder::new()
        .stack_size(STACK_BUDGET_BYTES)
        .spawn(|| {
            assert_rader_backend_forward_matches_direct::<super::Bluestein>(67, 1.0e-10);
            assert_rader_backend_forward_matches_direct::<super::HalfCyclicWinograd>(521, 1.0e-8);
        })
        .expect("regression thread must spawn")
        .join()
        .expect("cache initialization must fit the bounded stack");
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
    assert_rader_strategies_match(521, 1.0e-9);
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
