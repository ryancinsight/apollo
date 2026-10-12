use super::storage::{
    extend_storage_from_owner_values, owner_values_from_storage, write_storage_from_owner_values,
    STORAGE_PAR_LEN_THRESHOLD,
};
use super::*;
use apollo_fft::{fft_1d_complex_inplace, PrecisionProfile};
use eunomia::assert_abs_diff_eq;
use leto::Array1;
use proptest::prelude::*;

fn exactly_sparse_signal() -> Vec<Complex64> {
    let plan = SparseFftPlan::new(8, 2).expect("plan");
    let mut spectrum = SparseSpectrum::new(8);
    spectrum
        .insert(1, Complex64::new(3.0, -1.0))
        .expect("insert");
    spectrum
        .insert(5, Complex64::new(-0.5, 2.0))
        .expect("insert");
    plan.inverse(&spectrum).expect("inverse")
}

#[test]
fn moirai_storage_conversion_helpers_match_serial_formulas_at_threshold() {
    let values: Vec<Complex32> = (0..STORAGE_PAR_LEN_THRESHOLD)
        .map(|index| {
            Complex32::new(
                (index as f32 * 0.03125).sin(),
                (index as f32 * 0.015625).cos(),
            )
        })
        .collect();

    let owner_values = owner_values_from_storage(&values);
    assert_eq!(owner_values.len(), values.len());
    for (actual, expected) in owner_values.iter().zip(values.iter()) {
        assert_eq!(actual.re.to_bits(), f64::from(expected.re).to_bits());
        assert_eq!(actual.im.to_bits(), f64::from(expected.im).to_bits());
    }

    let mut converted = Vec::new();
    extend_storage_from_owner_values::<Complex32>(&mut converted, &owner_values);
    assert_eq!(converted.len(), values.len());
    for (actual, expected) in converted.iter().zip(values.iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }

    let mut output = vec![Complex32::new(0.0, 0.0); values.len()];
    write_storage_from_owner_values(&mut output, &owner_values);
    for (actual, expected) in output.iter().zip(values.iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }
}

#[test]
fn typed_paths_support_complex64_complex32_and_mixed_f16_storage() {
    let plan = SparseFftPlan::new(8, 2).expect("plan");
    let signal64 = exactly_sparse_signal();
    let expected = plan.forward(&signal64).expect("forward");

    let mut frequencies64 = Vec::new();
    let mut values64 = Vec::new();
    plan.forward_typed_into(
        &signal64,
        &mut frequencies64,
        &mut values64,
        PrecisionProfile::HIGH_ACCURACY_F64,
    )
    .expect("typed complex64 forward");
    assert_eq!(frequencies64, expected.frequencies);
    for (actual, expected) in values64.iter().zip(expected.values.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }

    let signal32: Vec<Complex32> = signal64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect();
    let represented32: Vec<Complex64> = signal32.iter().copied().map(CpuStorage::to_cpu).collect();
    let expected32 = plan
        .forward(&represented32)
        .expect("represented f32 forward");
    let mut frequencies32 = Vec::new();
    let mut values32 = Vec::new();
    plan.forward_typed_into(
        &signal32,
        &mut frequencies32,
        &mut values32,
        PrecisionProfile::F32_STORAGE_F64_COMPUTE,
    )
    .expect("typed complex32 forward");
    assert_eq!(frequencies32, expected32.frequencies);
    for (actual, expected) in values32.iter().zip(expected32.values.iter()) {
        assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
        assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
    }

    let mut recovered32 = vec![Complex32::new(0.0, 0.0); plan.len()];
    plan.inverse_typed_into(
        &frequencies32,
        &values32,
        &mut recovered32,
        PrecisionProfile::F32_STORAGE_F64_COMPUTE,
    )
    .expect("typed complex32 inverse");
    for (actual, expected) in recovered32.iter().zip(signal32.iter()) {
        assert!((actual.re - expected.re).abs() < 1.0e-5);
        assert!((actual.im - expected.im).abs() < 1.0e-5);
    }

    let signal16: Vec<[F16; 2]> = signal64
        .iter()
        .map(|value| {
            [
                F16::from_f32(value.re as f32),
                F16::from_f32(value.im as f32),
            ]
        })
        .collect();
    let represented16: Vec<Complex64> = signal16.iter().copied().map(CpuStorage::to_cpu).collect();
    let expected16 = plan
        .forward(&represented16)
        .expect("represented F16 forward");
    let mut frequencies16 = Vec::new();
    let mut values16 = Vec::new();
    plan.forward_typed_into(
        &signal16,
        &mut frequencies16,
        &mut values16,
        PrecisionProfile::F16_STORAGE_F64_COMPUTE,
    )
    .expect("typed F16 forward");
    assert_eq!(frequencies16, expected16.frequencies);
    for (actual, expected) in values16.iter().zip(expected16.values.iter()) {
        let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
        assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
        assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
    }
}

#[test]
fn typed_paths_reject_profile_and_shape_mismatch() {
    let plan = SparseFftPlan::new(4, 1).expect("plan");
    let signal = vec![Complex32::new(1.0, 0.0); 4];
    let mut frequencies = Vec::new();
    let mut values = Vec::new();
    let err = plan
        .forward_typed_into(
            &signal,
            &mut frequencies,
            &mut values,
            PrecisionProfile::HIGH_ACCURACY_F64,
        )
        .expect_err("profile mismatch");
    assert!(matches!(
        err,
        ApolloError::Validation { field, .. } if field == "precision_profile"
    ));

    let mut output = vec![Complex32::new(0.0, 0.0); 4];
    let err = plan
        .inverse_typed_into(
            &[0, 1],
            &[Complex32::new(1.0, 0.0)],
            &mut output,
            PrecisionProfile::F32_STORAGE_F64_COMPUTE,
        )
        .expect_err("sparse shape mismatch");
    assert!(matches!(
        err,
        ApolloError::Validation { field, .. } if field == "sparse_values"
    ));
}

#[test]
fn leto_forward_matches_slice_reference() {
    let plan = SparseFftPlan::new(8, 2).expect("plan");
    let signal = exactly_sparse_signal();
    let input = leto::Array1::from_shape_vec([8], signal.clone()).expect("leto input");
    let expected = plan.forward(&signal).expect("slice forward");

    let actual = plan.forward_leto(input.view()).expect("leto forward");

    assert_eq!(actual.frequencies, expected.frequencies);
    for (actual, expected) in actual.values.iter().zip(expected.values.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_strided_forward_matches_slice_reference() {
    let plan = SparseFftPlan::new(8, 2).expect("plan");
    let signal = exactly_sparse_signal();
    let mut backing = Vec::with_capacity(16);
    for sample in signal.iter().copied() {
        backing.push(sample);
        backing.push(Complex64::new(1000.0, -1000.0));
    }
    let input = leto::Array1::from_shape_vec([16], backing).expect("leto input");
    let strided = input.view().slice(&[(0, 16, 2)]).expect("strided view");
    let expected = plan.forward(&signal).expect("slice forward");

    let actual = plan.forward_leto(strided).expect("leto forward");

    assert_eq!(actual.frequencies, expected.frequencies);
    for (actual, expected) in actual.values.iter().zip(expected.values.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn leto_inverse_matches_slice_reference() {
    let plan = SparseFftPlan::new(8, 2).expect("plan");
    let signal = exactly_sparse_signal();
    let spectrum = plan.forward(&signal).expect("forward");
    let expected = plan.inverse(&spectrum).expect("slice inverse");

    let actual = plan.inverse_leto(&spectrum).expect("leto inverse");
    let actual = actual.view();
    let actual = actual.as_slice().expect("contiguous leto output");

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn typed_leto_forward_and_inverse_match_typed_slice_reference() {
    let plan = SparseFftPlan::new(8, 2).expect("plan");
    let signal64 = exactly_sparse_signal();
    let signal32: Vec<Complex32> = signal64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect();
    let input = leto::Array1::from_shape_vec([8], signal32.clone()).expect("leto input");
    let mut expected_freq = Vec::new();
    let mut expected_values = Vec::new();
    plan.forward_typed_into(
        &signal32,
        &mut expected_freq,
        &mut expected_values,
        PrecisionProfile::F32_STORAGE_F64_COMPUTE,
    )
    .expect("typed slice forward");

    let actual = plan
        .forward_leto_typed(input.view(), PrecisionProfile::F32_STORAGE_F64_COMPUTE)
        .expect("typed leto forward");
    let actual_values_view = actual.values.view();
    let actual_values = actual_values_view
        .as_slice()
        .expect("contiguous leto values");

    assert_eq!(actual.frequencies, expected_freq);
    for (actual, expected) in actual_values.iter().zip(expected_values.iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }

    let values_array =
        leto::Array1::from_shape_vec([expected_values.len()], expected_values.clone())
            .expect("leto values");
    let actual_signal = plan
        .inverse_leto_typed(
            &expected_freq,
            values_array.view(),
            PrecisionProfile::F32_STORAGE_F64_COMPUTE,
        )
        .expect("typed leto inverse");
    let actual_signal_view = actual_signal.view();
    let actual_signal = actual_signal_view
        .as_slice()
        .expect("contiguous leto signal");
    let mut expected_signal = vec![Complex32::new(0.0, 0.0); plan.len()];
    plan.inverse_typed_into(
        &expected_freq,
        &expected_values,
        &mut expected_signal,
        PrecisionProfile::F32_STORAGE_F64_COMPUTE,
    )
    .expect("typed slice inverse");

    for (actual, expected) in actual_signal.iter().zip(expected_signal.iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }
}

proptest! {
    /// Exact K-sparse recovery: if a signal is K-sparse in the DFT domain,
    /// SparseFftPlan recovers the exact nonzero coefficients.
    ///
    /// Proof: in exact arithmetic, N-K non-support DFT bins are zero.
    /// The min-heap retains all K nonzero bins since their magnitudes exceed
    /// zero; no zero-magnitude bin can displace a nonzero bin (Theorem:
    /// Exact Recovery for K-Sparse Signals, module-level proof).
    #[test]
    fn k_sparse_roundtrip_recovers_exact_support(
        // Generate N as power-of-two in {8, 16, 32}
        n_exp in 3_u32..=5_u32,
        // Generate k nonzero DFT bins from the first k frequency indices
        k in 1_usize..=4_usize,
        // Real parts for k nonzero coefficients (magnitude >= 0.5 to clear threshold)
        re_parts in prop::collection::vec(0.5_f64..2.0_f64, 4),
    ) {
        let n = 1_usize << n_exp;
        let k_clamped = k.min(n / 2);
        // Build a K-sparse signal in frequency domain: set k DFT bins, IFFT to time domain.
        let mut freq_domain = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..k_clamped {
            freq_domain[i + 1] = Complex64::new(re_parts[i % 4], 0.0);
        }
        // IFFT to get the time-domain signal.
        let mut arr = Array1::from(freq_domain.clone());
        apollo_fft::ifft_1d_complex_inplace(&mut arr);
        let signal: Vec<Complex64> = arr.iter().copied().collect();

        let plan = SparseFftPlan::new(n, k_clamped).expect("plan");
        let spectrum = plan.forward(&signal).expect("forward");
        let dense = spectrum.to_dense();

        // Verify: the nonzero DFT coefficients in dense match the input freq_domain.
        for i in 0..k_clamped {
            let bin = i + 1;
            let expected_mag = re_parts[i % 4];
            prop_assert!(
                (dense[bin].norm() - expected_mag).abs() < 1.0e-9,
                "K-sparse recovery: bin {bin} magnitude {}, expected {}",
                dense[bin].norm(),
                expected_mag
            );
        }
    }

    /// Top-K selection optimality: retained bins have greater or equal energy
    /// than any alternative K-subset of the DFT coefficients.
    ///
    /// Proof: SparseFftPlan retains the K largest-magnitude DFT coefficients.
    /// For any alternative K-subset, Σ|alt_k|² ≤ Σ|retained_k|² by definition
    /// of top-K ordering (Theorem: Top-K Optimality, module-level proof).
    #[test]
    fn top_k_retained_energy_dominates_alternatives(
        signal_re in prop::collection::vec(-3.0_f64..3.0_f64, 8),
    ) {
        let n = 8;
        let k = 3;
        let signal: Vec<Complex64> = signal_re.iter()
            .map(|&r| Complex64::new(r, 0.0))
            .collect();
        let plan = SparseFftPlan::new(n, k).expect("plan");
        let spectrum = plan.forward(&signal).expect("forward");

        // Compute the full DFT to get all coefficients.
        let mut arr = Array1::from(signal.clone());
        fft_1d_complex_inplace(&mut arr);
        let full_dft: Vec<Complex64> = arr.iter().copied().collect();

        // Energy in retained K bins.
        let retained_energy: f64 = spectrum.frequencies.iter()
            .map(|&bin| full_dft[bin].norm_sqr())
            .sum();

        // Energy in an alternative K-subset: last K bins by index.
        let alt_bins: Vec<usize> = (n - k..n).collect();
        let alt_energy: f64 = alt_bins.iter()
            .map(|&bin| full_dft[bin].norm_sqr())
            .sum();

        // Retained energy must be >= any alternative K-subset energy.
        prop_assert!(
            retained_energy >= alt_energy - 1.0e-10,
            "top-K optimality violated: retained_energy={retained_energy}, alt_energy={alt_energy}"
        );
    }

    /// Retained coefficient values equal the DFT at those frequency indices.
    ///
    /// SparseFftPlan computes the full DFT then selects top-K. The retained
    /// (frequency, value) pairs must satisfy |value - FFT(signal)[frequency]| < 1e-9.
    #[test]
    fn retained_values_equal_dft_at_those_indices(
        signal_re in prop::collection::vec(-2.0_f64..2.0_f64, 8),
    ) {
        let n = 8;
        let k = 4;
        let signal: Vec<Complex64> = signal_re.iter()
            .map(|&r| Complex64::new(r, 0.0))
            .collect();
        let plan = SparseFftPlan::new(n, k).expect("plan");
        let spectrum = plan.forward(&signal).expect("forward");

        // Compute the full DFT independently.
        let mut arr = Array1::from(signal);
        fft_1d_complex_inplace(&mut arr);
        let full_dft: Vec<Complex64> = arr.iter().copied().collect();

        // Each retained value must match the DFT at that index.
        for (&bin, &val) in spectrum.frequencies.iter().zip(spectrum.values.iter()) {
            prop_assert!(
                (val - full_dft[bin]).norm() < 1.0e-9,
                "retained value at bin {bin}: got {val}, expected {}", full_dft[bin]
            );
        }
    }
}
