use super::*;

#[test]
fn validation_suite_produces_value_semantic_reports_and_satisfies_schema() {
    let report = run_validation_suite().expect("validation suite");

    // Value semantic assertions
    assert!(report.fft_cpu.roundtrip_max_abs_error <= CPU_ROUNDTRIP_LIMIT);
    assert!(report.fft_cpu.parseval_relative_error <= CPU_PARSEVAL_LIMIT);
    assert!(report.nufft.passed);
    assert!(report.external.published_references.passed);
    assert_eq!(report.external.published_references.attempted, 59);
    assert_eq!(report.external.dft.backend, "dft");
    assert_eq!(report.external.numpy.backend, "numpy");

    // Schema structure assertions
    let value = serde_json::to_value(&report).expect("serialize validation report");
    let object = value
        .as_object()
        .expect("validation report is a JSON object");

    for key in [
        "fft_cpu",
        "fft_gpu",
        "nufft",
        "external",
        "benchmarks",
        "environment",
    ] {
        assert!(object.contains_key(key), "missing top-level key {key}");
    }

    let fft_cpu = object["fft_cpu"]
        .as_object()
        .expect("fft_cpu is a JSON object");
    for key in [
        "roundtrip_max_abs_error",
        "parseval_relative_error",
        "stability_max_abs_delta",
        "non_finite_input_propagates",
        "passed",
        "precision_profiles",
    ] {
        assert!(fft_cpu.contains_key(key), "missing fft_cpu key {key}");
    }

    let external = object["external"]
        .as_object()
        .expect("external is a JSON object");
    for key in [
        "passed",
        "pyfftw_checkout_present",
        "dft",
        "numpy",
        "pyfftw",
        "robustness_passed",
        "precision_comparisons",
        "published_references",
    ] {
        assert!(external.contains_key(key), "missing external key {key}");
    }
}

#[test]
fn published_reference_suite_checks_computed_fixture_values() {
    let report = run_published_reference_suite().expect("published references");
    assert_eq!(report.attempted, 59);
    assert!(report.passed);
    for fixture in &report.fixtures {
        assert!(
            fixture.max_abs_error <= fixture.threshold,
            "{} exceeded threshold: {} > {}",
            fixture.fixture,
            fixture.max_abs_error,
            fixture.threshold
        );
        assert!(!fixture.reference.is_empty());
    }
}

#[test]
fn test_melinoe_zero_copy_boundary_policy_integration() {
    use melinoe::{brand_scope, Borrowed, CellCowExt, MelinoeCell, Retained};
    use std::borrow::Cow;

    let input_signal = [1.0, 2.0, 3.0, 4.0];
    brand_scope(|token| {
        let cells: Vec<MelinoeCell<'_, f64>> =
            input_signal.iter().copied().map(MelinoeCell::new).collect();

        // Zero-copy borrow boundary
        let borrowed = cells.borrow_cow_with(&token, Borrowed);
        assert!(matches!(borrowed, Cow::Borrowed(_)));
        assert_eq!(borrowed.as_ref(), &input_signal[..]);

        // Cloned retain boundary
        let retained = cells.borrow_cow_with(&token, Retained);
        assert!(matches!(retained, Cow::Owned(_)));
        assert_eq!(retained.as_ref(), &input_signal[..]);
    });
}

#[test]
fn test_moirai_melinoe_parallel_partitioning() {
    use melinoe::{brand_scope, MelinoeCell};
    use moirai::par_partition_for_each;

    let input_signal = [0.0f64; 16];
    brand_scope(|token| {
        let mut cells: Vec<MelinoeCell<'_, f64>> =
            input_signal.iter().copied().map(MelinoeCell::new).collect();

        // Run Moirai's parallel partitioning over the Melinoe cell region
        par_partition_for_each(cells.as_mut_slice(), 4, |start, mut shard| {
            for (j, slot) in shard.iter_mut().enumerate() {
                *slot = (start + j) as f64;
            }
        });

        let snap = token.share();
        for (i, cell) in cells.iter().enumerate() {
            assert_eq!(*cell.borrow(snap), i as f64);
        }
    });
}

#[test]
fn test_leto_validation_boundary() {
    use leto::{Array2, Storage};

    let leto = Array2::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("leto construction");

    assert_eq!(leto.shape(), [2, 3]);
    assert_eq!(leto.strides(), [3, 1]);
    assert_eq!(
        leto.storage().as_slice(),
        &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}
