use super::*;
use eunomia::{assert_abs_diff_eq, F64};

fn sample_window(window_len: usize) -> Vec<f64> {
    (0..window_len)
        .map(|index| {
            let t = F64::from_count(index).0;
            (t * 0.125).sin() - (t * 0.03125).cos()
        })
        .collect()
}

#[test]
fn moirai_parallel_direct_bins_match_serial_formula_at_threshold() {
    let window_len = 128;
    let bin_count = DIRECT_PAR_OP_THRESHOLD / window_len;
    let window = sample_window(window_len);
    let mut actual = vec![Complex64::new(0.0, 0.0); bin_count];

    direct_bins_into(&window, &mut actual).expect("parallel direct bins");

    for (bin, actual) in actual.iter().enumerate() {
        let expected = direct_bin_scalar(&window, window_len, bin);
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn hermes_direct_bin_matches_scalar_formula_at_threshold() {
    let window_len = HERMES_DIRECT_BIN_LEN_THRESHOLD;
    let window = sample_window(window_len);

    for bin in [0usize, 1, 17, 64, 127] {
        let actual = direct_bin_hermes(&window, window_len, bin);
        let expected = direct_bin_scalar(&window, window_len, bin);
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn direct_bin_weights_match_component_formula() {
    let n = 16;
    let bin = 5;
    let mut weights = vec![0.0; n];

    fill_direct_bin_weights(&mut weights, n, bin, DirectBinComponent::Imaginary);

    for (index, actual) in weights.iter().copied().enumerate() {
        let b = F64::from_count(bin).0;
        let i = F64::from_count(index).0;
        let len = F64::from_count(n).0;
        let angle = -std::f64::consts::TAU * b * i / len;
        assert_eq!(actual.to_bits(), angle.sin().to_bits(), "index={index}");
    }
}

#[test]
fn initialized_bins_match_the_direct_formula_at_a_non_power_of_two_length() {
    let window_len = 48;
    let window = sample_window(window_len);
    let table = modulation_table(window_len);
    let mut tracked = vec![ModulatedBin::new(0, window_len); window_len];
    let mut bins = vec![Complex64::new(0.0, 0.0); window_len];

    initialize_bins(&window, &table, &mut tracked, &mut bins).expect("initialize");

    for (bin, actual) in bins.iter().enumerate() {
        let expected = direct_bin_scalar(&window, window_len, bin);
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn initialize_rejects_a_table_of_another_length() {
    let window = sample_window(8);
    let table = modulation_table(6);
    let mut tracked = vec![ModulatedBin::new(0, 8); 3];
    let mut bins = vec![Complex64::new(0.0, 0.0); 3];

    assert_eq!(
        initialize_bins(&window, &table, &mut tracked, &mut bins).unwrap_err(),
        SdftError::InitialWindowLengthMismatch
    );
}

/// Advancing through a whole window returns every phase to where it
/// started, so the leaving and entering samples of every later update
/// read one table entry.
#[test]
fn a_full_turn_of_advances_returns_every_phase_exactly() {
    let window_len = 48;
    let table = modulation_table(window_len);
    let mut tracked: Vec<_> = (0..window_len)
        .map(|bin| ModulatedBin::new(bin, window_len))
        .collect();
    let start: Vec<usize> = tracked.iter().map(|state| state.phase).collect();

    for (bin, state) in tracked.iter_mut().enumerate() {
        for _ in 0..window_len {
            state.advance(bin, &table, 0.5);
        }
    }

    let end: Vec<usize> = tracked.iter().map(|state| state.phase).collect();
    assert_eq!(start, end);
}

/// A refreshed bin matches the bin advanced sample by sample, to rounding.
#[test]
fn refresh_agrees_with_the_advanced_sum() {
    let window_len = 48;
    let bin_count = 7;
    let table = modulation_table(window_len);
    let stream: Vec<f64> = sample_window(3 * window_len);
    let mut tracked: Vec<_> = (0..bin_count)
        .map(|bin| ModulatedBin::new(bin, window_len))
        .collect();
    let mut bins = vec![Complex64::new(0.0, 0.0); bin_count];
    initialize_bins(&stream[..window_len], &table, &mut tracked, &mut bins).expect("initialize");

    for (outgoing, incoming) in stream.iter().zip(&stream[window_len..]) {
        update_bins(&mut tracked, &mut bins, &table, *outgoing, *incoming);
    }
    let window = &stream[2 * window_len..];
    let (head, tail) = window.split_at(11);

    for bin in 0..bin_count {
        let advanced = bins[bin];
        let mut state = tracked[bin];
        let mut refreshed = Complex64::new(0.0, 0.0);
        refresh_bin(bin, &mut state, &mut refreshed, &table, (head, tail));
        assert_abs_diff_eq!(advanced.re, refreshed.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(advanced.im, refreshed.im, epsilon = 1.0e-12);
        let direct = direct_bin_scalar(window, window_len, bin);
        assert_abs_diff_eq!(refreshed.re, direct.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(refreshed.im, direct.im, epsilon = 1.0e-12);
    }
}

#[test]
fn moirai_parallel_update_bins_match_the_serial_advance_at_threshold() {
    let window_len = UPDATE_PAR_BIN_THRESHOLD;
    let table = modulation_table(window_len);
    let mut tracked: Vec<_> = (0..window_len)
        .map(|bin| ModulatedBin::new(bin, window_len))
        .collect();
    for (bin, state) in tracked.iter_mut().enumerate() {
        let b = F64::from_count(bin).0;
        state.sum = Complex64::new(b * 0.25, -b * 0.125);
    }
    let mut expected_tracked = tracked.clone();
    let mut bins = vec![Complex64::new(0.0, 0.0); window_len];

    update_bins(&mut tracked, &mut bins, &table, -0.5, 1.25);

    for (bin, (state, actual)) in expected_tracked.iter_mut().zip(&bins).enumerate() {
        state.advance(bin, &table, 1.25 - -0.5);
        assert_eq!(state.phase, tracked[bin].phase, "bin={bin}");
        let expected = state.value(bin, &table);
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}
