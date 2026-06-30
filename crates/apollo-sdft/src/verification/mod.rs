//! Verification modules for SDFT.

#[cfg(test)]
mod tests {
    use crate::{SdftError, SdftPlan};
    use approx::assert_abs_diff_eq;

    fn assert_bins_match(actual: &[eunomia::Complex64], expected: &[eunomia::Complex64]) {
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-10);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-10);
        }
    }

    #[test]
    fn plan_preserves_window_configuration() {
        let plan = SdftPlan::new(64, 16).expect("plan");
        assert_eq!(plan.config().window_len(), 64);
        assert_eq!(plan.config().bin_count(), 16);
        assert_eq!(plan.window_len(), 64);
        assert_eq!(plan.bin_count(), 16);
    }

    #[test]
    fn rejects_invalid_window_configuration() {
        assert_eq!(SdftPlan::new(0, 1).unwrap_err(), SdftError::EmptyWindow);
        assert_eq!(SdftPlan::new(8, 0).unwrap_err(), SdftError::EmptyBinCount);
        assert_eq!(
            SdftPlan::new(8, 9).unwrap_err(),
            SdftError::BinCountExceedsWindow
        );
    }

    #[test]
    fn initial_state_matches_direct_bins() {
        let plan = SdftPlan::new(8, 5).expect("plan");
        let window = [1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 0.0, 1.25];
        let state = plan.state_from_window(&window).expect("state");
        let direct = plan.direct_bins(&window).expect("direct");

        assert_eq!(state.updates(), 0);
        assert_eq!(state.window(), window);
        assert_bins_match(state.bins(), &direct);
    }

    #[test]
    fn sliding_updates_match_direct_dft_of_current_window() {
        let plan = SdftPlan::new(6, 6).expect("plan");
        let initial = [0.0, 1.0, -0.5, 0.25, 2.0, -1.0];
        let mut state = plan.state_from_window(&initial).expect("state");

        for (step, sample) in [0.75, -0.25, 1.5, -2.0, 0.125].into_iter().enumerate() {
            let bins = state.update(sample).to_vec();
            let current_window = state.window();
            let direct = plan.direct_bins(&current_window).expect("direct");
            assert_eq!(state.updates(), step + 1);
            assert_bins_match(&bins, &direct);
        }
    }

    #[test]
    fn zero_state_tracks_pushed_samples() {
        let plan = SdftPlan::new(4, 4).expect("plan");
        let mut state = plan.zero_state();
        state.update(1.0);
        state.update(2.0);
        state.update(3.0);
        state.update(4.0);

        assert_eq!(state.window(), vec![1.0, 2.0, 3.0, 4.0]);
        let direct = plan.direct_bins(&state.window()).expect("direct");
        assert_bins_match(state.bins(), &direct);
    }

    #[test]
    fn rejects_initial_window_length_mismatch() {
        let plan = SdftPlan::new(4, 3).expect("plan");
        assert_eq!(
            plan.state_from_window(&[1.0, 2.0]).unwrap_err(),
            SdftError::InitialWindowLengthMismatch
        );
        assert_eq!(
            plan.direct_bins(&[1.0, 2.0]).unwrap_err(),
            SdftError::InitialWindowLengthMismatch
        );
    }

    #[test]
    fn zero_state_has_zero_bins() {
        let plan = SdftPlan::new(8, 8).unwrap();
        let state = plan.zero_state();
        for bin in state.bins() {
            assert!(
                bin.norm() < 1e-14,
                "zero state must have zero bins, got {:?}",
                bin
            );
        }
    }

    #[test]
    fn update_counter_increments_correctly() {
        let plan = SdftPlan::new(4, 4).unwrap();
        let mut state = plan.zero_state();
        assert_eq!(state.updates(), 0);
        state.update(1.0);
        assert_eq!(state.updates(), 1);
        state.update(2.0);
        assert_eq!(state.updates(), 2);
    }

    use proptest::prelude::*;

    proptest! {
        /// After N updates from zero initial state, SDFT bins equal the DFT of those N samples.
        ///
        /// Proof: from all-zero initial state, after N updates feeding s[0]..s[N-1],
        /// the window is [s[0], ..., s[N-1]] and the recurrence correctly computes
        /// X_k = sum_{i=0}^{N-1} s[i] * exp(-2pi i k i / N).
        #[test]
        fn after_full_window_update_bins_equal_dft(
            samples in proptest::collection::vec(-1.0f64..1.0f64, 4..17usize),
        ) {
            let n = samples.len();
            let plan = SdftPlan::new(n, n).unwrap();
            let mut state = plan.zero_state();
            for &s in &samples {
                state.update(s);
            }
            let bins = state.bins();
            for k in 0..n {
                let angle = -std::f64::consts::TAU * k as f64 / n as f64;
                let expected: eunomia::Complex64 = samples.iter().enumerate()
                    .map(|(i, &x)| {
                        let phase = angle * i as f64;
                        eunomia::Complex64::new(x * phase.cos(), x * phase.sin())
                    })
                    .sum();
                let err = (bins[k] - expected).norm();
                prop_assert!(err < 1e-9, "bin {} mismatch: got {:?}, expected {:?}", k, bins[k], expected);
            }
        }
    }
}
