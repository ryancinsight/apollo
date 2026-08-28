//! Selection-contract coverage for the shared four-step route.

use super::try_four_step;
use crate::application::execution::kernel::tuning::{
    FOUR_STEP_THRESHOLD, ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD,
};
use eunomia::Complex64;

fn signal(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
        })
        .collect()
}

#[test]
fn selection_starts_at_the_threshold_for_either_split() {
    let mut below = signal(1_024);
    let original = below.clone();
    assert!(!try_four_step::<f64, false, false>(
        &mut below,
        FOUR_STEP_THRESHOLD
    ));
    assert_eq!(
        below, original,
        "a rejected route must not mutate its input"
    );

    // Asymmetric splits are admitted. They were excluded while their cost
    // was unmeasured; measuring it showed the exclusion left odd powers at
    // about 2.5x RustFFT beside even neighbours at 1.05x to 1.34x, so badly
    // that n = 2048 cost more than n = 4096 (gap_audit.md#odd-power-routing).
    let mut asymmetric = signal(8_192);
    assert!(try_four_step::<f64, false, false>(
        &mut asymmetric,
        FOUR_STEP_THRESHOLD
    ));

    let mut selected = signal(4_096);
    assert!(try_four_step::<f64, false, false>(
        &mut selected,
        FOUR_STEP_THRESHOLD
    ));

    // Contract, not constant: whatever the crossover is, an even power of two
    // one step below it stays on the Stockham route untouched.
    let below_len = ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD >> 2;
    let mut one_dimensional_below = signal(below_len);
    let original = one_dimensional_below.clone();
    assert!(!try_four_step::<f64, false, false>(
        &mut one_dimensional_below,
        ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD
    ));
    assert_eq!(one_dimensional_below, original);

    let mut one_dimensional_selected = signal(ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD);
    assert!(try_four_step::<f64, false, false>(
        &mut one_dimensional_selected,
        ONE_DIMENSIONAL_FOUR_STEP_THRESHOLD
    ));
}

#[test]
fn selected_normalized_round_trip_recovers_the_input() {
    let source = signal(4_096);
    let mut data = source.clone();
    assert!(try_four_step::<f64, false, false>(
        &mut data,
        FOUR_STEP_THRESHOLD
    ));
    assert!(try_four_step::<f64, true, true>(
        &mut data,
        FOUR_STEP_THRESHOLD
    ));

    let l1: f64 = source.iter().map(|value| value.re.hypot(value.im)).sum();
    let stages = f64::from(source.len().trailing_zeros());
    let bound = 32.0 * stages * (f64::EPSILON / 2.0) * l1;
    let worst = data
        .iter()
        .zip(&source)
        .map(|(actual, expected)| (actual.re - expected.re).hypot(actual.im - expected.im))
        .fold(0.0_f64, f64::max);
    assert!(
        worst <= bound,
        "normalized round trip differs by {worst:.3e} > {bound:.3e}"
    );
}
