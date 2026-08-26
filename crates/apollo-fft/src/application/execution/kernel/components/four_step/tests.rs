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
fn selection_starts_at_the_even_power_threshold() {
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

    let mut asymmetric = signal(8_192);
    let original = asymmetric.clone();
    assert!(!try_four_step::<f64, false, false>(
        &mut asymmetric,
        FOUR_STEP_THRESHOLD
    ));
    assert_eq!(
        asymmetric, original,
        "an asymmetric split must remain on the Stockham route"
    );

    let mut selected = signal(4_096);
    assert!(try_four_step::<f64, false, false>(
        &mut selected,
        FOUR_STEP_THRESHOLD
    ));

    let mut one_dimensional_below = signal(16_384);
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
