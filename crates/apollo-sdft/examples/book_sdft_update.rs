//! Sliding DFT: one sliding update matches a fresh recomputation.
//!
//! A state seeded from an 8-sample window advances by one sample. The
//! tracked bins must equal the direct bins of the shifted window: the
//! recurrence carries no stale content.

extern crate apollo_sdft;

use apollo_sdft::SdftPlan;

fn main() {
    let plan = SdftPlan::new(8, 3).expect("valid sliding-DFT configuration");
    let window = [0.1, -0.4, 0.9, 0.2, -0.7, 0.5, 0.3, -0.1_f64];
    let mut state = plan.state_from_window(&window).expect("valid window");
    let next = 0.6_f64;
    let tracked = state.update(next).to_vec();
    let mut shifted = window[1..].to_vec();
    shifted.push(next);
    let fresh = plan.direct_bins(&shifted).expect("valid window");
    assert_eq!(
        tracked.len(),
        fresh.len(),
        "tracked and direct bin counts agree"
    );
    for (tracked, fresh) in tracked.iter().zip(fresh.iter()) {
        assert!(
            (tracked - fresh).norm() < 1e-9,
            "sliding update matches recomputation: {tracked} vs {fresh}"
        );
    }
    println!("ok");
}
