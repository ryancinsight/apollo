//! Mellin moment of a constant signal against its closed form.
//!
//! For `f(r) = 1` on `[1, 2]`, the moment `M(s) = int f(r) r^(s-1) dr` at
//! `s = 2` is `int_1^2 r dr = 3/2`. The trapezoid rule is exact on the
//! linear integrand, so the computed moment matches to rounding.

extern crate apollo_mellin;

use apollo_mellin::MellinPlan;

fn main() {
    let plan = MellinPlan::new(1_024, 1.0, 2.0).expect("valid scale config");
    let signal = vec![1.0_f64; 1_025];
    let moment = plan.moment(&signal, 1.0, 2.0, 2.0).expect("valid moment");
    assert!(
        (moment - 1.5).abs() < 1e-9,
        "M(2) of the unit signal on [1, 2] is 3/2, got {moment}"
    );
    println!("ok");
}
